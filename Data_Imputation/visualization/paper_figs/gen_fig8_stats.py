"""Evaluate all ablation variants + KNN baseline + main model for figure 8.

Reads checkpoints from Data_Imputation/ablation/experiments/{variant}/best_model.pth
and evaluates each on the same set of artificially-masked samples.

Output: cache/fig8_stats.npz
"""
import sys
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
ABLATION_DIR = DATA_IMPUTATION_DIR / "ablation"
sys.path.insert(0, str(DATA_IMPUTATION_DIR))
sys.path.insert(0, str(ABLATION_DIR))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal  # main model
from models.fno_cbam_ablation import FNO_CBAM_Ablation     # ablation variants

KNN_FILLED_DIR = Path("/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled")
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Variants and where to find them
ABLATION_VARIANTS = {
    "fno_only":      dict(use_cbam=False),
    "cbam_basic":    dict(use_cbam=True),
    "cbam_boundary": dict(use_cbam=True),
    "cbam_full":     dict(use_cbam=True),
}
MAIN_MODEL_PATH = DATA_IMPUTATION_DIR / "experiments/jaxa_finetune/best_model.pth"

WINDOW_SIZE = 30
GPU_ID = 3
SERIES_ID = 0
GAUSSIAN_SIGMA = 1.0
SEED = 42

NUM_SAMPLES_PER_LEVEL = 15
MASK_LEVELS = [
    ("low",  0.30),
    ("mid",  0.55),
    ("high", 0.75),
]


# -- mask generator (same as fig4/6) --
class SquareMaskGenerator:
    def __init__(self, mask_ratio, min_size=10, max_size=50, seed=None):
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_mask):
        H, W = valid_mask.shape
        artificial = np.zeros((H, W), dtype=np.float32)
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return artificial
        target = int(valid_count * self.mask_ratio)
        current = 0
        ys, xs = np.where(valid_mask == 1)
        if len(ys) == 0:
            return artificial
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        attempts, max_attempts = 0, 2000
        while current < target and attempts < max_attempts:
            size = self.rng.integers(self.min_size, self.max_size + 1)
            if y_max - size < y_min or x_max - size < x_min:
                attempts += 1
                continue
            y0 = self.rng.integers(y_min, max(y_min + 1, y_max - size + 1))
            x0 = self.rng.integers(x_min, max(x_min + 1, x_max - size + 1))
            y1 = min(y0 + size, H)
            x1 = min(x0 + size, W)
            region = valid_mask[y0:y1, x0:x1].copy()
            artificial[y0:y1, x0:x1] = np.where(region == 1, 1.0,
                                                artificial[y0:y1, x0:x1])
            current = (artificial * valid_mask).sum()
            attempts += 1
        return artificial


def knn_inpaint_2d(sst, mask_to_fill, valid_ocean, k=20, power=2.0):
    """2D IDW inpainting baseline."""
    H, W = sst.shape
    src = (mask_to_fill == 0) & (valid_ocean == 1) & ~np.isnan(sst)
    sy, sx = np.where(src)
    if len(sy) == 0:
        return sst.copy()
    tree = cKDTree(np.column_stack([sy, sx]))
    src_vals = sst[sy, sx]
    ty, tx = np.where(mask_to_fill == 1)
    if len(ty) == 0:
        return sst.copy()
    distances, indices = tree.query(np.column_stack([ty, tx]),
                                    k=min(k, len(sy)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / (distances ** power + 1e-10)
    vals = src_vals[indices]
    filled_vals = (vals * weights).sum(axis=1) / weights.sum(axis=1)
    out = sst.copy()
    out[ty, tx] = filled_vals
    return out


def gauss_filter(sst, land_mask, sigma, fill_region=None):
    """fill_region 给定时只在该区(填充/masked区)写回滤波值，观测保留原值。"""
    valid = ~np.isnan(sst) & (land_mask == 0)
    if valid.sum() == 0:
        return sst
    mean_v = np.nanmean(sst)
    fill = sst.copy()
    fill[~valid] = mean_v
    out = gaussian_filter(fill, sigma=sigma)
    write = valid if fill_region is None else (valid & (fill_region == 1))
    return np.where(write, out, np.where(valid, sst, np.nan))


def predict_model(model, sst_seq, miss_seq, artificial_mask, norm_mean, norm_std, device):
    sst_input = sst_seq.copy()
    sst_input[-1] = np.where(artificial_mask > 0, norm_mean, sst_input[-1])
    mask_seq = miss_seq.copy().astype(np.float32)
    mask_seq[-1] = artificial_mask
    sst_norm = (sst_input - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)
    st = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mt = torch.from_numpy(mask_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(st, mt)
    pred_k = pred.squeeze().cpu().numpy() * norm_std + norm_mean
    full = sst_seq[-1].copy()
    full = np.where(artificial_mask > 0, pred_k, full)
    return full


def load_main_model(device):
    ckpt = torch.load(MAIN_MODEL_PATH, map_location=device, weights_only=False)
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("norm_mean", 299.9221), ckpt.get("norm_std", 2.6919)


def load_ablation_model(variant, device):
    ckpt_path = ABLATION_DIR / "experiments" / variant / "best_model.pth"
    if not ckpt_path.exists():
        return None, None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ABLATION_VARIANTS[variant]
    model = FNO_CBAM_Ablation(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16, use_cbam=cfg["use_cbam"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("norm_mean", 299.9221), ckpt.get("norm_std", 2.6919)


def boundary_pixels(mask, dilation=1):
    """Pixels within `dilation` of the mask edge (used to evaluate boundary error)."""
    from scipy.ndimage import binary_dilation, binary_erosion
    dilated = binary_dilation(mask.astype(bool), iterations=dilation)
    eroded = binary_erosion(mask.astype(bool), iterations=dilation)
    return dilated & ~eroded  # boundary band


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load all models
    print("\nLoading models...")
    main_model, m_mean, m_std = load_main_model(device)
    print(f"  main model: loaded (mean={m_mean:.2f}, std={m_std:.2f})")

    ablation_models = {}
    for v in ABLATION_VARIANTS:
        m, mu, sd = load_ablation_model(v, device)
        if m is None:
            print(f"  ablation/{v}: ckpt MISSING — will be skipped")
        else:
            ablation_models[v] = (m, mu, sd)
            print(f"  ablation/{v}: loaded")

    # Load data
    path = KNN_FILLED_DIR / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    print(f"\nLoading {path}")
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:]
        obs_all = f["original_obs_mask"][:]
        miss_all = f["original_missing_mask"][:]
        land = f["land_mask"][:]
        lat = f["latitude"][:]
        lon = f["longitude"][:]
    T = sst_all.shape[0]
    ocean = 1 - land

    rng = np.random.default_rng(SEED)
    start = WINDOW_SIZE - 1
    candidates = np.array([i for i in range(start, T) if obs_all[i].sum() > 5000])
    print(f"candidate frames: {len(candidates)}")

    # Per-method metric collector: dict[method] -> list of records
    methods = ["knn"] + list(ablation_models.keys()) + ["main"]
    print(f"Methods to evaluate: {methods}")
    all_records = []

    for name, level in MASK_LEVELS:
        print(f"\n== Level: {name} ({level:.0%}) ==")
        idx_sample = rng.choice(candidates,
                                size=min(NUM_SAMPLES_PER_LEVEL, len(candidates)),
                                replace=False)
        gen = SquareMaskGenerator(mask_ratio=level, seed=int(level * 1000))
        for idx in tqdm(idx_sample, desc=f"  {name}"):
            sst_seq = np.zeros((WINDOW_SIZE, *sst_all.shape[1:]), dtype=np.float32)
            miss_seq = np.zeros_like(sst_seq, dtype=np.float32)
            for t in range(WINDOW_SIZE):
                s = max(0, idx - (WINDOW_SIZE - 1) + t)
                sst_seq[t] = sst_all[s]
                miss_seq[t] = miss_all[s]
            obs_30 = obs_all[idx]
            eligible = obs_30 * ocean
            if eligible.sum() < 2000:
                continue
            mask = gen.generate(eligible.astype(np.float32))
            gt = sst_seq[-1].copy()
            eval_mask = (mask * eligible).astype(bool)
            bnd_mask = boundary_pixels(mask, dilation=2) & eval_mask
            actual_ratio = float(mask.sum() / (eligible.sum() + 1e-8))

            preds = {}
            # KNN
            knn_input = gt.copy()
            knn_input[mask > 0] = np.nan
            knn_pred = knn_inpaint_2d(knn_input, mask, ocean, k=20, power=2.0)
            knn_pred = gauss_filter(knn_pred, land, GAUSSIAN_SIGMA, fill_region=(mask > 0))
            preds["knn"] = knn_pred

            # Ablation variants
            for v, (mdl, mu, sd) in ablation_models.items():
                p = predict_model(mdl, sst_seq, miss_seq, mask, mu, sd, device)
                p = gauss_filter(p, land, GAUSSIAN_SIGMA, fill_region=(mask > 0))
                preds[v] = p

            # Main model
            p = predict_model(main_model, sst_seq, miss_seq, mask, m_mean, m_std, device)
            p = gauss_filter(p, land, GAUSSIAN_SIGMA, fill_region=(mask > 0))
            preds["main"] = p

            # Metrics per method (in K)
            for method_name, pred in preds.items():
                if eval_mask.sum() > 0:
                    diff = pred[eval_mask] - gt[eval_mask]
                    mae = float(np.abs(diff).mean())
                    rmse = float(np.sqrt((diff ** 2).mean()))
                    mx = float(np.abs(diff).max())
                else:
                    mae = rmse = mx = np.nan
                if bnd_mask.sum() > 0:
                    bnd_diff = pred[bnd_mask] - gt[bnd_mask]
                    bnd_mae = float(np.abs(bnd_diff).mean())
                else:
                    bnd_mae = np.nan
                all_records.append(dict(
                    method=method_name, level=name, idx=int(idx),
                    actual_ratio=actual_ratio,
                    mae=mae, rmse=rmse, max=mx, bnd_mae=bnd_mae,
                ))

    out = CACHE_DIR / "fig8_stats.npz"
    np.savez_compressed(out,
                        records=np.array(all_records, dtype=object))
    print(f"\nsaved -> {out}")
    print(f"Total records: {len(all_records)}")

    # Quick summary
    print("\nSummary (mean MAE per method × level):")
    print(f"{'method':<15}{'low':>10}{'mid':>10}{'high':>10}{'all':>10}")
    for m in methods:
        if m not in {r["method"] for r in all_records}:
            continue
        row = [f"{m:<15}"]
        for lv, _ in MASK_LEVELS + [("all", None)]:
            if lv == "all":
                mas = [r["mae"] for r in all_records if r["method"] == m]
            else:
                mas = [r["mae"] for r in all_records
                       if r["method"] == m and r["level"] == lv]
            if mas:
                row.append(f"{np.mean(mas):>10.4f}")
            else:
                row.append(f"{'-':>10}")
        print("".join(row))


if __name__ == "__main__":
    main()
