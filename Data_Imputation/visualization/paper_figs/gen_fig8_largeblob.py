"""Re-evaluate all methods (DL variants + traditional baselines) under a
LARGE-BLOB mask condition — square patches of size 80-200 px (~4-10°),
simulating realistic cloud bands rather than small scattered holes.

This is the eval that actually stress-tests interpolation methods, since
the interior of an 80+ pixel mask has no nearby anchor points.

Output: cache/fig8_largeblob_stats.npz with per-sample records (one record
per method per sample). Drop-in compatible with fig8_ablation.py via the
LARGE_BLOB_CACHE switch.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion

DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
ABLATION_DIR = DATA_IMPUTATION_DIR / "ablation"
BASELINES_DIR = DATA_IMPUTATION_DIR / "baselines"
PAPER_FIGS_CACHE = Path(__file__).parent / "cache"
PAPER_FIGS_CACHE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_IMPUTATION_DIR))
sys.path.insert(0, str(ABLATION_DIR))
sys.path.insert(0, str(BASELINES_DIR))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from models.fno_cbam_ablation import FNO_CBAM_Ablation
from dineof import dineof_fill
from simple_interp import linear_interp_2d, cubic_interp_2d

KNN_FILLED_DIR = Path("/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled")
MAIN_MODEL_PATH = DATA_IMPUTATION_DIR / "experiments/jaxa_finetune/best_model.pth"
ABLATION_VARIANTS = {
    "fno_only":      dict(use_cbam=False),
    "cbam_basic":    dict(use_cbam=True),
    "cbam_boundary": dict(use_cbam=True),
    "cbam_full":     dict(use_cbam=True),
}

WINDOW_SIZE = 30
GPU_ID = 3
SERIES_ID = 0
GAUSSIAN_SIGMA = 1.0
SEED = 42
NUM_SAMPLES_PER_LEVEL = 15

# Large-blob mask sizes: 80-200 pixels (~4-10° at 0.05° res)
MASK_LEVELS = [
    ("low",  0.30),
    ("mid",  0.55),
    ("high", 0.75),
]
MIN_BLOB_SIZE = 80
MAX_BLOB_SIZE = 200


class LargeBlobMaskGenerator:
    """Generate masks composed of LARGE square patches (80-200 px).

    Fewer patches per sample; harder interior reconstruction. Better
    proxy for real cloud cover than the small-square mask used elsewhere.
    """
    def __init__(self, mask_ratio, min_size=MIN_BLOB_SIZE, max_size=MAX_BLOB_SIZE, seed=None):
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_mask):
        H, W = valid_mask.shape
        art = np.zeros((H, W), dtype=np.float32)
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return art
        target = int(valid_count * self.mask_ratio)
        current = 0
        ys, xs = np.where(valid_mask == 1)
        if len(ys) == 0:
            return art
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        attempts, max_attempts = 0, 3000
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
            art[y0:y1, x0:x1] = np.where(region == 1, 1.0,
                                          art[y0:y1, x0:x1])
            current = (art * valid_mask).sum()
            attempts += 1
        return art


# ---------- shared helpers ----------

def knn_inpaint_2d(sst, mask_to_fill, valid_ocean, k=20, power=2.0):
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


def boundary_pixels(mask, dilation=2):
    dilated = binary_dilation(mask.astype(bool), iterations=dilation)
    eroded = binary_erosion(mask.astype(bool), iterations=dilation)
    return dilated & ~eroded


def predict_model(model, sst_seq, miss_seq, art_mask, norm_mean, norm_std, device):
    sst_input = sst_seq.copy()
    sst_input[-1] = np.where(art_mask > 0, norm_mean, sst_input[-1])
    mask_seq = miss_seq.copy().astype(np.float32)
    mask_seq[-1] = art_mask
    sst_norm = (sst_input - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)
    st = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mt = torch.from_numpy(mask_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(st, mt)
    pred_k = pred.squeeze().cpu().numpy() * norm_std + norm_mean
    full = sst_seq[-1].copy()
    full = np.where(art_mask > 0, pred_k, full)
    return full


def load_main_model(device):
    ckpt = torch.load(MAIN_MODEL_PATH, map_location=device, weights_only=False)
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64,
                                  width=64, depth=6,
                                  cbam_reduction_ratio=16).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("norm_mean", 299.9221), ckpt.get("norm_std", 2.6919)


def load_ablation(variant, device):
    p = ABLATION_DIR / "experiments" / variant / "best_model.pth"
    if not p.exists():
        return None, None, None
    ckpt = torch.load(p, map_location=device, weights_only=False)
    cfg = ABLATION_VARIANTS[variant]
    model = FNO_CBAM_Ablation(out_size=(451, 351), modes1=80, modes2=64,
                              width=64, depth=6, cbam_reduction_ratio=16,
                              use_cbam=cfg["use_cbam"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("norm_mean", 299.9221), ckpt.get("norm_std", 2.6919)


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Loading models...")
    main_model, m_mean, m_std = load_main_model(device)
    abl_models = {}
    for v in ABLATION_VARIANTS:
        m, mu, sd = load_ablation(v, device)
        if m is None:
            print(f"  ablation/{v}: MISSING")
        else:
            abl_models[v] = (m, mu, sd)
            print(f"  ablation/{v}: loaded")

    path = KNN_FILLED_DIR / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:]
        obs_all = f["original_obs_mask"][:]
        miss_all = f["original_missing_mask"][:]
        land = f["land_mask"][:]
        ts = [t.decode() if isinstance(t, bytes) else t for t in f["timestamps"][:]]
    T = sst_all.shape[0]
    ocean = 1 - land

    rng = np.random.default_rng(SEED)
    start = WINDOW_SIZE - 1
    candidates = np.array([i for i in range(start, T) if obs_all[i].sum() > 5000])

    all_records = []
    t_start = time.time()

    for name, level in MASK_LEVELS:
        print(f"\n== Large-blob {name} ({level:.0%}) ==")
        idx_sample = rng.choice(candidates,
                                size=min(NUM_SAMPLES_PER_LEVEL, len(candidates)),
                                replace=False)
        gen = LargeBlobMaskGenerator(mask_ratio=level, seed=int(level * 1000) + 7)
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
            art = gen.generate(eligible.astype(np.float32))
            if art.sum() == 0:
                continue
            gt = sst_seq[-1].copy()
            eval_mask = (art * eligible).astype(bool)
            bnd_mask = boundary_pixels(art, dilation=2) & eval_mask
            actual_ratio = float(art.sum() / (eligible.sum() + 1e-8))

            preds = {}
            # KNN-IDW baseline
            knn_input = gt.copy()
            knn_input[art > 0] = np.nan
            preds["knn"] = gauss_filter(
                knn_inpaint_2d(knn_input, art, ocean, k=20, power=2.0),
                land, GAUSSIAN_SIGMA, fill_region=(art > 0))
            # Linear / Cubic 2D
            src_v = (art == 0) & (ocean == 1) & ~np.isnan(gt)
            for nm, fn in (("linear_2d", linear_interp_2d),
                           ("cubic_2d", cubic_interp_2d)):
                lf = fn(np.where(src_v, gt, np.nan), src_v, ocean)
                comp = gt.copy()
                comp[art > 0] = lf[art > 0]
                preds[nm] = gauss_filter(comp, land, GAUSSIAN_SIGMA, fill_region=(art > 0))
            # DINEOF
            valid = (np.tile(ocean[None], (WINDOW_SIZE, 1, 1)) == 1) & ~np.isnan(sst_seq)
            valid[-1] = valid[-1] & (art == 0)
            data = np.where(valid, sst_seq, np.nan)
            d_filled = dineof_fill(data.astype(np.float32),
                                   valid.astype(np.uint8), k=20,
                                   max_iter=20, tol=5e-4, center=True)
            d_out = gt.copy()
            d_out[art > 0] = d_filled[-1][art > 0]
            preds["dineof"] = gauss_filter(d_out, land, GAUSSIAN_SIGMA, fill_region=(art > 0))
            # Ablation variants
            for v, (mdl, mu, sd) in abl_models.items():
                p = predict_model(mdl, sst_seq, miss_seq, art, mu, sd, device)
                preds[v] = gauss_filter(p, land, GAUSSIAN_SIGMA, fill_region=(art > 0))
            # Main model
            p = predict_model(main_model, sst_seq, miss_seq, art,
                              m_mean, m_std, device)
            preds["main"] = gauss_filter(p, land, GAUSSIAN_SIGMA, fill_region=(art > 0))

            for method, pred in preds.items():
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
                    method=method, level=name, idx=int(idx), ts=ts[idx],
                    actual_ratio=actual_ratio,
                    mae=mae, rmse=rmse, max=mx, bnd_mae=bnd_mae,
                ))

    out = PAPER_FIGS_CACHE / "fig8_largeblob_stats.npz"
    np.savez_compressed(out, records=np.array(all_records, dtype=object))
    print(f"\nSaved {len(all_records)} records → {out}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    # Summary
    methods_present = sorted({r["method"] for r in all_records})
    print("\nSummary (large-blob mask, mean MAE per method × level):")
    print(f"{'method':<15}{'low':>10}{'mid':>10}{'high':>10}{'all':>10}")
    for m in ["linear_2d", "cubic_2d", "dineof", "knn",
              "fno_only", "cbam_basic", "cbam_boundary", "cbam_full", "main"]:
        if m not in methods_present:
            continue
        row = [f"{m:<15}"]
        for lv, _ in MASK_LEVELS + [("all", None)]:
            if lv == "all":
                mas = [r["mae"] for r in all_records if r["method"] == m]
            else:
                mas = [r["mae"] for r in all_records
                       if r["method"] == m and r["level"] == lv]
            row.append(f"{np.mean(mas):>10.4f}" if mas else f"{'-':>10}")
        print("".join(row))


if __name__ == "__main__":
    main()
