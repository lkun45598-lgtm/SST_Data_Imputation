"""Generate evaluation cache for figures 4 and 6.

For each of 3 masking levels (low/mid/high), runs N samples on the test split:
  - GT  = jaxa_knn_filled value at originally-observed pixels (= true observation)
  - KNN baseline = 2D inverse-distance KNN inpainting on the artificially-masked input
  - FNO-CBAM = model prediction (with Gaussian σ=1.0 post-filter)

Saves:
  cache/eval_cache.npz
    cases: dict with case_low / case_mid / case_high
            each: arrays (gt, fno, knn, mask, land_mask, lat, lon, ts, metrics)
    stats: per-sample metrics for fig6 (errors, mask_ratio, etc.)
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

# /Data_Imputation/visualization/paper_figs/gen_eval_data.py -> parents[2] = Data_Imputation
DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DATA_IMPUTATION_DIR.parent
sys.path.insert(0, str(DATA_IMPUTATION_DIR))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator

# -- config --
# Paper "Ours" = the deployed hour-00 model (experiments/jaxa_finetune). The full
# system is a family of 24 per-hour fine-tuned models; hour 00 is the
# representative shown in Fig. 4-6/8. See Section 3.4 (per-hour fine-tuning).
KNN_FILLED_DIR = DATA_IMPUTATION_DIR / "experiments/hourly_data/h12"   # 与 realgap 图/部署一致 (h12)
MODEL_PATH = DATA_IMPUTATION_DIR / "experiments/jaxa_finetune_h12_realmask/best_model.pth"  # 新 real-cloud 模型
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
GPU_ID = 7                       # 0/5/6 在训基线, 3/1/2/4 在重训 -> 用 7 (仅 cjm PINN, 有余量)
SERIES_ID = 8                    # held-out validation series (train=0..7); 不再在训练序列 0 上评估
NUM_SAMPLES_PER_LEVEL = 12   # per masking level
GAUSSIAN_SIGMA = 1.0
SEED = 42

MASK_LEVELS = [
    ("low",  0.30),
    ("mid",  0.55),
    ("high", 0.75),
]


# -- mask generator (square blobs) --
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
        attempts, max_attempts = 0, 1500
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


# -- KNN baseline (2D inverse distance, k=20) --
def knn_inpaint_2d(sst, mask_to_fill, valid_ocean, k=20, power=2.0):
    """Fill positions where mask_to_fill==1 using IDW over surrounding valid pixels.

    Args:
        sst: [H, W] complete SST (will be queried only at mask==0)
        mask_to_fill: [H, W] 1=position to fill
        valid_ocean: [H, W] 1=ocean (eligible to be source or target)
    Returns:
        filled SST array (same as sst except at mask_to_fill positions)
    """
    H, W = sst.shape
    # Source pixels: ocean AND not in mask_to_fill (= unmasked + valid)
    src = (mask_to_fill == 0) & (valid_ocean == 1) & ~np.isnan(sst)
    sy, sx = np.where(src)
    if len(sy) == 0:
        return sst.copy()
    tree = cKDTree(np.column_stack([sy, sx]))
    src_vals = sst[sy, sx]

    ty, tx = np.where(mask_to_fill == 1)
    if len(ty) == 0:
        return sst.copy()
    distances, indices = tree.query(np.column_stack([ty, tx]), k=min(k, len(sy)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / (distances ** power + 1e-10)
    vals = src_vals[indices]
    filled_vals = (vals * weights).sum(axis=1) / weights.sum(axis=1)

    out = sst.copy()
    out[ty, tx] = filled_vals
    return out


# -- model wrapper --
def load_model(device):
    print(f"loading model: {MODEL_PATH}")
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16,
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = ckpt.get("norm_mean", 299.9221)
    std = ckpt.get("norm_std", 2.6919)
    print(f"  epoch={ckpt.get('epoch', 'N/A')}, mean={mean:.2f}, std={std:.2f}")
    return model, mean, std


def gauss_filter(sst, land_mask, sigma, fill_region=None):
    """高斯滤波。fill_region 给定时只在该区(模型填充区)写回滤波值，
    原始观测保留原值(真值不被滤波);None 时退回整场平滑(旧行为)。
    """
    valid = ~np.isnan(sst) & (land_mask == 0)
    if valid.sum() == 0:
        return sst
    mean_v = np.nanmean(sst)
    fill = sst.copy()
    fill[~valid] = mean_v
    out = gaussian_filter(fill, sigma=sigma)
    if fill_region is None:
        write = valid
    else:
        write = valid & (fill_region == 1)
    return np.where(write, out, np.where(valid, sst, np.nan))


def predict_fno(model, sst_seq, miss_seq, artificial_mask, norm_mean, norm_std, device):
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
    # Output composition
    full = sst_seq[-1].copy()
    full = np.where(artificial_mask > 0, pred_k, full)
    return full


def metric_at_mask(pred, gt, mask):
    valid = mask > 0
    if valid.sum() == 0:
        return dict(mae=np.nan, rmse=np.nan, max=np.nan, n=0)
    diff = (pred[valid] - gt[valid])  # already in K, error in K = error in °C
    return dict(
        mae=float(np.abs(diff).mean()),
        rmse=float(np.sqrt((diff**2).mean())),
        max=float(np.abs(diff).max()),
        n=int(valid.sum()),
    )


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model, norm_mean, norm_std = load_model(device)

    # Data
    path = KNN_FILLED_DIR / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    print(f"loading {path}")
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:]
        obs_all = f["original_obs_mask"][:]
        miss_all = f["original_missing_mask"][:]
        land = f["land_mask"][:]
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        ts = f["timestamps"][:]
        ts = [t.decode() if isinstance(t, bytes) else t for t in ts]
    T = sst_all.shape[0]
    ocean = 1 - land

    cases = {}
    stats_records = []  # for fig6

    rng = np.random.default_rng(SEED)
    start = WINDOW_SIZE - 1
    candidate_indices = np.array([i for i in range(start, T) if obs_all[i].sum() > 5000])
    print(f"candidate frames (>5000 obs pixels): {len(candidate_indices)}")

    # real-cloud-shaped artificial masks (与训练/realgap 一致, 取代方块)
    cloud_bank = build_cloud_bank([str(KNN_FILLED_DIR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]],
                                  max_donors=400, seed=7)

    for name, level in MASK_LEVELS:
        print(f"\n== Level: {name} (mask_ratio={level:.2f}) ==")
        # sample frames
        idx_sample = rng.choice(candidate_indices,
                                size=min(NUM_SAMPLES_PER_LEVEL, len(candidate_indices)),
                                replace=False)
        per_sample = []
        gen = RealCloudMaskGenerator(cloud_bank, seed=int(level * 1000))

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
            mask = gen.generate(eligible.astype(np.float32), target_ratio=level)

            gt = sst_seq[-1].copy()
            # Save the original observation mask so fig4 can show the TRUE
            # ground truth panel (sparse, observed-only) rather than the
            # KNN-completed field.
            obs_mask_30 = obs_30.astype(np.uint8)
            # FNO prediction
            fno = predict_fno(model, sst_seq, miss_seq, mask,
                              norm_mean, norm_std, device)
            fno = gauss_filter(fno, land, GAUSSIAN_SIGMA, fill_region=mask)
            # KNN baseline: input has NaN at masked positions, fill via 2D IDW
            knn_input = gt.copy()
            knn_input[mask > 0] = np.nan
            knn = knn_inpaint_2d(knn_input, mask, ocean, k=20, power=2.0)
            knn = gauss_filter(knn, land, GAUSSIAN_SIGMA, fill_region=mask)

            eval_mask = mask * eligible
            m_fno = metric_at_mask(fno, gt, eval_mask)
            m_knn = metric_at_mask(knn, gt, eval_mask)
            actual_ratio = float(mask.sum() / (eligible.sum() + 1e-8))
            rec = dict(
                idx=int(idx), ts=ts[idx], level=name, target_ratio=level,
                actual_ratio=actual_ratio,
                fno_mae=m_fno["mae"], fno_rmse=m_fno["rmse"], fno_max=m_fno["max"],
                knn_mae=m_knn["mae"], knn_rmse=m_knn["rmse"], knn_max=m_knn["max"],
            )
            per_sample.append((rec, gt, fno, knn, mask, eval_mask, obs_mask_30))
            stats_records.append(rec)

        # pick median-MAE sample as the representative case
        if not per_sample:
            continue
        maes = [p[0]["fno_mae"] for p in per_sample]
        med_pos = int(np.argsort(maes)[len(maes) // 2])
        rec, gt, fno, knn, mask, eval_mask, obs_mask_30 = per_sample[med_pos]
        print(f"  representative idx={rec['idx']} ts={rec['ts']} "
              f"fno_mae={rec['fno_mae']:.3f} knn_mae={rec['knn_mae']:.3f}")
        cases[name] = dict(
            gt=gt, fno=fno, knn=knn, mask=mask, eval_mask=eval_mask,
            obs_mask=obs_mask_30,           # TRUE observed pixels (no KNN fill)
            land=land, lat=lat, lon=lon, ts=rec["ts"], idx=rec["idx"],
            actual_ratio=rec["actual_ratio"],
            fno_metrics=dict(mae=rec["fno_mae"], rmse=rec["fno_rmse"], maxv=rec["fno_max"]),
            knn_metrics=dict(mae=rec["knn_mae"], rmse=rec["knn_rmse"], maxv=rec["knn_max"]),
        )

    # save
    out = CACHE_DIR / "eval_cache.npz"
    np.savez_compressed(out, cases=np.array(cases, dtype=object),
                        stats=np.array(stats_records, dtype=object))
    print(f"\nsaved -> {out}")

    # Also dump JSON of per-sample stats for quick eyeballing
    with open(CACHE_DIR / "eval_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2, default=str)
    print(f"saved -> {CACHE_DIR / 'eval_stats.json'}")


if __name__ == "__main__":
    main()
