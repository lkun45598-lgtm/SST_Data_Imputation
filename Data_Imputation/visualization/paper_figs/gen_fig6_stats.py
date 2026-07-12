"""Aggregate statistics over many samples for figure 6.

Runs FNO-CBAM eval on ~45 samples across 3 mask levels and accumulates:
  - spatial_err_sum, spatial_err_count: per-pixel mean error map
  - err_samples_flat: subsampled error values (for histogram)
  - pred_samples / gt_samples: paired pred-vs-GT (for scatter)
  - sample_records: per-sample mask_ratio + MAE/RMSE (richer than eval_stats)
"""
import sys
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATA_IMPUTATION_DIR))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

# Paper "Ours" = deployed hour-00 model (experiments/jaxa_finetune).
KNN_FILLED_DIR = Path("/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled")
MODEL_PATH = DATA_IMPUTATION_DIR / "experiments/jaxa_finetune/best_model.pth"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
GPU_ID = 3
SERIES_ID = 0
GAUSSIAN_SIGMA = 1.0
SEED = 42

# More samples per level for richer statistics
NUM_SAMPLES_PER_LEVEL = 20
MASK_LEVELS = [
    ("low",  0.30),
    ("mid",  0.55),
    ("high", 0.75),
]

# Subsample pixels per sample so npz stays small
PIXELS_PER_SAMPLE = 4000


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
    full = sst_seq[-1].copy()
    full = np.where(artificial_mask > 0, pred_k, full)
    return full


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading model: {MODEL_PATH}")
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16,
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_mean = ckpt.get("norm_mean", 299.9221)
    norm_std = ckpt.get("norm_std", 2.6919)

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
    T, H, W = sst_all.shape
    ocean = 1 - land

    rng = np.random.default_rng(SEED)
    start = WINDOW_SIZE - 1
    candidates = np.array([i for i in range(start, T) if obs_all[i].sum() > 5000])
    print(f"candidate frames: {len(candidates)}")

    # Spatial accumulators
    spatial_err_sum = np.zeros((H, W), dtype=np.float64)
    spatial_err_count = np.zeros((H, W), dtype=np.int64)

    # Subsamples for histogram / scatter
    err_buffer = []
    pred_buffer = []
    gt_buffer = []
    sample_records = []

    for name, level in MASK_LEVELS:
        print(f"\n== Level: {name} (ratio={level:.2f}) ==")
        idx_sample = rng.choice(candidates,
                                size=min(NUM_SAMPLES_PER_LEVEL, len(candidates)),
                                replace=False)
        gen = SquareMaskGenerator(mask_ratio=level, seed=int(level * 1000))
        for idx in tqdm(idx_sample, desc=f"  {name}"):
            sst_seq = np.zeros((WINDOW_SIZE, H, W), dtype=np.float32)
            miss_seq = np.zeros((WINDOW_SIZE, H, W), dtype=np.float32)
            for t in range(WINDOW_SIZE):
                s = max(0, idx - (WINDOW_SIZE - 1) + t)
                sst_seq[t] = sst_all[s]
                miss_seq[t] = miss_all[s]
            obs_30 = obs_all[idx]
            eligible = obs_30 * ocean
            if eligible.sum() < 2000:
                continue
            mask = gen.generate(eligible.astype(np.float32))
            actual_ratio = float(mask.sum() / (eligible.sum() + 1e-8))
            gt = sst_seq[-1].copy()
            pred = predict_fno(model, sst_seq, miss_seq, mask,
                               norm_mean, norm_std, device)
            pred = gauss_filter(pred, land, GAUSSIAN_SIGMA, fill_region=(mask > 0))

            eval_mask = (mask * eligible).astype(bool)
            err = pred - gt
            mae = float(np.abs(err[eval_mask]).mean())
            rmse = float(np.sqrt((err[eval_mask] ** 2).mean()))

            # Spatial accumulator: error per pixel at masked positions
            valid_pix = eval_mask & ~np.isnan(pred) & ~np.isnan(gt)
            spatial_err_sum[valid_pix] += np.abs(err[valid_pix])
            spatial_err_count[valid_pix] += 1

            # Subsample pixels for histogram/scatter
            yy, xx = np.where(valid_pix)
            if len(yy) > 0:
                take = min(PIXELS_PER_SAMPLE, len(yy))
                pick = rng.choice(len(yy), size=take, replace=False)
                err_buffer.append(err[yy[pick], xx[pick]])
                pred_buffer.append(pred[yy[pick], xx[pick]])
                gt_buffer.append(gt[yy[pick], xx[pick]])

            sample_records.append(dict(
                level=name, idx=int(idx), ts=ts[idx],
                actual_ratio=actual_ratio, mae=mae, rmse=rmse,
            ))

    err_flat = np.concatenate(err_buffer) if err_buffer else np.array([])
    pred_flat = np.concatenate(pred_buffer) if pred_buffer else np.array([])
    gt_flat = np.concatenate(gt_buffer) if gt_buffer else np.array([])

    print(f"\nTotal sampled pixels: {len(err_flat):,}")
    print(f"Spatial error map: max_count_per_pix={spatial_err_count.max()}, "
          f"covered_pixels={(spatial_err_count > 0).sum():,}")

    # Save
    out = CACHE_DIR / "fig6_stats.npz"
    np.savez_compressed(
        out,
        spatial_err_sum=spatial_err_sum.astype(np.float32),
        spatial_err_count=spatial_err_count.astype(np.int32),
        err_flat=err_flat.astype(np.float32),
        pred_flat=pred_flat.astype(np.float32),
        gt_flat=gt_flat.astype(np.float32),
        sample_records=np.array(sample_records, dtype=object),
        land_mask=land.astype(np.uint8),
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
