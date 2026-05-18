"""Aggregate statistics for figure 7 (per-hour analysis).

Two evaluation parts:
  Part A: artificial-mask MAE/RMSE per hour, for H=00..08 (KNN data limitation)
  Part B: diurnal cycle from existing reconstruction archive (24 hours)
          - mean SST per hour over many days (model output vs raw obs at obs pixels)
          - point time series at (19°N, 115°E)
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import h5py
import netCDF4 as nc
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATA_IMPUTATION_DIR))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

# H=00 uses the base FNO_CBAM directory (no h00/ subdir); H=01..23 use hourly_data/h{HH}/
KNN_BASE_DIR = Path("/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled")
KNN_HOURLY_ROOT = DATA_IMPUTATION_DIR / "experiments/hourly_data"
HOURLY_MODEL_ROOT = DATA_IMPUTATION_DIR / "experiments"
RECON_NC_ROOT = Path("/data/sst_data/SST_Data_Imputation")
RAW_NC_ROOT = Path("/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3")
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
GPU_ID = 3
GAUSSIAN_SIGMA = 1.0
SEED = 42

# All 24 hours now available (H=00 in base dir, H=01..23 in hourly_data/h{HH}/)
EVAL_HOURS = list(range(0, 24))
SAMPLES_PER_HOUR = 10
SERIES_TO_USE = 0  # all hours have series_00 .. series_08; we use series_00 only

# Diurnal cycle settings: scan 2024-07 (a full month with good coverage)
DIURNAL_MONTH = "202407"
POINT_LAT = 19.0
POINT_LON = 115.0


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


def gauss_filter(sst, land_mask, sigma):
    valid = ~np.isnan(sst) & (land_mask == 0)
    if valid.sum() == 0:
        return sst
    mean_v = np.nanmean(sst)
    fill = sst.copy()
    fill[~valid] = mean_v
    out = gaussian_filter(fill, sigma=sigma)
    return np.where(valid, out, np.nan)


def load_hourly_model(hour, device):
    """Load model for hour H. H=00 uses base 'jaxa_finetune', H>=1 uses 'jaxa_finetune_h{HH}'."""
    if hour == 0:
        path = HOURLY_MODEL_ROOT / "jaxa_finetune/best_model.pth"
    else:
        path = HOURLY_MODEL_ROOT / f"jaxa_finetune_h{hour:02d}/best_model.pth"
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16,
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = ckpt.get("norm_mean", 299.9221)
    std = ckpt.get("norm_std", 2.6919)
    return model, mean, std


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


# ====== Part A: per-hour MAE/RMSE for H=00..08 ======
def eval_per_hour(device):
    print("\n===== Part A: per-hour MAE/RMSE (H=00..08, artificial mask 0.50) =====")
    per_hour_metrics = {}
    rng = np.random.default_rng(SEED)
    gen = SquareMaskGenerator(mask_ratio=0.50, seed=SEED)

    for H in EVAL_HOURS:
        if H == 0:
            knn_path = KNN_BASE_DIR / f"jaxa_knn_filled_{SERIES_TO_USE:02d}.h5"
        else:
            knn_path = KNN_HOURLY_ROOT / f"h{H:02d}" / f"jaxa_knn_filled_{SERIES_TO_USE:02d}.h5"
        if not knn_path.exists():
            print(f"  H={H:02d}: KNN data missing at {knn_path}, skipping")
            continue
        print(f"\n  H={H:02d}: load model + data")
        model, mean, std = load_hourly_model(H, device)
        with h5py.File(knn_path, "r") as f:
            sst_all = f["sst_data"][:]
            obs_all = f["original_obs_mask"][:]
            miss_all = f["original_missing_mask"][:]
            land = f["land_mask"][:]
        T = sst_all.shape[0]
        ocean = 1 - land
        start = WINDOW_SIZE - 1
        candidates = np.array([i for i in range(start, T) if obs_all[i].sum() > 5000])
        idx_sample = rng.choice(candidates, size=min(SAMPLES_PER_HOUR, len(candidates)),
                                replace=False)
        mae_list, rmse_list = [], []
        for idx in tqdm(idx_sample, desc=f"    H={H:02d}", leave=False):
            sst_seq = np.zeros((WINDOW_SIZE, *sst_all.shape[1:]), dtype=np.float32)
            miss_seq = np.zeros_like(sst_seq, dtype=np.float32)
            for t in range(WINDOW_SIZE):
                s = max(0, idx - (WINDOW_SIZE - 1) + t)
                sst_seq[t] = sst_all[s]
                miss_seq[t] = miss_all[s]
            eligible = obs_all[idx] * ocean
            if eligible.sum() < 2000:
                continue
            mask = gen.generate(eligible.astype(np.float32))
            gt = sst_seq[-1].copy()
            pred = predict_fno(model, sst_seq, miss_seq, mask, mean, std, device)
            pred = gauss_filter(pred, land, GAUSSIAN_SIGMA)
            eval_m = (mask * eligible).astype(bool)
            if eval_m.sum() == 0:
                continue
            diff = pred[eval_m] - gt[eval_m]
            mae_list.append(float(np.abs(diff).mean()))
            rmse_list.append(float(np.sqrt((diff ** 2).mean())))
        per_hour_metrics[H] = dict(
            n=len(mae_list),
            mae_mean=float(np.mean(mae_list)) if mae_list else np.nan,
            mae_std=float(np.std(mae_list)) if mae_list else np.nan,
            rmse_mean=float(np.mean(rmse_list)) if rmse_list else np.nan,
            rmse_std=float(np.std(rmse_list)) if rmse_list else np.nan,
        )
        print(f"    H={H:02d}: n={len(mae_list)}, MAE={per_hour_metrics[H]['mae_mean']:.4f}, "
              f"RMSE={per_hour_metrics[H]['rmse_mean']:.4f}")
        # free model
        del model
        torch.cuda.empty_cache()
    return per_hour_metrics


# ====== Part B: diurnal cycle from archive ======
def compute_diurnal_from_archive():
    """For each hour 0..23, average SST across the SCS region across all days in DIURNAL_MONTH.

    Returns:
      hourly_mean_sst:   [24] mean reconstructed SST in K over all days/ocean
      hourly_mean_obs:   [24] mean raw observed SST at the same days
      hourly_n_obs_pix:  [24] number of observed pixels averaged
      point_per_day:     {day_str: {hour: pred_K}} reconstruction at the point
      point_obs_per_day: {day_str: {hour: obs_K or NaN}}
    """
    month_dir = RECON_NC_ROOT / DIURNAL_MONTH
    if not month_dir.exists():
        print(f"WARN: archive not found {month_dir}")
        return None
    raw_month_dir = RAW_NC_ROOT / DIURNAL_MONTH

    print(f"\n===== Part B: diurnal cycle from archive {DIURNAL_MONTH} =====")

    day_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()])
    print(f"days available: {len(day_dirs)}")

    # Setup: read one file to get lat/lon
    first_nc = list(day_dirs[0].iterdir())[0]
    with nc.Dataset(first_nc) as ds:
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
    # Closest grid index to the point
    iy = int(np.argmin(np.abs(lat - POINT_LAT)))
    ix = int(np.argmin(np.abs(lon - POINT_LON)))
    print(f"point lat={lat[iy]:.3f} lon={lon[ix]:.3f}")

    # Per-hour accumulators
    sum_pred = np.zeros(24, dtype=np.float64)
    count_pred = np.zeros(24, dtype=np.int64)
    sum_obs = np.zeros(24, dtype=np.float64)
    count_obs = np.zeros(24, dtype=np.int64)
    point_pred = {h: [] for h in range(24)}
    point_obs = {h: [] for h in range(24)}

    for day_dir in tqdm(day_dirs, desc="diurnal aggregate"):
        for h in range(24):
            day_str = day_dir.name  # e.g., "15"
            ts_str = f"{DIURNAL_MONTH}{day_str}{h:02d}0000"
            pred_path = day_dir / f"{ts_str}.nc"
            raw_path = raw_month_dir / day_str / f"{ts_str}.nc"
            if not pred_path.exists():
                continue
            try:
                with nc.Dataset(pred_path) as ds:
                    pred = ds.variables["sea_surface_temperature"][:]
                    if pred.ndim == 3:
                        pred = pred[0]
                    if hasattr(pred, "mask"):
                        pred = np.where(pred.mask, np.nan, pred.filled())
                    pred = np.array(pred, dtype=np.float32)
            except Exception as e:
                continue
            valid = ~np.isnan(pred)
            if valid.sum() > 0:
                sum_pred[h] += float(np.nanmean(pred[valid]))
                count_pred[h] += 1
                # Point value
                if not np.isnan(pred[iy, ix]):
                    point_pred[h].append(float(pred[iy, ix]))

            # Raw observation
            if raw_path.exists():
                try:
                    with nc.Dataset(raw_path) as ds:
                        raw = ds.variables["sea_surface_temperature"][:]
                        if raw.ndim == 3:
                            raw = raw[0]
                        if hasattr(raw, "mask"):
                            raw = np.where(raw.mask, np.nan, raw.filled())
                        raw = np.array(raw, dtype=np.float32)
                except Exception:
                    raw = None
                if raw is not None:
                    valid_raw = ~np.isnan(raw)
                    if valid_raw.sum() > 0:
                        sum_obs[h] += float(np.nanmean(raw[valid_raw]))
                        count_obs[h] += 1
                    if not np.isnan(raw[iy, ix]):
                        point_obs[h].append(float(raw[iy, ix]))

    hourly_mean_pred = np.where(count_pred > 0, sum_pred / np.maximum(count_pred, 1), np.nan)
    hourly_mean_obs = np.where(count_obs > 0, sum_obs / np.maximum(count_obs, 1), np.nan)

    # Point summaries
    point_pred_mean = np.array([np.mean(point_pred[h]) if point_pred[h] else np.nan
                                for h in range(24)])
    point_pred_std = np.array([np.std(point_pred[h]) if point_pred[h] else np.nan
                               for h in range(24)])
    point_obs_mean = np.array([np.mean(point_obs[h]) if point_obs[h] else np.nan
                               for h in range(24)])
    point_obs_count = np.array([len(point_obs[h]) for h in range(24)])

    print(f"\nHourly mean pred (K): {[f'{v:.2f}' for v in hourly_mean_pred]}")
    print(f"Hourly obs count: {count_obs}")
    print(f"Point obs count per hour: {point_obs_count}")

    return dict(
        hourly_mean_pred=hourly_mean_pred,
        hourly_mean_obs=hourly_mean_obs,
        hourly_count_obs=count_obs,
        point_pred_mean=point_pred_mean,
        point_pred_std=point_pred_std,
        point_obs_mean=point_obs_mean,
        point_obs_count=point_obs_count,
        point_lat=float(lat[iy]),
        point_lon=float(lon[ix]),
    )


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    per_hour = eval_per_hour(device)
    diurnal = compute_diurnal_from_archive()

    out = CACHE_DIR / "fig7_stats.npz"
    np.savez_compressed(
        out,
        per_hour=np.array(per_hour, dtype=object),
        diurnal=np.array(diurnal, dtype=object) if diurnal is not None else None,
    )
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
