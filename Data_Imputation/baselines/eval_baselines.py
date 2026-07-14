"""Evaluate traditional gap-filling baselines on the SAME test samples as
gen_fig8_stats.py, so they can be added directly to the comparison plot.

Methods evaluated:
  - linear_2d : 2D linear interpolation of day-30
  - cubic_2d  : 2D cubic spline interpolation of day-30
  - dineof    : DINEOF (temporal SVD-based) on the 30-day stack

Reads the same KNN-filled JAXA series and reproduces the same artificial
masks (same seed) so the resulting metrics are directly comparable to the
DL model results in cache/fig8_stats.npz.

Output: cache/baseline_stats.npz with per-sample records.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation, binary_erosion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IMPUTATION_DIR = PROJECT_ROOT / "Data_Imputation"
BASELINES_DIR = DATA_IMPUTATION_DIR / "baselines"
PAPER_FIGS_CACHE = DATA_IMPUTATION_DIR / "visualization/paper_figs/cache"
PAPER_FIGS_CACHE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASELINES_DIR))
from dineof import dineof_fill, dineof_with_cv
from simple_interp import linear_interp_2d, cubic_interp_2d

# ---- Settings matching gen_fig8_stats.py ----
KNN_FILLED_DIR = Path("/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled")
WINDOW_SIZE = 30
SERIES_ID = 8   # held-out validation series (train=0..7, val=8); must match gen_fig6_stats.py
GAUSSIAN_SIGMA = 1.0
SEED = 42
NUM_SAMPLES_PER_LEVEL = 15
MASK_LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]


# Mask generator IDENTICAL to gen_fig8_stats.py
class SquareMaskGenerator:
    def __init__(self, mask_ratio, min_size=10, max_size=50, seed=None):
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
            art[y0:y1, x0:x1] = np.where(region == 1, 1.0,
                                          art[y0:y1, x0:x1])
            current = (art * valid_mask).sum()
            attempts += 1
        return art


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


# ---- Baseline wrappers (operate on the SAME inputs the DL model sees) ----

def baseline_linear(sst_seq, mask_seq, art_mask, ocean):
    """Apply linear interp to day-30 with artificial mask."""
    gt = sst_seq[-1]
    src_valid = (art_mask == 0) & (ocean == 1) & ~np.isnan(gt)
    filled = linear_interp_2d(np.where(src_valid, gt, np.nan),
                              src_valid, ocean)
    out = gt.copy()
    out[art_mask > 0] = filled[art_mask > 0]
    return out


def baseline_cubic(sst_seq, mask_seq, art_mask, ocean):
    gt = sst_seq[-1]
    src_valid = (art_mask == 0) & (ocean == 1) & ~np.isnan(gt)
    filled = cubic_interp_2d(np.where(src_valid, gt, np.nan),
                             src_valid, ocean)
    out = gt.copy()
    out[art_mask > 0] = filled[art_mask > 0]
    return out


def baseline_dineof(sst_seq, mask_seq, art_mask, ocean, *, k=20):
    """DINEOF on the 30-day stack. Uses mean-centering and k=20 modes
    (best on 30-day windows in our sweep)."""
    T = sst_seq.shape[0]
    valid = (np.tile(ocean[None], (T, 1, 1)) == 1) & ~np.isnan(sst_seq)
    valid[-1] = valid[-1] & (art_mask == 0)
    data = np.where(valid, sst_seq, np.nan)
    filled = dineof_fill(data.astype(np.float32),
                         valid.astype(np.uint8),
                         k=k, max_iter=20, tol=5e-4, center=True)
    gt = sst_seq[-1]
    out = gt.copy()
    out[art_mask > 0] = filled[-1][art_mask > 0]
    return out


def metrics_at(eval_mask, pred, gt, bnd_mask=None):
    out = {}
    if eval_mask.sum() > 0:
        diff = pred[eval_mask] - gt[eval_mask]
        out["mae"] = float(np.abs(diff).mean())
        out["rmse"] = float(np.sqrt((diff ** 2).mean()))
        out["max"] = float(np.abs(diff).max())
    else:
        out["mae"] = out["rmse"] = out["max"] = np.nan
    if bnd_mask is not None and bnd_mask.sum() > 0:
        bnd_diff = pred[bnd_mask] - gt[bnd_mask]
        out["bnd_mae"] = float(np.abs(bnd_diff).mean())
    else:
        out["bnd_mae"] = np.nan
    return out


def main():
    t_start = time.time()
    path = KNN_FILLED_DIR / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    print(f"Loading {path}")
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:]
        obs_all = f["original_obs_mask"][:]
        miss_all = f["original_missing_mask"][:]
        land = f["land_mask"][:]
        ts = f["timestamps"][:]
        ts = [t.decode() if isinstance(t, bytes) else t for t in ts]
    T = sst_all.shape[0]
    ocean = 1 - land

    rng = np.random.default_rng(SEED)
    start = WINDOW_SIZE - 1
    candidates = np.array([i for i in range(start, T) if obs_all[i].sum() > 5000])

    methods = {
        "linear_2d": baseline_linear,
        "cubic_2d":  baseline_cubic,
        "dineof":    lambda sst, msk, art, oc: baseline_dineof(sst, msk, art, oc, k=20),
    }

    all_records = []

    for name, level in MASK_LEVELS:
        print(f"\n== Level {name} ({level:.0%}) ==")
        idx_sample = rng.choice(candidates,
                                size=min(NUM_SAMPLES_PER_LEVEL, len(candidates)),
                                replace=False)
        gen = SquareMaskGenerator(mask_ratio=level, seed=int(level * 1000))
        for idx in tqdm(idx_sample, desc=f"  {name}"):
            sst_seq = np.zeros((WINDOW_SIZE, *sst_all.shape[1:]), dtype=np.float32)
            for t in range(WINDOW_SIZE):
                s = max(0, idx - (WINDOW_SIZE - 1) + t)
                sst_seq[t] = sst_all[s]
            miss_seq = np.zeros_like(sst_seq, dtype=np.float32)
            for t in range(WINDOW_SIZE):
                s = max(0, idx - (WINDOW_SIZE - 1) + t)
                miss_seq[t] = miss_all[s]

            obs_30 = obs_all[idx]
            eligible = obs_30 * ocean
            if eligible.sum() < 2000:
                continue
            art = gen.generate(eligible.astype(np.float32))
            gt = sst_seq[-1].copy()
            eval_mask = (art * eligible).astype(bool)
            bnd_mask = boundary_pixels(art, dilation=2) & eval_mask
            actual_ratio = float(art.sum() / (eligible.sum() + 1e-8))

            for method_name, fn in methods.items():
                pred = fn(sst_seq, miss_seq, art, ocean)
                pred = gauss_filter(pred, land, GAUSSIAN_SIGMA, fill_region=(art > 0))
                m = metrics_at(eval_mask, pred, gt, bnd_mask)
                m.update(method=method_name, level=name, idx=int(idx),
                         ts=ts[idx], actual_ratio=actual_ratio)
                all_records.append(m)

    out = PAPER_FIGS_CACHE / "baseline_stats.npz"
    np.savez_compressed(out, records=np.array(all_records, dtype=object))
    print(f"\nSaved {len(all_records)} records → {out}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    # Summary
    print("\nSummary (mean MAE per method × level):")
    print(f"{'method':<12}{'low':>10}{'mid':>10}{'high':>10}{'all':>10}")
    for m in methods:
        row = [f"{m:<12}"]
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
