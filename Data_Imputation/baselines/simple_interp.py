"""Simple 2D interpolation baselines for SST gap filling.

Operates on a single 2D snapshot (the day-30 target) — these methods don't
exploit temporal information, providing a lower-bound reference.

Implemented:
  - linear_interp_2d : scipy.interpolate.griddata with method='linear'
  - cubic_interp_2d  : scipy.interpolate.griddata with method='cubic'

For each, observed pixels (mask==1) anchor the interpolation; missing
pixels (mask==0) get filled. Land pixels are excluded.
"""
import numpy as np
from scipy.interpolate import griddata


def interp_2d(sst, valid, ocean, *, method="linear"):
    """Interpolate missing values in a 2D SST snapshot.

    Args:
        sst:   (H, W) float — SST with NaN at missing positions
        valid: (H, W) uint8/bool — 1 where SST is observed (anchor)
        ocean: (H, W) uint8/bool — 1 where pixel is ocean (target domain)
        method: 'linear' | 'cubic' | 'nearest'

    Returns:
        filled: (H, W) float — observed pixels untouched; missing ocean pixels
                filled by interpolation; land pixels = NaN.
    """
    valid = valid.astype(bool)
    ocean = ocean.astype(bool)
    H, W = sst.shape

    src = valid & ocean
    if src.sum() < 3:
        # not enough anchors → return NaN
        out = sst.copy().astype(np.float32)
        return out

    sy, sx = np.where(src)
    sv = sst[sy, sx]
    points = np.column_stack([sy, sx])

    # Targets: ocean pixels NOT in valid
    tgt = ocean & ~valid
    ty, tx = np.where(tgt)
    if len(ty) == 0:
        out = sst.copy().astype(np.float32)
        return out

    targets = np.column_stack([ty, tx])
    interp_vals = griddata(points, sv, targets, method=method)

    # Fallback for NaN (extrapolation regions) → nearest
    nan_mask = np.isnan(interp_vals)
    if nan_mask.any():
        nn_vals = griddata(points, sv, targets[nan_mask], method="nearest")
        interp_vals[nan_mask] = nn_vals

    out = sst.copy().astype(np.float32)
    out[ty, tx] = interp_vals
    # land stays NaN
    out[~ocean] = np.nan
    return out


def linear_interp_2d(sst, valid, ocean):
    return interp_2d(sst, valid, ocean, method="linear")


def cubic_interp_2d(sst, valid, ocean):
    return interp_2d(sst, valid, ocean, method="cubic")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 60, 80
    yy, xx = np.mgrid[:H, :W]
    truth = 20 + 0.05 * yy + 0.03 * xx
    ocean = np.ones_like(truth, dtype=bool)
    ocean[:5, :] = False    # fake "land" strip on top
    valid = (rng.random((H, W)) > 0.4) & ocean
    sst = np.where(valid, truth, np.nan)

    for method in ("linear", "cubic"):
        out = interp_2d(sst, valid, ocean, method=method)
        err = np.abs(out - truth)[ocean & ~valid]
        print(f"{method:>7s}: MAE={err.mean():.4f}, max={err.max():.4f}")
