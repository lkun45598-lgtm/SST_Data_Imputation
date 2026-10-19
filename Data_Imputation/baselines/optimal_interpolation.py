"""Optimal Interpolation (OI) / Ordinary Kriging baseline for SST gap-filling.

The classical oceanographic gap-filling standard (Reynolds OI, DIVA/DIVAnd family).
For each masked pixel we estimate the value from its K nearest OBSERVED ocean
neighbours via local ordinary kriging with an exponential covariance whose
variogram (nugget / sill / range) is fitted empirically from the observed field.

This is a pure 2-D SPATIAL interpolator: it uses only same-frame observations
(as operational SST-OI does), NOT the 30-day temporal window that the deep
baselines / DINEOF exploit — an asymmetry to state honestly in the paper.

Interface mirrors the project's KNN baseline (knn_inpaint_2d) so it drops into
the same evaluation harness:
    filled, vgm = oi_reconstruct(sst, mask_to_fill, valid_ocean)
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit


def fit_variogram(coords, vals, n_pairs=30000, max_lag=60.0, nbins=20, seed=0):
    """Empirical exponential variogram fit -> (nugget, sill, range) in pixel units.
    gamma(r) = nugget + sill * (1 - exp(-r / range)).
    """
    rng = np.random.default_rng(seed)
    n = len(vals)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    d = np.sqrt(((coords[i] - coords[j]) ** 2).sum(1))
    g = 0.5 * (vals[i] - vals[j]) ** 2
    keep = (d > 0) & (d <= max_lag)
    d, g = d[keep], g[keep]
    bins = np.linspace(0, max_lag, nbins + 1)
    which = np.digitize(d, bins)
    lag, semi = [], []
    for b in range(1, nbins + 1):
        sel = which == b
        if sel.sum() > 10:
            lag.append(d[sel].mean())
            semi.append(g[sel].mean())
    lag, semi = np.array(lag), np.array(semi)
    sill0 = float(semi.max()) if len(semi) else float(np.var(vals))
    rng0 = max_lag / 3.0

    def vmodel(r, nug, sill, rng_):
        return nug + sill * (1.0 - np.exp(-r / rng_))

    try:
        p, _ = curve_fit(vmodel, lag, semi, p0=[0.02 * sill0, sill0, rng0],
                         bounds=([0, 1e-6, 1.0], [sill0, 3 * sill0, max_lag * 3]),
                         maxfev=8000)
        nug, sill, rng_ = float(p[0]), float(p[1]), float(p[2])
    except Exception:
        nug, sill, rng_ = 0.02 * sill0, sill0, rng0
    # guard against a degenerate all-nugget fit (kills spatial structure)
    if sill < 1e-6:
        sill = max(sill0, 1e-3)
    return nug, sill, rng_


def oi_reconstruct(sst, mask_to_fill, valid_ocean, k=30, vgm=None, chunk=4000):
    """Local ordinary-kriging fill of mask_to_fill==1 pixels.

    Args:
        sst:          [H,W] field (queried only at observed pixels)
        mask_to_fill: [H,W] 1 = pixel to reconstruct
        valid_ocean:  [H,W] 1 = ocean (eligible source/target)
        k:            neighbours per target
        vgm:          (nugget, sill, range); fitted from the field if None
    Returns:
        (filled_field, vgm)
    """
    src = (mask_to_fill == 0) & (valid_ocean == 1) & ~np.isnan(sst)
    sy, sx = np.where(src)
    if len(sy) == 0:
        out = sst.copy()
        out[mask_to_fill == 1] = np.nan
        return out, (0.0, 1.0, 1.0)
    coords = np.column_stack([sy, sx]).astype(np.float64)
    vals = sst[sy, sx].astype(np.float64)
    if vgm is None:
        vgm = fit_variogram(coords, vals)
    nug, sill, rng_ = vgm

    def cov(r):
        return sill * np.exp(-r / rng_)

    tree = cKDTree(coords)
    ty, tx = np.where(mask_to_fill == 1)
    T = len(ty)
    out = sst.copy()
    if T == 0:
        return out, vgm
    tcoords = np.column_stack([ty, tx]).astype(np.float64)
    K = min(k, len(sy))
    di = np.arange(K)
    est = np.empty(T, dtype=np.float64)

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        q = tcoords[s:e]
        dist, ind = tree.query(q, k=K)
        if K == 1:
            dist, ind = dist[:, None], ind[:, None]
        ncoords = coords[ind]                              # (m,K,2)
        nvals = vals[ind]                                  # (m,K)
        diff = ncoords[:, :, None, :] - ncoords[:, None, :, :]
        pd = np.sqrt((diff ** 2).sum(-1))                  # (m,K,K)
        C = cov(pd)
        C[:, di, di] += nug + 1e-6                         # nugget + jitter
        c0 = cov(dist)                                     # (m,K)
        m = e - s
        A = np.zeros((m, K + 1, K + 1))
        A[:, :K, :K] = C
        A[:, :K, K] = 1.0
        A[:, K, :K] = 1.0
        b = np.zeros((m, K + 1, 1))
        b[:, :K, 0] = c0
        b[:, K, 0] = 1.0
        try:
            w = np.linalg.solve(A, b)[:, :, 0]             # (m,K+1)
            wk = w[:, :K]
        except np.linalg.LinAlgError:
            wk = np.full((m, K), 1.0 / K)
        est[s:e] = (wk * nvals).sum(1)

    out[ty, tx] = est
    return out, vgm
