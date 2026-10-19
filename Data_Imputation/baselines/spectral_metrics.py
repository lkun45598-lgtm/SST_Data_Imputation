"""Radially-averaged power spectral density (PSD) comparison for reconstruction
structural fidelity, following the convention in SST reconstruction literature
(MAESSTRO, Ocean Sci. 2024; dADRSR Mediterranean SST super-resolution, Ocean Sci.
2024): compare pred vs truth energy per spatial wavenumber, rather than a single
scalar mean-gradient ratio (which cannot distinguish "correct fine structure"
from "wrong-but-similarly-sized noise" — e.g. DINEOF's over-sharp artifacts).
Note: those two papers report PSD as a log-log curve (visual comparison), not
as a single scalar — see hf_log_ratio below for why this codebase needs a
scalar summary and where that specific construction comes from.

hf_log_ratio (log-spectral distance, LSD, in the high-wavenumber band) is the
scalar summary used in place of grad_ratio: 0 = fine-scale (frontal) energy
matches truth on a log scale; >0 = excess high-freq energy (noisy/over-sharp);
<0 = missing high-freq content (over-smoothed). A *linear* power ratio was
tried first and rejected: empirically, gt high-wavenumber energy in a 128x128
patch spans ~7 orders of magnitude across samples (0.0002 to ~5000) because
real SST patches are physically low-pass dominated — a calm, texture-free
stretch of open ocean has almost no true high-frequency content, which is
normal, not a data-quality problem. Dividing by a near-zero true denominator
turns ordinary prediction noise into a ratio of 10^4-10^7 regardless of how
the ratio is aggregated (summed vs per-bin-averaged) or floored (a fixed
epsilon or an energy-fraction cutoff both either explode or discard the
majority of samples — verified empirically on this eval set).

LSD itself is a well-established metric — decades of use in speech/audio
super-resolution (see the Wikipedia entry "Log-spectral distance") — for
exactly this degenerate-denominator failure mode: log(near-zero) is a large
negative number, not a blow-up, so no sample needs to be discarded. It has
been carried over to 2D geophysical-field super-resolution at least once
(a spatiotemporal-diffusion climate paper, arXiv:2604.21903), whose authors
explicitly flag it as "uncommon in climate science." Be accurate in the paper
about that provenance: this is an audio-derived metric adapted here to solve
a numerical problem, not an SST-reconstruction-literature convention — cite it
as such rather than implying it is standard practice for this domain.
"""
import numpy as np


def _extract_patch(field, cy, cx, size):
    H, W = field.shape
    half = size // 2
    y0 = max(0, min(cy - half, H - size))
    x0 = max(0, min(cx - half, W - size))
    return field[y0:y0 + size, x0:x0 + size]


def radial_psd(patch, valid, min_valid_frac=0.5):
    """2D FFT power spectrum of `patch`, radially averaged into wavenumber bins.
    Invalid pixels (land/NaN) are filled with the patch mean before windowing.
    Returns None if too much of the patch is invalid (unreliable spectrum)."""
    if valid.sum() < valid.size * min_valid_frac:
        return None
    mean_v = np.nanmean(patch[valid])
    filled = np.where(valid, patch, mean_v).astype(np.float64)
    filled -= filled.mean()
    H, W = filled.shape
    window = np.outer(np.hanning(H), np.hanning(W))
    F = np.fft.fftshift(np.fft.fft2(filled * window))
    psd2d = np.abs(F) ** 2
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    nbins = min(H, W) // 2
    return np.array([psd2d[r == i].mean() for i in range(nbins)])


def psd_ratio_at_mask(pred, gt, ocean_valid, art_mask, patch_size=128, hf_frac=0.25,
                       log_eps=1e-6):
    """Compare radial PSD of pred vs gt in a fixed-size patch centered on the
    artificial-mask centroid. Returns (radial_ratio, hf_log_ratio) or (None, nan)
    if the patch can't be reliably analyzed.

    hf_frac: fraction of the highest wavenumber bins summed into the scalar
    hf_log_ratio (fine-scale/frontal fidelity summary).

    hf_log_ratio = log10(sum(pred power in HF bins) + log_eps)
                 - log10(sum(gt power in HF bins) + log_eps)
    log_eps=1e-6 is set below the smallest observed true HF-band energy on this
    eval set (~2e-4) so it never dominates a genuine (if small) signal, while
    still preventing log(0). `ratio` (the per-bin linear curve) is kept only
    for optional plotting/diagnostics — do not aggregate it directly, it has
    the same degenerate-denominator problem hf_log_ratio was built to avoid.
    """
    ys, xs = np.where(art_mask > 0)
    if len(ys) == 0:
        return None, np.nan
    cy, cx = int(ys.mean()), int(xs.mean())
    p_pred = _extract_patch(pred, cy, cx, patch_size)
    p_gt = _extract_patch(gt, cy, cx, patch_size)
    p_valid = _extract_patch(ocean_valid, cy, cx, patch_size).astype(bool) & np.isfinite(p_gt) & np.isfinite(p_pred)

    psd_pred = radial_psd(p_pred, p_valid)
    psd_gt = radial_psd(p_gt, p_valid)
    if psd_pred is None or psd_gt is None:
        return None, np.nan

    ratio = psd_pred / (psd_gt + 1e-9)   # per-bin curve, diagnostics/plotting only
    nbins = len(ratio)
    hf_start = int(nbins * (1 - hf_frac))
    pred_hf = psd_pred[hf_start:].sum()
    gt_hf = psd_gt[hf_start:].sum()
    hf_log_ratio = float(np.log10(pred_hf + log_eps) - np.log10(gt_hf + log_eps))
    return ratio, hf_log_ratio


def semivariogram(field, valid, max_lag=40, n_lags=20, n_pairs=20000, rng=None):
    """Empirical semivariogram gamma(d) = 0.5 * mean((field[i]-field[j])^2) over
    pixel pairs at distance d, restricted to `valid` pixels. Used to compare the
    spatial-structure statistics of a withheld/filled region against truth
    (or, for real no-reference cloud gaps, against a clear-sky reference
    distribution) — complementary to PSD: same physical content (Wiener-
    Khinchin), but a real-space diagnostic reviewers in the geostatistics/OI
    tradition (this repo's own oi_reconstruct is a kriging method) expect.

    Returns (lags, gamma) with `nbins=n_lags` bins from 0 to max_lag (pixels).
    Subsamples n_pairs random valid-pixel pairs per call for tractability —
    exact on small patches, approximate on large ones.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ys, xs = np.where(valid)
    n = len(ys)
    if n < 20:
        return None, None
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    dy = ys[i] - ys[j]
    dx = xs[i] - xs[j]
    d = np.sqrt(dy ** 2 + dx ** 2)
    keep = (d > 0) & (d <= max_lag)
    d = d[keep]
    diff2 = (field[ys[i][keep], xs[i][keep]] - field[ys[j][keep], xs[j][keep]]) ** 2

    bin_edges = np.linspace(0, max_lag, n_lags + 1)
    bin_idx = np.digitize(d, bin_edges) - 1
    lags = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    gamma = np.full(n_lags, np.nan)
    for b in range(n_lags):
        sel = bin_idx == b
        if sel.sum() >= 10:
            gamma[b] = 0.5 * np.nanmean(diff2[sel])
    return lags, gamma
