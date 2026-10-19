"""Structure-aware metrics for SST reconstruction."""

import numpy as np
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity


def reference_robust_range(reference, valid_reference_mask, *,
                           lower_percentile=1.0, upper_percentile=99.0):
    """Return a method-independent robust SST range from valid reference pixels."""
    ref = np.asarray(reference, dtype=np.float64)
    valid_reference = np.asarray(valid_reference_mask, dtype=bool)

    if ref.ndim != 2 or valid_reference.shape != ref.shape:
        raise ValueError("reference and valid_reference_mask must be same-shape 2-D arrays")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    values = ref[valid_reference & np.isfinite(ref)]
    if values.size == 0:
        return np.nan
    low, high = np.percentile(values, [lower_percentile, upper_percentile])
    data_range = float(high - low)
    return data_range if data_range > 0 else np.nan


def masked_window_ssim(prediction, reference, evaluation_mask,
                       valid_reference_mask, *, win_size=7,
                       data_range=25.0):
    """Average standard local SSIM at eligible extra-mask centers.

    SSIM uses the complete regular window around each center. A center is
    eligible when it belongs to the artificial extra mask, its full window lies
    inside the array, and every window pixel has a finite prediction and a valid
    finite reference value. Pixels outside the extra mask may therefore provide
    context to a window centered inside the evaluation region.

    Returns:
        ``(mean_ssim, n_centers, n_candidates)``. The score is NaN when no
        center is eligible.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    valid_reference = np.asarray(valid_reference_mask, dtype=bool)

    if pred.ndim != 2:
        raise ValueError("prediction and reference must be 2-D arrays")
    if ref.shape != pred.shape or evaluation.shape != pred.shape or valid_reference.shape != pred.shape:
        raise ValueError("all inputs must have the same shape")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be an odd integer of at least 3")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if min(pred.shape) < win_size:
        return np.nan, 0, 0

    finite_reference = valid_reference & np.isfinite(ref)
    radius = win_size // 2
    full_window = np.zeros(pred.shape, dtype=bool)
    full_window[radius:-radius, radius:-radius] = True
    candidates = evaluation & finite_reference & full_window
    n_candidates = int(candidates.sum())

    valid_pair = finite_reference & np.isfinite(pred)
    kernel = np.ones((win_size, win_size), dtype=np.float64)
    valid_count = convolve(
        valid_pair.astype(np.float64), kernel, mode="constant", cval=0.0
    )
    eligible_centers = candidates & (valid_count == win_size ** 2)
    n_centers = int(eligible_centers.sum())
    if n_centers == 0:
        return np.nan, 0, n_candidates

    safe_pred = np.where(valid_pair, pred, 0.0).astype(np.float64, copy=False)
    safe_ref = np.where(valid_pair, ref, 0.0).astype(np.float64, copy=False)
    _, ssim_map = structural_similarity(
        safe_pred,
        safe_ref,
        win_size=win_size,
        data_range=float(data_range),
        full=True,
    )
    return float(np.mean(ssim_map[eligible_centers])), n_centers, n_candidates
