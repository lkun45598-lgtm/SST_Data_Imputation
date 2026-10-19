"""Held-out raw-ocean metrics for causal hourly v4 proxy evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from Data_Imputation.baselines.structural_metrics import (
    masked_window_ssim,
    reference_robust_range,
)


def _central_gradient_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, int]:
    """Compare central gradient magnitudes only on complete withheld stencils."""

    if min(reference.shape) < 3:
        return np.nan, np.nan, 0
    centers = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    count = int(centers.sum())
    if count == 0:
        return np.nan, np.nan, 0
    ref_x = (reference[1:-1, 2:] - reference[1:-1, :-2]) / 2.0
    ref_y = (reference[2:, 1:-1] - reference[:-2, 1:-1]) / 2.0
    pred_x = (prediction[1:-1, 2:] - prediction[1:-1, :-2]) / 2.0
    pred_y = (prediction[2:, 1:-1] - prediction[:-2, 1:-1]) / 2.0
    ref_magnitude = np.hypot(ref_x[centers], ref_y[centers])
    pred_magnitude = np.hypot(pred_x[centers], pred_y[centers])
    gradient_mae = float(np.mean(np.abs(pred_magnitude - ref_magnitude)))
    if count < 2 or np.std(ref_magnitude) == 0.0 or np.std(pred_magnitude) == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(ref_magnitude, pred_magnitude)[0, 1])
    return gradient_mae, correlation, count


def evaluate_withheld(
    reference: np.ndarray,
    prediction: np.ndarray,
    withheld_mask: np.ndarray,
    land_mask: np.ndarray,
    *,
    ssim_window: int = 7,
) -> Dict[str, float | int]:
    """Evaluate only artificially withheld finite ocean truth pixels.

    Gradient samples require the center and all four central-difference
    neighbours to be withheld, finite, and ocean pixels. SSIM is reported at
    withheld centers whose complete regular window is finite and valid; this is
    the same project-wide SSIM convention used for other reconstruction models.
    """

    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    withheld = np.asarray(withheld_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if (
        reference.ndim != 2
        or prediction.shape != reference.shape
        or withheld.shape != reference.shape
        or land.shape != reference.shape
    ):
        raise ValueError("reference, prediction, withheld_mask, and land_mask must be matching 2D arrays")
    valid = withheld & ~land & np.isfinite(reference) & np.isfinite(prediction)
    pixel_count = int(valid.sum())
    mae = float(np.mean(np.abs(prediction[valid] - reference[valid]))) if pixel_count else np.nan
    gradient_mae, gradient_correlation, gradient_count = _central_gradient_metrics(
        reference, prediction, valid
    )
    valid_reference = ~land & np.isfinite(reference)
    data_range = reference_robust_range(reference, valid_reference)
    if np.isfinite(data_range):
        ssim, ssim_centers, ssim_candidates = masked_window_ssim(
            prediction,
            reference,
            valid,
            valid_reference,
            win_size=int(ssim_window),
            data_range=float(data_range),
        )
    else:
        ssim, ssim_centers, ssim_candidates = np.nan, 0, 0
    return {
        "pixel_count": pixel_count,
        "mae": mae,
        "gradient_mae": gradient_mae,
        "gradient_correlation": gradient_correlation,
        "gradient_pixel_count": gradient_count,
        "ssim": float(ssim),
        "ssim_center_count": int(ssim_centers),
        "ssim_candidate_count": int(ssim_candidates),
    }
