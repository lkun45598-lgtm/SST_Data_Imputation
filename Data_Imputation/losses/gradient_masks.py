"""Validity masks for strict first-order SST gradient comparisons.

Gradient samples are only defined when the complete reference stencil is
available.  In particular, no one-sided fallback is used at land, missing,
or array boundaries.
"""

import numpy as np
import torch


def _as_bchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim == 4:
        return mask
    raise ValueError(f"expected a (B,H,W) or (B,1,H,W) mask, got {tuple(mask.shape)}")


def torch_gradient_edge_masks(valid_ref_mask: torch.Tensor,
                              supervision_mask: torch.Tensor):
    """Return strict valid masks for horizontal and vertical adjacent pairs.

    Both endpoints must be valid reference ocean pixels, and at least one
    endpoint must be in the artificial supervision mask.
    """
    valid = _as_bchw(valid_ref_mask) > 0
    supervised = _as_bchw(supervision_mask) > 0
    pair_y = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    pair_x = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    sup_y = supervised[:, :, 1:, :] | supervised[:, :, :-1, :]
    sup_x = supervised[:, :, :, 1:] | supervised[:, :, :, :-1]
    return (pair_y & sup_y).to(dtype=valid_ref_mask.dtype), (pair_x & sup_x).to(dtype=valid_ref_mask.dtype)


def torch_boundary_edge_masks(artificial_mask: torch.Tensor,
                              valid_ref_mask: torch.Tensor):
    """Return strict valid masks for mask-transition adjacent pairs."""
    artificial = _as_bchw(artificial_mask) > 0
    valid = _as_bchw(valid_ref_mask) > 0
    boundary_y = artificial[:, :, 1:, :] != artificial[:, :, :-1, :]
    boundary_x = artificial[:, :, :, 1:] != artificial[:, :, :, :-1]
    pair_y = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    pair_x = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    return (boundary_y & pair_y).to(dtype=valid_ref_mask.dtype), (boundary_x & pair_x).to(dtype=valid_ref_mask.dtype)


def central_gradient_magnitude(field: np.ndarray, valid_ref_mask: np.ndarray):
    """Compute central gradient magnitude and its strict validity mask.

    A center is valid only if it and all four axial neighbors are valid
    reference pixels with finite field values.  The outermost array boundary
    is always invalid, so ``np.gradient``'s one-sided behavior is avoided.
    """
    values = np.asarray(field, dtype=np.float32)
    ref = np.asarray(valid_ref_mask, dtype=bool)
    if values.ndim != 2 or ref.shape != values.shape:
        raise ValueError("field and valid_ref_mask must be 2-D arrays of equal shape")

    finite = np.isfinite(values)
    valid = np.zeros(values.shape, dtype=bool)
    if values.shape[0] < 3 or values.shape[1] < 3:
        return np.full(values.shape, np.nan, dtype=np.float32), valid

    center = ref[1:-1, 1:-1] & finite[1:-1, 1:-1]
    stencil = (
        ref[:-2, 1:-1] & ref[2:, 1:-1] &
        ref[1:-1, :-2] & ref[1:-1, 2:] &
        finite[:-2, 1:-1] & finite[2:, 1:-1] &
        finite[1:-1, :-2] & finite[1:-1, 2:]
    )
    valid[1:-1, 1:-1] = center & stencil

    gy = (values[2:, 1:-1] - values[:-2, 1:-1]) * 0.5
    gx = (values[1:-1, 2:] - values[1:-1, :-2]) * 0.5
    magnitude = np.full(values.shape, np.nan, dtype=np.float32)
    magnitude[1:-1, 1:-1] = np.where(valid[1:-1, 1:-1], np.hypot(gy, gx), np.nan)
    return magnitude, valid
