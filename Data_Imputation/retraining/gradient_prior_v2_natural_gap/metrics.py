"""Pixel/edge-weighted validation metrics and no-reference real-gap diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .losses import (
    _adjacent_differences,
    _as_b1hw,
    _upper_for_edges,
    natural_deployment_seam_masks,
    natural_edge_masks,
)


def supervised_error_terms(
    composed: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
    *,
    norm_std: float,
) -> dict[str, torch.Tensor]:
    valid = _as_b1hw(artificial_mask).bool()
    error = (composed - target) * float(norm_std)
    selected = valid.to(error.dtype)
    return {
        "absolute_sum": (torch.abs(error) * selected).sum(),
        "squared_sum": (error.square() * selected).sum(),
        "count": selected.sum(),
    }


def supervised_gradient_error_terms(
    composed: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return strict central gradient-magnitude error on artificial centres."""
    prediction = _as_b1hw(composed)
    reference = _as_b1hw(target)
    artificial = _as_b1hw(artificial_mask).bool()
    observed = _as_b1hw(original_observed_mask).bool()
    if not (
        prediction.shape
        == reference.shape
        == artificial.shape
        == observed.shape
    ):
        raise ValueError("gradient metric inputs must have equal BCHW shapes")
    if prediction.shape[-2] < 3 or prediction.shape[-1] < 3:
        return prediction.new_zeros(()), artificial.sum() * 0

    finite = torch.isfinite(prediction) & torch.isfinite(reference)
    stencil_valid = (
        observed[:, :, 1:-1, 1:-1]
        & observed[:, :, :-2, 1:-1]
        & observed[:, :, 2:, 1:-1]
        & observed[:, :, 1:-1, :-2]
        & observed[:, :, 1:-1, 2:]
        & finite[:, :, 1:-1, 1:-1]
        & finite[:, :, :-2, 1:-1]
        & finite[:, :, 2:, 1:-1]
        & finite[:, :, 1:-1, :-2]
        & finite[:, :, 1:-1, 2:]
    )
    selected = artificial[:, :, 1:-1, 1:-1] & stencil_valid

    pred_y = 0.5 * (
        prediction[:, :, 2:, 1:-1] - prediction[:, :, :-2, 1:-1]
    )
    pred_x = 0.5 * (
        prediction[:, :, 1:-1, 2:] - prediction[:, :, 1:-1, :-2]
    )
    ref_y = 0.5 * (
        reference[:, :, 2:, 1:-1] - reference[:, :, :-2, 1:-1]
    )
    ref_x = 0.5 * (
        reference[:, :, 1:-1, 2:] - reference[:, :, 1:-1, :-2]
    )
    error = torch.abs(torch.hypot(pred_y, pred_x) - torch.hypot(ref_y, ref_x))
    total = torch.where(selected, error * float(norm_std), 0.0).sum()
    return total, selected.sum()


def supervised_boundary_error_terms(
    composed: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    artificial = _as_b1hw(artificial_mask).bool()
    observed = _as_b1hw(original_observed_mask).bool()
    boundary_y = artificial[:, :, 1:, :] != artificial[:, :, :-1, :]
    boundary_x = artificial[:, :, :, 1:] != artificial[:, :, :, :-1]
    valid_y = (
        boundary_y
        & observed[:, :, 1:, :]
        & observed[:, :, :-1, :]
    )
    valid_x = (
        boundary_x
        & observed[:, :, :, 1:]
        & observed[:, :, :, :-1]
    )
    pred_y, pred_x = _adjacent_differences(composed)
    target_y, target_x = _adjacent_differences(target)
    error_y = torch.abs(pred_y - target_y) * float(norm_std)
    error_x = torch.abs(pred_x - target_x) * float(norm_std)
    total = (
        (error_y * valid_y.to(error_y.dtype)).sum()
        + (error_x * valid_x.to(error_x.dtype)).sum()
    )
    return total, valid_y.sum() + valid_x.sum()


def natural_exceedance_terms(
    composed: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_upper: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = _upper_for_edges(monthly_upper, composed)
    grad_y, grad_x = _adjacent_differences(composed)
    grad_y = torch.abs(grad_y) * float(norm_std)
    grad_x = torch.abs(grad_x) * float(norm_std)
    valid_y, valid_x = natural_edge_masks(natural_mask, land_mask)
    exceed_y = (grad_y > threshold) & valid_y
    exceed_x = (grad_x > threshold) & valid_x
    return exceed_y.sum() + exceed_x.sum(), valid_y.sum() + valid_x.sum()


def seam_exceedance_terms(
    composed: torch.Tensor,
    natural_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_upper: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = _upper_for_edges(monthly_upper, composed)
    grad_y, grad_x = _adjacent_differences(composed)
    grad_y = torch.abs(grad_y) * float(norm_std)
    grad_x = torch.abs(grad_x) * float(norm_std)
    valid_y, valid_x = natural_deployment_seam_masks(
        natural_mask, artificial_mask, land_mask
    )
    exceed_y = (grad_y > threshold) & valid_y
    exceed_x = (grad_x > threshold) & valid_x
    return exceed_y.sum() + exceed_x.sum(), valid_y.sum() + valid_x.sum()


def edge_difference_values(
    composed: torch.Tensor,
    edge_y_mask: torch.Tensor,
    edge_x_mask: torch.Tensor,
    *,
    norm_std: float,
) -> torch.Tensor:
    """Return finite physical adjacent differences selected by supplied edge masks."""
    grad_y, grad_x = _adjacent_differences(composed)
    valid_y = _as_b1hw(edge_y_mask).bool()
    valid_x = _as_b1hw(edge_x_mask).bool()
    if valid_y.shape != grad_y.shape or valid_x.shape != grad_x.shape:
        raise ValueError("edge masks must match their adjacent-difference fields")
    values_y = torch.abs(grad_y) * float(norm_std)
    values_x = torch.abs(grad_x) * float(norm_std)
    selected_y = values_y[valid_y & torch.isfinite(values_y)]
    selected_x = values_x[valid_x & torch.isfinite(values_x)]
    return torch.cat([selected_y, selected_x])


def local_predicted_outlier_terms(
    composed: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_upper: torch.Tensor,
    *,
    norm_std: float,
    window: int = 9,
    min_valid: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count natural-gap centres whose value departs from a local ocean median."""
    if window < 3 or window % 2 == 0:
        raise ValueError("local-outlier window must be odd and >= 3")
    if not 1 <= min_valid <= window * window:
        raise ValueError("min_valid is outside the local window")
    natural = _as_b1hw(natural_mask).bool()
    ocean = ~_as_b1hw(land_mask).bool()
    physical = composed * float(norm_std)
    radius = window // 2
    padded_values = F.pad(physical, (radius, radius, radius, radius), value=float("nan"))
    padded_ocean = F.pad(ocean.to(physical.dtype), (radius, radius, radius, radius))
    patches = F.unfold(padded_values, kernel_size=window)
    ocean_patches = F.unfold(padded_ocean, kernel_size=window) > 0
    patches = torch.where(
        ocean_patches,
        patches,
        torch.full_like(patches, float("nan")),
    )
    median = torch.nanmedian(patches, dim=1).values
    valid_count = ocean_patches.sum(dim=1)
    centres = natural.reshape(natural.shape[0], -1) & ocean.reshape(ocean.shape[0], -1)
    candidates = centres & (valid_count >= min_valid) & torch.isfinite(median)
    centre_values = physical.reshape(physical.shape[0], -1)
    threshold = monthly_upper.to(physical.device, physical.dtype).reshape(-1, 1)
    if threshold.shape[0] != physical.shape[0]:
        raise ValueError("monthly_upper batch size mismatch")
    outliers = (torch.abs(centre_values - median) > threshold) & candidates
    return outliers.sum(), candidates.sum()


class EpochMetricAccumulator:
    """Accumulate numerators and counts without batch-size weighting bias."""

    _OUTPUTS = {
        "mae": "absolute_error",
        "gradient_mae": "gradient_error",
        "boundary_mae": "boundary_error",
        "natural_exceedance_rate": "natural_exceedance",
        "seam_exceedance_rate": "seam_exceedance",
        "local_outlier_rate": "local_outlier",
        "low_energy_ratio": "low_energy_ratio",
    }

    def __init__(self) -> None:
        self.numerators: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    @staticmethod
    def _number(value: float | int | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def add(
        self,
        name: str,
        *,
        numerator: float | int | torch.Tensor,
        count: float | int | torch.Tensor,
    ) -> None:
        self.numerators[name] = self.numerators.get(name, 0.0) + self._number(numerator)
        self.counts[name] = self.counts.get(name, 0.0) + self._number(count)

    def _mean(self, name: str) -> float | None:
        count = self.counts.get(name, 0.0)
        if count <= 0:
            return None
        return self.numerators.get(name, 0.0) / count

    def finalize(self) -> dict[str, float | None]:
        result = {
            output: self._mean(source) for output, source in self._OUTPUTS.items()
        }
        squared = self._mean("squared_error")
        result["rmse"] = None if squared is None else math.sqrt(max(squared, 0.0))
        return result
