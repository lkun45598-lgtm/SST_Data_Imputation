"""Mask-strict supervised and no-reference losses for real JAXA cloud gaps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _as_b1hw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor
    raise ValueError(f"expected (B,H,W) or (B,1,H,W), got {tuple(tensor.shape)}")


def _mask(mask: torch.Tensor) -> torch.Tensor:
    return _as_b1hw(mask).bool()


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected = valid.to(dtype=values.dtype)
    count = selected.sum()
    loss = (values * selected).sum() / count.clamp_min(1.0)
    return loss, count


def output_composition(
    raw_prediction: torch.Tensor,
    last_input: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    land_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve original input outside reconstruction ocean and on land."""
    reconstruction = _mask(reconstruction_mask)
    land = _mask(land_mask)
    if raw_prediction.shape != last_input.shape or raw_prediction.shape != reconstruction.shape:
        raise ValueError("prediction, last input, and masks must have equal BCHW shapes")
    use_prediction = reconstruction & ~land
    return torch.where(use_prediction, raw_prediction, last_input)


def supervised_mse_loss(
    composed: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    artificial = _mask(artificial_mask)
    return _masked_mean((composed - target).square(), artificial)


def _adjacent_differences(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return field[:, :, 1:, :] - field[:, :, :-1, :], field[:, :, :, 1:] - field[:, :, :, :-1]


def _supervised_edge_masks(
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    artificial = _mask(artificial_mask)
    observed = _mask(original_observed_mask)
    valid_y = observed[:, :, 1:, :] & observed[:, :, :-1, :]
    valid_x = observed[:, :, :, 1:] & observed[:, :, :, :-1]
    touches_y = artificial[:, :, 1:, :] | artificial[:, :, :-1, :]
    touches_x = artificial[:, :, :, 1:] | artificial[:, :, :, :-1]
    return valid_y & touches_y, valid_x & touches_x


def supervised_gradient_loss(
    composed: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_y, pred_x = _adjacent_differences(composed)
    target_y, target_x = _adjacent_differences(target)
    mask_y, mask_x = _supervised_edge_masks(artificial_mask, original_observed_mask)
    errors = torch.cat(
        [torch.abs(pred_y - target_y).reshape(-1), torch.abs(pred_x - target_x).reshape(-1)]
    )
    valid = torch.cat([mask_y.reshape(-1), mask_x.reshape(-1)])
    return _masked_mean(errors, valid)


def supervised_boundary_loss(
    composed: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    artificial = _mask(artificial_mask)
    observed = _mask(original_observed_mask)
    valid_y = observed[:, :, 1:, :] & observed[:, :, :-1, :]
    valid_x = observed[:, :, :, 1:] & observed[:, :, :, :-1]
    boundary_y = artificial[:, :, 1:, :] != artificial[:, :, :-1, :]
    boundary_x = artificial[:, :, :, 1:] != artificial[:, :, :, :-1]
    pred_y, pred_x = _adjacent_differences(composed)
    target_y, target_x = _adjacent_differences(target)
    errors = torch.cat(
        [torch.abs(pred_y - target_y).reshape(-1), torch.abs(pred_x - target_x).reshape(-1)]
    )
    valid = torch.cat(
        [(valid_y & boundary_y).reshape(-1), (valid_x & boundary_x).reshape(-1)]
    )
    return _masked_mean(errors, valid)


def natural_edge_masks(
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    natural = _mask(natural_mask)
    ocean = ~_mask(land_mask)
    natural_ocean = natural & ocean
    mask_y = natural_ocean[:, :, 1:, :] & natural_ocean[:, :, :-1, :]
    mask_x = natural_ocean[:, :, :, 1:] & natural_ocean[:, :, :, :-1]
    return mask_y, mask_x


def natural_observation_seam_masks(
    natural_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    land_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    natural = _mask(natural_mask)
    observed = _mask(original_observed_mask)
    artificial = _mask(artificial_mask)
    ocean = ~_mask(land_mask)
    preserved = observed & ~artificial & ocean
    natural = natural & ocean
    seam_y = (
        (natural[:, :, 1:, :] & preserved[:, :, :-1, :])
        | (preserved[:, :, 1:, :] & natural[:, :, :-1, :])
    )
    seam_x = (
        (natural[:, :, :, 1:] & preserved[:, :, :, :-1])
        | (preserved[:, :, :, 1:] & natural[:, :, :, :-1])
    )
    return seam_y, seam_x


def _upper_for_edges(monthly_upper: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    upper = monthly_upper.to(device=field.device, dtype=field.dtype).reshape(-1, 1, 1, 1)
    if upper.shape[0] != field.shape[0] or torch.any(upper <= 0):
        raise ValueError("monthly_upper must contain one positive threshold per batch item")
    return upper


def _soft_upper_penalty(
    gradient: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    return F.relu(torch.abs(gradient) / upper - 1.0).square()


def natural_high_gradient_loss(
    composed: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_upper: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    upper = _upper_for_edges(monthly_upper, composed)
    grad_y, grad_x = _adjacent_differences(composed)
    grad_y = grad_y * float(norm_std)
    grad_x = grad_x * float(norm_std)
    mask_y, mask_x = natural_edge_masks(natural_mask, land_mask)
    penalties = torch.cat(
        [
            _soft_upper_penalty(grad_y, upper).reshape(-1),
            _soft_upper_penalty(grad_x, upper).reshape(-1),
        ]
    )
    valid = torch.cat([mask_y.reshape(-1), mask_x.reshape(-1)])
    return _masked_mean(penalties, valid)


def natural_seam_loss(
    composed: torch.Tensor,
    natural_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_upper: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    upper = _upper_for_edges(monthly_upper, composed)
    grad_y, grad_x = _adjacent_differences(composed)
    grad_y = grad_y * float(norm_std)
    grad_x = grad_x * float(norm_std)
    mask_y, mask_x = natural_observation_seam_masks(
        natural_mask, original_observed_mask, artificial_mask, land_mask
    )
    penalties = torch.cat(
        [
            _soft_upper_penalty(grad_y, upper).reshape(-1),
            _soft_upper_penalty(grad_x, upper).reshape(-1),
        ]
    )
    valid = torch.cat([mask_y.reshape(-1), mask_x.reshape(-1)])
    return _masked_mean(penalties, valid)


def natural_seam_continuity_loss(
    composed: torch.Tensor,
    natural_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    land_mask: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match a natural-gap seam difference to the one-sided proxy difference.

    A seam edge is eligible only when the preserved ocean side has one more
    preserved-ocean pixel in the outward direction.  This estimates the
    continuation gradient without using unavailable natural-gap targets.
    """
    natural = _mask(natural_mask)
    observed = _mask(original_observed_mask)
    artificial = _mask(artificial_mask)
    ocean = ~_mask(land_mask)
    preserved = observed & ~artificial & ocean

    # For each orientation, compare the natural-to-preserved edge with the
    # adjacent preserved-to-preserved edge on the outward side.
    upper_natural = (
        natural[:, :, 2:, :]
        & preserved[:, :, 1:-1, :]
        & preserved[:, :, :-2, :]
    )
    upper_error = (
        (composed[:, :, 2:, :] - composed[:, :, 1:-1, :])
        - (composed[:, :, 1:-1, :] - composed[:, :, :-2, :])
    )
    lower_natural = (
        natural[:, :, :-2, :]
        & preserved[:, :, 1:-1, :]
        & preserved[:, :, 2:, :]
    )
    lower_error = (
        (composed[:, :, :-2, :] - composed[:, :, 1:-1, :])
        - (composed[:, :, 1:-1, :] - composed[:, :, 2:, :])
    )
    left_natural = (
        natural[:, :, :, 2:]
        & preserved[:, :, :, 1:-1]
        & preserved[:, :, :, :-2]
    )
    left_error = (
        (composed[:, :, :, 2:] - composed[:, :, :, 1:-1])
        - (composed[:, :, :, 1:-1] - composed[:, :, :, :-2])
    )
    right_natural = (
        natural[:, :, :, :-2]
        & preserved[:, :, :, 1:-1]
        & preserved[:, :, :, 2:]
    )
    right_error = (
        (composed[:, :, :, :-2] - composed[:, :, :, 1:-1])
        - (composed[:, :, :, 1:-1] - composed[:, :, :, 2:])
    )

    errors = torch.cat(
        [
            torch.abs(upper_error).reshape(-1),
            torch.abs(lower_error).reshape(-1),
            torch.abs(left_error).reshape(-1),
            torch.abs(right_error).reshape(-1),
        ]
    ) * float(norm_std)
    valid = torch.cat(
        [
            upper_natural.reshape(-1),
            lower_natural.reshape(-1),
            left_natural.reshape(-1),
            right_natural.reshape(-1),
        ]
    )
    return _masked_mean(errors, valid)


def _box_sum(field: torch.Tensor, window: int) -> torch.Tensor:
    kernel = torch.ones((1, 1, window, window), device=field.device, dtype=field.dtype)
    return F.conv2d(field, kernel, padding=window // 2)


def _land_aware_lowpass(
    field: torch.Tensor,
    ocean: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ocean_float = ocean.to(field.dtype)
    count = _box_sum(ocean_float, window)
    filtered = _box_sum(field * ocean_float, window) / count.clamp_min(1.0)
    return filtered, count > 0


def _central_gradient_energy(
    field: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(field)
    gy = torch.zeros_like(field)
    gx[:, :, :, 1:-1] = 0.5 * (field[:, :, :, 2:] - field[:, :, :, :-2])
    gy[:, :, 1:-1, :] = 0.5 * (field[:, :, 2:, :] - field[:, :, :-2, :])
    stencil = torch.zeros_like(valid)
    stencil[:, :, 1:-1, 1:-1] = (
        valid[:, :, 1:-1, 1:-1]
        & valid[:, :, :-2, 1:-1]
        & valid[:, :, 2:, 1:-1]
        & valid[:, :, 1:-1, :-2]
        & valid[:, :, 1:-1, 2:]
    )
    return gx.square() + gy.square(), stencil


def _sqrt_zero(value: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(value.clamp_min(0.0) + epsilon) - epsilon**0.5


def natural_low_energy_loss(
    composed: torch.Tensor,
    target_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    *,
    beta: float,
    lowpass_window: int,
    energy_window: int,
    min_natural_support: float,
    norm_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if lowpass_window % 2 == 0 or energy_window % 2 == 0:
        raise ValueError("lowpass and energy windows must be odd")
    if not 0 <= beta <= 1 or not 0 <= min_natural_support <= 1:
        raise ValueError("beta and min_natural_support must lie in [0, 1]")
    natural = _mask(natural_mask)
    ocean = ~_mask(land_mask)
    pred_low, pred_valid = _land_aware_lowpass(composed, ocean, lowpass_window)
    proxy_low, proxy_valid = _land_aware_lowpass(target_proxy, ocean, lowpass_window)
    pred_sq, pred_grad_valid = _central_gradient_energy(pred_low, pred_valid & ocean)
    proxy_sq, proxy_grad_valid = _central_gradient_energy(proxy_low, proxy_valid & ocean)
    grad_valid = pred_grad_valid & proxy_grad_valid

    valid_float = grad_valid.to(composed.dtype)
    energy_count = _box_sum(valid_float, energy_window)
    pred_energy = _sqrt_zero(
        _box_sum(pred_sq * valid_float, energy_window) / energy_count.clamp_min(1.0)
    )
    proxy_energy = _sqrt_zero(
        _box_sum(proxy_sq * valid_float, energy_window) / energy_count.clamp_min(1.0)
    )
    pred_energy = pred_energy * float(norm_std)
    proxy_energy = proxy_energy * float(norm_std)

    area = float(energy_window * energy_window)
    natural_support = _box_sum(natural.to(composed.dtype), energy_window) / area
    eligible = (
        natural
        & ocean
        & (natural_support >= min_natural_support)
        & (energy_count >= area)
        & (proxy_energy > 0)
    )
    deficit = F.relu(float(beta) * proxy_energy - pred_energy).square()
    loss, count = _masked_mean(deficit, eligible)
    ratio, _ = _masked_mean(
        pred_energy / proxy_energy.clamp_min(1e-8), eligible
    )
    return loss, count, ratio


@dataclass(frozen=True)
class HybridLossWeights:
    gradient: float = 0.2
    boundary: float = 0.1
    high: float = 0.05
    seam: float = 0.05
    seam_continuity: float = 0.05
    low: float = 0.02
    low_energy_beta: float = 0.5


def hybrid_loss(
    *,
    raw_prediction: torch.Tensor,
    last_input: torch.Tensor,
    target: torch.Tensor,
    target_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_upper: torch.Tensor,
    norm_std: float,
    weights: HybridLossWeights,
    lowpass_window: int,
    energy_window: int,
    min_natural_support: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    natural = _mask(natural_mask)
    artificial = _mask(artificial_mask)
    observed = _mask(original_observed_mask)
    if torch.any(natural & artificial):
        raise ValueError("natural and artificial masks overlap")
    if torch.any(artificial & ~observed):
        raise ValueError("artificial supervision includes a non-observed pixel")

    reconstruction = natural | artificial
    composed = output_composition(raw_prediction, last_input, reconstruction, land_mask)
    mse, mse_count = supervised_mse_loss(composed, target, artificial)
    gradient, gradient_count = supervised_gradient_loss(
        composed, target, artificial, observed
    )
    boundary, boundary_count = supervised_boundary_loss(
        composed, target, artificial, observed
    )
    high, high_count = natural_high_gradient_loss(
        composed, natural, land_mask, monthly_upper, norm_std=norm_std
    )
    seam, seam_count = natural_seam_loss(
        composed,
        natural,
        observed,
        artificial,
        land_mask,
        monthly_upper,
        norm_std=norm_std,
    )
    seam_continuity, seam_continuity_count = natural_seam_continuity_loss(
        composed,
        natural,
        observed,
        artificial,
        land_mask,
        norm_std=norm_std,
    )
    low, low_count, low_ratio = natural_low_energy_loss(
        composed,
        target_proxy,
        natural,
        land_mask,
        beta=weights.low_energy_beta,
        lowpass_window=lowpass_window,
        energy_window=energy_window,
        min_natural_support=min_natural_support,
        norm_std=norm_std,
    )
    components = {
        "mse": mse,
        "gradient": gradient,
        "boundary": boundary,
        "high": high,
        "seam": seam,
        "seam_continuity": seam_continuity,
        "low": low,
    }
    counts = {
        "mse": mse_count,
        "gradient": gradient_count,
        "boundary": boundary_count,
        "high": high_count,
        "seam": seam_count,
        "seam_continuity": seam_continuity_count,
        "low": low_count,
        "low_energy_ratio": low_ratio,
    }
    total = (
        mse
        + weights.gradient * gradient
        + weights.boundary * boundary
        + weights.high * high
        + weights.seam * seam
        + weights.seam_continuity * seam_continuity
        + weights.low * low
    )
    return total, components, counts, composed
