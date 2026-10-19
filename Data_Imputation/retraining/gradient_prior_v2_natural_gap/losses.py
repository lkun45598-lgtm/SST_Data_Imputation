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


def _has_four_neighbor(mask: torch.Tensor) -> torch.Tensor:
    """Return positions with at least one four-connected ``True`` neighbor."""
    adjacent = torch.zeros_like(mask)
    adjacent[:, :, 1:, :] |= mask[:, :, :-1, :]
    adjacent[:, :, :-1, :] |= mask[:, :, 1:, :]
    adjacent[:, :, :, 1:] |= mask[:, :, :, :-1]
    adjacent[:, :, :, :-1] |= mask[:, :, :, 1:]
    return adjacent


def boundary_transition_composition(
    raw_prediction: torch.Tensor,
    preserved_input: torch.Tensor,
    causal_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
) -> torch.Tensor:
    """Adaptively blend the first natural-cloud layer with the causal proxy.

    The deployment field preserves non-cloud ocean and land values exactly. A
    first-layer natural-cloud pixel considers an equal causal proxy/FNO blend.
    It is accepted only when it does not increase the absolute jump across any
    adjacent preserved-ocean edge and strictly reduces at least one such jump.
    All rejected and deeper natural-cloud pixels use the unmodified FNO
    prediction. ``causal_proxy`` must be the unmasked current proxy field, not
    the model input where cloud pixels were replaced with the normalization
    mean.
    """
    natural = _mask(natural_mask)
    land = _mask(land_mask)
    if causal_proxy.shape != raw_prediction.shape:
        raise ValueError("prediction and causal proxy must have equal BCHW shapes")
    base = output_composition(raw_prediction, preserved_input, natural, land)
    natural_ocean = natural & ~land
    preserved_ocean = ~natural & ~land
    first_cloud_layer = natural_ocean & _has_four_neighbor(preserved_ocean)
    blended = 0.5 * (raw_prediction + causal_proxy)
    non_worsening = torch.ones_like(first_cloud_layer)
    strictly_improved = torch.zeros_like(first_cloud_layer)

    def compare_edge(
        cloud_values: torch.Tensor,
        proxy_values: torch.Tensor,
        preserved_values: torch.Tensor,
        cloud_mask: torch.Tensor,
        preserved_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = cloud_mask & preserved_mask
        change = torch.abs(proxy_values - preserved_values) - torch.abs(
            cloud_values - preserved_values
        )
        return (~valid) | (change <= 0), valid & (change < 0)

    # The four slices refer to the natural-cloud endpoint of each cross-seam edge.
    no_worse, improved = compare_edge(
        raw_prediction[:, :, 1:, :],
        blended[:, :, 1:, :],
        preserved_input[:, :, :-1, :],
        natural_ocean[:, :, 1:, :],
        preserved_ocean[:, :, :-1, :],
    )
    non_worsening[:, :, 1:, :] &= no_worse
    strictly_improved[:, :, 1:, :] |= improved
    no_worse, improved = compare_edge(
        raw_prediction[:, :, :-1, :],
        blended[:, :, :-1, :],
        preserved_input[:, :, 1:, :],
        natural_ocean[:, :, :-1, :],
        preserved_ocean[:, :, 1:, :],
    )
    non_worsening[:, :, :-1, :] &= no_worse
    strictly_improved[:, :, :-1, :] |= improved
    no_worse, improved = compare_edge(
        raw_prediction[:, :, :, 1:],
        blended[:, :, :, 1:],
        preserved_input[:, :, :, :-1],
        natural_ocean[:, :, :, 1:],
        preserved_ocean[:, :, :, :-1],
    )
    non_worsening[:, :, :, 1:] &= no_worse
    strictly_improved[:, :, :, 1:] |= improved
    no_worse, improved = compare_edge(
        raw_prediction[:, :, :, :-1],
        blended[:, :, :, :-1],
        preserved_input[:, :, :, 1:],
        natural_ocean[:, :, :, :-1],
        preserved_ocean[:, :, :, 1:],
    )
    non_worsening[:, :, :, :-1] &= no_worse
    strictly_improved[:, :, :, :-1] |= improved

    use_blend = first_cloud_layer & non_worsening & strictly_improved
    return torch.where(use_blend, blended, base)


def boundary_ramp_composition(
    raw_prediction: torch.Tensor,
    preserved_input: torch.Tensor,
    causal_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    *,
    max_layers: int = 3,
    max_seam_gradient: float = 0.6,
) -> torch.Tensor:
    """Apply a conservative multi-layer ramp at a natural-gap boundary.

    The existing adaptive proxy/FNO composition is used as the candidate.
    For the first ``max_layers`` four-connected layers, only candidate values
    exceeding ``max_seam_gradient`` from the already anchored exterior are
    clipped toward that exterior.  Thus pixels already improved by the old
    composition are not moved, while extreme one-pixel jumps are bounded.

    Values are expected in the same normalized units as the model inputs.
    ``causal_proxy`` is accepted for API parity with the deployment composer;
    it is only used as a finite fallback if a raw prediction is non-finite.
    """
    if raw_prediction.shape != preserved_input.shape:
        raise ValueError("prediction and preserved input must have equal shapes")
    if causal_proxy.shape != raw_prediction.shape:
        raise ValueError("prediction and causal proxy must have equal shapes")
    if max_layers <= 0:
        raise ValueError("max_layers must be positive")
    if max_seam_gradient <= 0:
        raise ValueError("max_seam_gradient must be positive")

    natural = _mask(natural_mask)
    land = _mask(land_mask)
    natural_ocean = natural & ~land
    preserved_ocean = ~natural & ~land
    result = boundary_transition_composition(
        raw_prediction,
        preserved_input,
        causal_proxy,
        natural,
        land,
    )
    finite_raw = torch.isfinite(raw_prediction)
    candidate = torch.where(torch.isfinite(result), result, causal_proxy)

    known = preserved_ocean.clone()
    reference = preserved_input.clone()
    remaining = natural_ocean.clone()
    step = float(max_seam_gradient)

    for layer in range(1, int(max_layers) + 1):
        frontier = remaining & _has_four_neighbor(known)
        if not torch.any(frontier):
            break

        # Propagate an exterior reference through already completed layers.
        values = reference
        sum_neighbours = torch.zeros_like(values)
        count_neighbours = torch.zeros_like(values)
        sum_neighbours[:, :, 1:, :] += values[:, :, :-1, :] * known[:, :, :-1, :]
        count_neighbours[:, :, 1:, :] += known[:, :, :-1, :]
        sum_neighbours[:, :, :-1, :] += values[:, :, 1:, :] * known[:, :, 1:, :]
        count_neighbours[:, :, :-1, :] += known[:, :, 1:, :]
        sum_neighbours[:, :, :, 1:] += values[:, :, :, :-1] * known[:, :, :, :-1]
        count_neighbours[:, :, :, 1:] += known[:, :, :, :-1]
        sum_neighbours[:, :, :, :-1] += values[:, :, :, 1:] * known[:, :, :, 1:]
        count_neighbours[:, :, :, :-1] += known[:, :, :, 1:]
        local_reference = sum_neighbours / count_neighbours.clamp_min(1.0)

        delta = candidate - local_reference
        bounded = local_reference + delta.clamp(min=-step, max=step)
        result = torch.where(frontier, bounded, result)
        reference = torch.where(frontier, bounded, reference)
        known = known | frontier
        remaining = remaining & ~frontier

    return result


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


def supervised_boundary_transition_loss(
    composed: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    land_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise value and slope continuity across artificial cloud edges.

    Each valid edge crosses from an artificial target pixel to a preserved
    original observation.  The artificial endpoint receives a robust value
    penalty and the cross-edge difference receives an L1 gradient penalty.
    Natural cloud, proxy, and land pixels never enter this supervised term.
    """
    artificial = _mask(artificial_mask)
    observed_ocean = _mask(original_observed_mask) & ~_mask(land_mask)
    valid_y = observed_ocean[:, :, 1:, :] & observed_ocean[:, :, :-1, :]
    valid_x = observed_ocean[:, :, :, 1:] & observed_ocean[:, :, :, :-1]
    boundary_y = artificial[:, :, 1:, :] != artificial[:, :, :-1, :]
    boundary_x = artificial[:, :, :, 1:] != artificial[:, :, :, :-1]

    artificial_value_y = torch.where(
        artificial[:, :, 1:, :],
        composed[:, :, 1:, :],
        composed[:, :, :-1, :],
    )
    target_value_y = torch.where(
        artificial[:, :, 1:, :],
        target[:, :, 1:, :],
        target[:, :, :-1, :],
    )
    artificial_value_x = torch.where(
        artificial[:, :, :, 1:],
        composed[:, :, :, 1:],
        composed[:, :, :, :-1],
    )
    target_value_x = torch.where(
        artificial[:, :, :, 1:],
        target[:, :, :, 1:],
        target[:, :, :, :-1],
    )
    pred_y, pred_x = _adjacent_differences(composed)
    target_y, target_x = _adjacent_differences(target)
    values = torch.cat(
        [
            0.5
            * (
                F.huber_loss(artificial_value_y, target_value_y, reduction="none")
                + torch.abs(pred_y - target_y)
            ).reshape(-1),
            0.5
            * (
                F.huber_loss(artificial_value_x, target_value_x, reduction="none")
                + torch.abs(pred_x - target_x)
            ).reshape(-1),
        ]
    )
    valid = torch.cat(
        [(valid_y & boundary_y).reshape(-1), (valid_x & boundary_x).reshape(-1)]
    )
    return _masked_mean(values, valid)


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


def natural_observation_seam_continuity_loss(
    composed: torch.Tensor,
    natural_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    land_mask: torch.Tensor,
    *,
    norm_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continue the observed exterior slope into natural cloud gaps.

    For each natural-gap pixel ``p``, an eligible outward pair ``q, r`` must
    both be original observations in the current frame.  The L1 residual
    matches the FNO-to-observation step at ``p-q`` to the observed exterior
    step at ``q-r``.  Proxy, temporal-fill, KNN, artificial-mask, and land
    pixels are never used as references.
    """
    natural = _mask(natural_mask)
    observed = _mask(original_observed_mask)
    artificial = _mask(artificial_mask)
    ocean = ~_mask(land_mask)
    natural = natural & ~artificial & ocean
    reference = observed & ~artificial & ocean & ~natural
    observations = composed.detach()

    # p is the natural pixel; q and r move outward from p in each direction.
    upper_valid = (
        natural[:, :, 2:, :]
        & reference[:, :, 1:-1, :]
        & reference[:, :, :-2, :]
    )
    upper_error = (
        (composed[:, :, 2:, :] - observations[:, :, 1:-1, :])
        - (observations[:, :, 1:-1, :] - observations[:, :, :-2, :])
    )
    lower_valid = (
        natural[:, :, :-2, :]
        & reference[:, :, 1:-1, :]
        & reference[:, :, 2:, :]
    )
    lower_error = (
        (composed[:, :, :-2, :] - observations[:, :, 1:-1, :])
        - (observations[:, :, 1:-1, :] - observations[:, :, 2:, :])
    )
    left_valid = (
        natural[:, :, :, 2:]
        & reference[:, :, :, 1:-1]
        & reference[:, :, :, :-2]
    )
    left_error = (
        (composed[:, :, :, 2:] - observations[:, :, :, 1:-1])
        - (observations[:, :, :, 1:-1] - observations[:, :, :, :-2])
    )
    right_valid = (
        natural[:, :, :, :-2]
        & reference[:, :, :, 1:-1]
        & reference[:, :, :, 2:]
    )
    right_error = (
        (composed[:, :, :, :-2] - observations[:, :, :, 1:-1])
        - (observations[:, :, :, 1:-1] - observations[:, :, :, 2:])
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
            upper_valid.reshape(-1),
            lower_valid.reshape(-1),
            left_valid.reshape(-1),
            right_valid.reshape(-1),
        ]
    )
    return _masked_mean(errors, valid)


def natural_deployment_seam_masks(
    natural_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    land_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every deployed natural-to-preserved ocean seam edge.

    Output composition predicts natural and artificial reconstruction pixels,
    and copies the input at every other ocean pixel.  This mask therefore
    includes seams to original observations as well as temporal/KNN proxies.
    """
    natural = _mask(natural_mask)
    artificial = _mask(artificial_mask)
    ocean = ~_mask(land_mask)
    natural_ocean = natural & ocean
    preserved = ocean & ~(natural | artificial)
    seam_y = (
        (natural_ocean[:, :, 1:, :] & preserved[:, :, :-1, :])
        | (preserved[:, :, 1:, :] & natural_ocean[:, :, :-1, :])
    )
    seam_x = (
        (natural_ocean[:, :, :, 1:] & preserved[:, :, :, :-1])
        | (preserved[:, :, :, 1:] & natural_ocean[:, :, :, :-1])
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
    mask_y, mask_x = natural_deployment_seam_masks(natural_mask, artificial_mask, land_mask)
    penalties = torch.cat(
        [
            _soft_upper_penalty(grad_y, upper).reshape(-1),
            _soft_upper_penalty(grad_x, upper).reshape(-1),
        ]
    )
    valid = torch.cat([mask_y.reshape(-1), mask_x.reshape(-1)])
    return _masked_mean(penalties, valid)


def natural_deployment_seam_continuity_loss(
    composed: torch.Tensor,
    target_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_proxy_highpass_upper: torch.Tensor,
    *,
    norm_std: float,
    lowpass_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match a deployed seam difference to a reliable low-pass exterior step.

    An eligible term has a natural prediction pixel p, its preserved output
    neighbour q, and one additional preserved exterior pixel r.  The exterior
    difference q-r comes from the confidence-gated low-pass proxy, not the
    raw KNN/temporal input.  No natural-gap proxy pixel is used as a target.
    """
    natural = _mask(natural_mask)
    artificial = _mask(artificial_mask)
    ocean = ~_mask(land_mask)
    preserved = ocean & ~(natural | artificial)
    proxy_low, proxy_confident = _proxy_lowpass_confidence(
        target_proxy,
        land_mask,
        monthly_proxy_highpass_upper,
        norm_std=norm_std,
        window=lowpass_window,
    )

    # p is natural; q and r move outward through the preserved input.
    upper_natural = (
        natural[:, :, 2:, :]
        & preserved[:, :, 1:-1, :]
        & preserved[:, :, :-2, :]
        & proxy_confident[:, :, 1:-1, :]
        & proxy_confident[:, :, :-2, :]
    )
    upper_error = (
        (composed[:, :, 2:, :] - composed[:, :, 1:-1, :])
        - (proxy_low[:, :, 1:-1, :].detach() - proxy_low[:, :, :-2, :].detach())
    )
    lower_natural = (
        natural[:, :, :-2, :]
        & preserved[:, :, 1:-1, :]
        & preserved[:, :, 2:, :]
        & proxy_confident[:, :, 1:-1, :]
        & proxy_confident[:, :, 2:, :]
    )
    lower_error = (
        (composed[:, :, :-2, :] - composed[:, :, 1:-1, :])
        - (proxy_low[:, :, 1:-1, :].detach() - proxy_low[:, :, 2:, :].detach())
    )
    left_natural = (
        natural[:, :, :, 2:]
        & preserved[:, :, :, 1:-1]
        & preserved[:, :, :, :-2]
        & proxy_confident[:, :, :, 1:-1]
        & proxy_confident[:, :, :, :-2]
    )
    left_error = (
        (composed[:, :, :, 2:] - composed[:, :, :, 1:-1])
        - (proxy_low[:, :, :, 1:-1].detach() - proxy_low[:, :, :, :-2].detach())
    )
    right_natural = (
        natural[:, :, :, :-2]
        & preserved[:, :, :, 1:-1]
        & preserved[:, :, :, 2:]
        & proxy_confident[:, :, :, 1:-1]
        & proxy_confident[:, :, :, 2:]
    )
    right_error = (
        (composed[:, :, :, :-2] - composed[:, :, :, 1:-1])
        - (proxy_low[:, :, :, 1:-1].detach() - proxy_low[:, :, :, 2:].detach())
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


def _proxy_lowpass_confidence(
    target_proxy: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_proxy_highpass_upper: torch.Tensor,
    *,
    norm_std: float,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a land-aware low-pass proxy and reliable proxy centres."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    ocean = ~_mask(land_mask)
    proxy_low, proxy_valid = _land_aware_lowpass(target_proxy, ocean, window)
    upper = _upper_for_edges(monthly_proxy_highpass_upper, target_proxy)
    residual_k = torch.abs((target_proxy - proxy_low) * float(norm_std))
    confident = ocean & proxy_valid & (residual_k <= upper)
    return proxy_low, confident


def natural_proxy_core_mask(
    target_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_proxy_highpass_upper: torch.Tensor,
    *,
    norm_std: float,
    window: int,
    min_natural_support: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return confident natural-gap low-pass centres and the proxy low-pass."""
    if not 0.0 <= min_natural_support <= 1.0:
        raise ValueError("min_natural_support must lie in [0, 1]")
    natural = _mask(natural_mask)
    ocean = ~_mask(land_mask)
    proxy_low, proxy_confident = _proxy_lowpass_confidence(
        target_proxy,
        land_mask,
        monthly_proxy_highpass_upper,
        norm_std=norm_std,
        window=window,
    )
    natural_support = _box_sum((natural & ocean).to(target_proxy.dtype), window)
    ocean_support = _box_sum(ocean.to(target_proxy.dtype), window)
    eligible = (
        natural
        & ocean
        & (ocean_support > 0)
        & (natural_support / ocean_support.clamp_min(1.0) >= min_natural_support)
        & proxy_confident
    )
    return eligible, proxy_low


def natural_coarse_proxy_loss(
    composed: torch.Tensor,
    target_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_proxy_highpass_upper: torch.Tensor,
    *,
    norm_std: float,
    lowpass_window: int,
    min_natural_support: float,
    huber_delta_k: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weakly align low-frequency natural cores without proxy pseudo-targets."""
    if huber_delta_k <= 0:
        raise ValueError("huber_delta_k must be positive")
    ocean = ~_mask(land_mask)
    pred_low, _ = _land_aware_lowpass(composed, ocean, lowpass_window)
    eligible, proxy_low = natural_proxy_core_mask(
        target_proxy,
        natural_mask,
        land_mask,
        monthly_proxy_highpass_upper,
        norm_std=norm_std,
        window=lowpass_window,
        min_natural_support=min_natural_support,
    )
    values = F.huber_loss(
        pred_low,
        proxy_low.detach(),
        delta=float(huber_delta_k) / float(norm_std),
        reduction="none",
    )
    return _masked_mean(values, eligible)


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
    boundary_transition: float = 0.05
    seam: float = 0.02


def hybrid_loss(
    *,
    raw_prediction: torch.Tensor,
    last_input: torch.Tensor,
    target: torch.Tensor,
    natural_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
    original_observed_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_upper: torch.Tensor,
    norm_std: float,
    weights: HybridLossWeights,
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
    boundary_transition, boundary_transition_count = supervised_boundary_transition_loss(
        composed, target, artificial, observed, land_mask
    )
    seam, seam_count = natural_seam_loss(
        composed,
        natural,
        artificial,
        land_mask,
        monthly_upper,
        norm_std=norm_std,
    )
    components = {
        "mse": mse,
        "gradient": gradient,
        "boundary_transition": boundary_transition,
        "seam": seam,
    }
    counts = {
        "mse": mse_count,
        "gradient": gradient_count,
        "boundary_transition": boundary_transition_count,
        "seam": seam_count,
    }
    total = (
        mse
        + weights.gradient * gradient
        + weights.boundary_transition * boundary_transition
        + weights.seam * seam
    )
    return total, components, counts, composed
