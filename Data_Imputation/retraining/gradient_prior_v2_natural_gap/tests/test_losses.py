from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.losses import (
    HybridLossWeights,
    hybrid_loss,
    natural_coarse_proxy_loss,
    natural_deployment_seam_continuity_loss,
    natural_deployment_seam_masks,
    natural_edge_masks,
    natural_high_gradient_loss,
    natural_low_energy_loss,
    natural_observation_seam_continuity_loss,
    natural_proxy_core_mask,
    natural_seam_loss,
    boundary_transition_composition,
    boundary_ramp_composition,
    output_composition,
    supervised_boundary_transition_loss,
    supervised_boundary_loss,
    supervised_gradient_loss,
    supervised_mse_loss,
)


def test_boundary_ramp_composition_preserves_anchors_and_limits_three_layers():
    raw = torch.full((1, 1, 11, 11), 10.0)
    proxy = torch.zeros_like(raw)
    preserved = torch.zeros_like(raw)
    natural = torch.zeros((1, 11, 11), dtype=torch.bool)
    natural[:, 1:10, 1:10] = True
    land = torch.zeros_like(natural)
    land[:, 0, 0] = True
    composed = boundary_ramp_composition(
        raw,
        preserved,
        proxy,
        natural,
        land,
        max_layers=3,
        max_seam_gradient=1.0,
    )

    # Original ocean and land are never modified.
    assert torch.equal(composed[:, :, 0, 1:], preserved[:, :, 0, 1:])
    assert composed[0, 0, 0, 0] == preserved[0, 0, 0, 0]
    # The first three natural layers form a bounded outward-to-inward ramp.
    assert composed[0, 0, 1, 1] == pytest.approx(1.0)
    assert composed[0, 0, 2, 2] == pytest.approx(2.0)
    assert composed[0, 0, 3, 3] == pytest.approx(3.0)
    # The unprocessed interior retains the old composition candidate.
    assert composed[0, 0, 5, 5] == pytest.approx(10.0)


def _mask(height=5, width=6):
    return torch.zeros((1, height, width), dtype=torch.bool)


def test_output_composition_uses_model_only_in_reconstruction_ocean():
    last_input = torch.full((1, 1, 3, 4), 3.0)
    raw_prediction = torch.full((1, 1, 3, 4), 9.0)
    reconstruction = _mask(3, 4)
    reconstruction[0, 1, 1] = True
    reconstruction[0, 1, 2] = True
    land = _mask(3, 4)
    land[0, 1, 2] = True

    composed = output_composition(raw_prediction, last_input, reconstruction, land)

    assert composed[0, 0, 1, 1] == 9.0
    assert composed[0, 0, 1, 2] == 3.0
    assert composed[0, 0, 0, 0] == 3.0


def test_boundary_transition_composition_preserves_observed_and_land_pixels():
    last_input = torch.full((1, 1, 5, 5), 3.0)
    raw_prediction = torch.full((1, 1, 5, 5), 9.0)
    natural = _mask(5, 5)
    natural[0, 2, 2] = True
    last_input[0, 0, 2, 2] = 0.0
    proxy = torch.full((1, 1, 5, 5), 3.0)
    land = _mask(5, 5)
    land[0, 0, 0] = True

    composed = boundary_transition_composition(
        raw_prediction, last_input, proxy, natural, land
    )

    assert composed[0, 0, 0, 1] == 3.0
    assert composed[0, 0, 0, 0] == 3.0
    assert composed[0, 0, 2, 2] == 6.0


def test_boundary_transition_composition_uses_raw_prediction_beyond_first_cloud_layer():
    last_input = torch.full((1, 1, 5, 5), 3.0)
    raw_prediction = torch.full((1, 1, 5, 5), 9.0)
    natural = _mask(5, 5)
    natural[:, 1:4, 1:4] = True
    last_input[:, :, 1:4, 1:4] = 0.0
    proxy = torch.full((1, 1, 5, 5), 3.0)
    land = _mask(5, 5)

    composed = boundary_transition_composition(
        raw_prediction, last_input, proxy, natural, land
    )

    assert composed[0, 0, 1, 2] == 6.0
    assert composed[0, 0, 2, 2] == 9.0


def test_boundary_transition_composition_rejects_a_proxy_blend_that_worsens_the_seam():
    last_input = torch.full((1, 1, 3, 3), 3.0)
    raw_prediction = torch.full((1, 1, 3, 3), 4.0)
    natural = _mask(3, 3)
    natural[0, 1, 1] = True
    last_input[0, 0, 1, 1] = 0.0
    proxy = torch.full((1, 1, 3, 3), 9.0)
    land = _mask(3, 3)

    composed = boundary_transition_composition(
        raw_prediction, last_input, proxy, natural, land
    )

    assert composed[0, 0, 1, 1] == 4.0


def test_natural_target_values_cannot_change_any_supervised_loss():
    prediction = torch.arange(30, dtype=torch.float32).reshape(1, 1, 5, 6) / 10
    target_a = prediction.clone()
    target_b = target_a.clone()
    artificial = _mask()
    artificial[0, 2, 2] = True
    natural = _mask()
    natural[0, 2, 3] = True
    observed = torch.ones((1, 5, 6), dtype=torch.bool)
    observed[natural] = False
    target_b[0, 0, 2, 3] = 10_000.0

    losses_a = (
        supervised_mse_loss(prediction, target_a, artificial)[0],
        supervised_gradient_loss(prediction, target_a, artificial, observed)[0],
        supervised_boundary_loss(prediction, target_a, artificial, observed)[0],
    )
    losses_b = (
        supervised_mse_loss(prediction, target_b, artificial)[0],
        supervised_gradient_loss(prediction, target_b, artificial, observed)[0],
        supervised_boundary_loss(prediction, target_b, artificial, observed)[0],
    )

    for first, second in zip(losses_a, losses_b):
        assert torch.equal(first, second)


def test_boundary_transition_loss_matches_value_and_cross_edge_gradient():
    target = torch.zeros((1, 1, 3, 4), dtype=torch.float32)
    target[0, 0, 1, :] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    prediction = target.clone()
    artificial = _mask(3, 4)
    artificial[0, 1, 1] = True
    observed = torch.ones((1, 3, 4), dtype=torch.bool)
    land = _mask(3, 4)

    matching, matching_count = supervised_boundary_transition_loss(
        prediction, target, artificial, observed, land
    )
    altered = prediction.clone()
    altered[0, 0, 1, 1] = 4.0
    changed, changed_count = supervised_boundary_transition_loss(
        altered, target, artificial, observed, land
    )

    assert matching == pytest.approx(0.0, abs=1e-8)
    assert changed > matching
    assert changed_count == matching_count
    assert matching_count > 0


def test_boundary_transition_loss_excludes_natural_pixels_from_supervision():
    target = torch.zeros((1, 1, 3, 4), dtype=torch.float32)
    prediction = target.clone()
    artificial = _mask(3, 4)
    artificial[0, 1, 1] = True
    natural = _mask(3, 4)
    natural[0, 1, 2] = True
    observed = torch.ones((1, 3, 4), dtype=torch.bool)
    observed[natural] = False
    land = _mask(3, 4)

    baseline, baseline_count = supervised_boundary_transition_loss(
        prediction, target, artificial, observed, land
    )
    altered_target = target.clone()
    altered_target[0, 0, 1, 2] = 10_000.0
    altered, altered_count = supervised_boundary_transition_loss(
        prediction, altered_target, artificial, observed, land
    )

    assert altered == baseline
    assert altered_count == baseline_count


def test_natural_interior_and_deployment_seam_edges_are_disjoint():
    natural = _mask(3, 5)
    natural[0, 1, 2:4] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)

    natural_y, natural_x = natural_edge_masks(natural, land)
    seam_y, seam_x = natural_deployment_seam_masks(natural, artificial, land)

    assert natural_y.sum() == 0
    assert natural_x.sum() == 1  # edge 2--3
    assert seam_y.sum() == 4  # two natural pixels, each with upper/lower preserved input
    assert seam_x.sum() == 2  # preserved 1--natural 2 and natural 3--preserved 4
    assert not torch.any(natural_x & seam_x)


def test_high_gradient_soft_hinge_is_zero_below_threshold_and_positive_above():
    natural = torch.ones((1, 3, 5), dtype=torch.bool)
    land = torch.zeros_like(natural)
    threshold = torch.tensor([1.0])
    below = torch.arange(5, dtype=torch.float32).repeat(3, 1)[None, None] * 0.5
    above = below * 4.0

    below_loss, below_count = natural_high_gradient_loss(
        below, natural, land, threshold, norm_std=1.0
    )
    above_loss, above_count = natural_high_gradient_loss(
        above, natural, land, threshold, norm_std=1.0
    )

    assert below_loss == 0
    assert above_loss > 0
    assert below_count == above_count == 22


def test_seam_soft_hinge_penalizes_all_deployment_seam_jumps():
    natural = _mask(3, 5)
    natural[0, 1, 2] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)
    threshold = torch.tensor([1.0])
    below = torch.zeros((1, 1, 3, 5))
    below[0, 0, 1, 2] = 0.5
    above = below.clone()
    above[0, 0, 1, 2] = 2.0

    below_loss, below_count = natural_seam_loss(
        below, natural, artificial, land, threshold, norm_std=1.0
    )
    above_loss, above_count = natural_seam_loss(
        above, natural, artificial, land, threshold, norm_std=1.0
    )

    assert below_loss == 0
    assert above_loss > 0
    assert below_count == above_count == 4


def test_full_seam_continuity_uses_lowpass_reference_and_rejects_bad_proxy_edge():
    natural = _mask(3, 5)
    natural[0, 1, 1] = True
    artificial = torch.ones((1, 3, 5), dtype=torch.bool)
    artificial[0, 1, 1] = False
    artificial[0, 1, 2:4] = False
    land = _mask(3, 5)

    linear = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 5)
    linear = linear.repeat(1, 1, 3, 1)
    linear_loss, linear_count = natural_deployment_seam_continuity_loss(
        linear,
        linear,
        natural,
        artificial,
        land,
        torch.tensor([1.0]),
        norm_std=1.0,
        lowpass_window=3,
    )
    bad_proxy = linear.clone()
    bad_proxy[0, 0, 1, 2] = 12.0
    _, bad_count = natural_deployment_seam_continuity_loss(
        linear,
        bad_proxy,
        natural,
        artificial,
        land,
        torch.tensor([1.0]),
        norm_std=1.0,
        lowpass_window=3,
    )

    assert linear_loss == pytest.approx(0.0, abs=1e-8)
    assert linear_count == 1
    assert bad_count == 0


def test_empty_seam_continuity_term_is_differentiable_zero():
    prediction = torch.randn((1, 1, 4, 4), requires_grad=True)
    natural = _mask(4, 4)
    artificial = _mask(4, 4)
    land = _mask(4, 4)

    loss, count = natural_deployment_seam_continuity_loss(
        prediction,
        prediction.detach(),
        natural,
        artificial,
        land,
        torch.tensor([1.0]),
        norm_std=2.0,
        lowpass_window=3,
    )
    loss.backward()

    assert loss == 0
    assert count == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_observation_seam_continuity_matches_the_real_exterior_slope():
    prediction = torch.zeros((1, 1, 3, 5), requires_grad=True)
    prediction.data[0, 0, 1, 1] = 0.0
    natural = _mask(3, 5)
    natural[0, 1, 1] = True
    observed = _mask(3, 5)
    observed[0, 1, 2:4] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)
    last_input = torch.zeros_like(prediction)
    last_input[0, 0, 1, 2:4] = torch.tensor([1.0, 2.0])
    composed = output_composition(prediction, last_input, natural, land)

    loss, count = natural_observation_seam_continuity_loss(
        composed, natural, observed, artificial, land, norm_std=1.0
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert count == 1
    assert prediction.grad[0, 0, 1, 1] == pytest.approx(0.0, abs=1e-8)


def test_observation_seam_continuity_penalizes_a_mismatched_fno_slope():
    prediction = torch.zeros((1, 1, 3, 5), requires_grad=True)
    prediction.data[0, 0, 1, 1] = 0.5
    natural = _mask(3, 5)
    natural[0, 1, 1] = True
    observed = _mask(3, 5)
    observed[0, 1, 2:4] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)
    last_input = torch.zeros_like(prediction)
    last_input[0, 0, 1, 2:4] = torch.tensor([1.0, 2.0])
    composed = output_composition(prediction, last_input, natural, land)

    loss, count = natural_observation_seam_continuity_loss(
        composed, natural, observed, artificial, land, norm_std=1.0
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.5)
    assert count == 1
    assert prediction.grad[0, 0, 1, 1] == pytest.approx(1.0)


def test_observation_seam_continuity_detaches_exterior_observations():
    prediction = torch.zeros((1, 1, 3, 5), requires_grad=True)
    prediction.data[0, 0, 1, 1] = 0.5
    natural = _mask(3, 5)
    natural[0, 1, 1] = True
    observed = _mask(3, 5)
    observed[0, 1, 2:4] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)
    last_input = torch.zeros((1, 1, 3, 5), requires_grad=True)
    with torch.no_grad():
        last_input[0, 0, 1, 2:4] = torch.tensor([1.0, 2.0])
    composed = output_composition(prediction, last_input, natural, land)

    loss, count = natural_observation_seam_continuity_loss(
        composed, natural, observed, artificial, land, norm_std=1.0
    )
    loss.backward()

    assert count == 1
    assert prediction.grad[0, 0, 1, 1] == pytest.approx(1.0)
    assert last_input.grad is None or torch.count_nonzero(last_input.grad) == 0


@pytest.mark.parametrize(
    "excluded", ("first_exterior", "second_exterior", "artificial", "land", "natural_land")
)
def test_observation_seam_continuity_excludes_non_original_exterior_pixels(excluded):
    prediction = torch.zeros((1, 1, 3, 5), requires_grad=True)
    natural = _mask(3, 5)
    natural[0, 1, 1] = True
    observed = _mask(3, 5)
    observed[0, 1, 2:4] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)
    if excluded == "first_exterior":
        observed[0, 1, 2] = False
    elif excluded == "second_exterior":
        observed[0, 1, 3] = False
    elif excluded == "artificial":
        artificial[0, 1, 2] = True
    elif excluded == "land":
        land[0, 1, 3] = True
    else:
        land[0, 1, 1] = True
    last_input = torch.zeros_like(prediction)
    composed = output_composition(prediction, last_input, natural, land)

    loss, count = natural_observation_seam_continuity_loss(
        composed, natural, observed, artificial, land, norm_std=1.0
    )
    loss.backward()

    assert loss == 0
    assert count == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_empty_observation_seam_continuity_term_is_differentiable_zero():
    prediction = torch.randn((1, 1, 4, 4), requires_grad=True)
    natural = _mask(4, 4)
    observed = torch.ones((1, 4, 4), dtype=torch.bool)
    artificial = _mask(4, 4)
    land = _mask(4, 4)

    loss, count = natural_observation_seam_continuity_loss(
        prediction, natural, observed, artificial, land, norm_std=2.0
    )
    loss.backward()

    assert loss == 0
    assert count == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_coarse_proxy_loss_has_gradient_only_at_confident_natural_cores():
    prediction = torch.zeros((1, 1, 7, 7), requires_grad=True)
    proxy = torch.zeros_like(prediction)
    proxy[:, :, 3, 3] = 0.2
    natural = _mask(7, 7)
    natural[:, 1:6, 1:6] = True
    land = _mask(7, 7)

    loss, count = natural_coarse_proxy_loss(
        prediction,
        proxy,
        natural,
        land,
        torch.tensor([1.0]),
        norm_std=1.0,
        lowpass_window=3,
        min_natural_support=1.0,
        huber_delta_k=0.5,
    )
    loss.backward()
    eligible, _ = natural_proxy_core_mask(
        proxy,
        natural,
        land,
        torch.tensor([1.0]),
        norm_std=1.0,
        window=3,
        min_natural_support=1.0,
    )

    assert count == 9
    assert torch.isfinite(prediction.grad).all()
    assert torch.all(eligible <= natural.unsqueeze(1))


def test_coarse_proxy_loss_excludes_anomalous_proxy_core():
    proxy = torch.zeros((1, 1, 7, 7))
    proxy[:, :, 3, 3] = 10.0
    natural = _mask(7, 7)
    natural[:, 1:6, 1:6] = True

    eligible, _ = natural_proxy_core_mask(
        proxy,
        natural,
        _mask(7, 7),
        torch.tensor([1.0]),
        norm_std=1.0,
        window=3,
        min_natural_support=1.0,
    )

    assert not eligible[0, 0, 3, 3]


def test_deployment_seam_masks_exclude_artificial_and_cover_preserved_inputs():
    natural = _mask(3, 7)
    natural[0, 0, 3] = True
    artificial = _mask(3, 7)
    artificial[0, 1, 3] = True
    land = _mask(3, 7)

    seam_y, seam_x = natural_deployment_seam_masks(natural, artificial, land)

    assert seam_y.sum() == 0
    assert seam_x.sum() == 2


def test_low_energy_term_prevents_flat_gap_but_not_matching_smooth_structure():
    yy, xx = torch.meshgrid(torch.arange(15), torch.arange(15), indexing="ij")
    proxy = (xx.float() * 0.2 + yy.float() * 0.1)[None, None]
    flat = torch.zeros_like(proxy)
    natural = torch.ones((1, 15, 15), dtype=torch.bool)
    land = torch.zeros_like(natural)

    flat_loss, flat_count, flat_ratio = natural_low_energy_loss(
        flat,
        proxy,
        natural,
        land,
        beta=0.5,
        lowpass_window=3,
        energy_window=3,
        min_natural_support=1.0,
    )
    matching_loss, matching_count, matching_ratio = natural_low_energy_loss(
        proxy,
        proxy,
        natural,
        land,
        beta=0.5,
        lowpass_window=3,
        energy_window=3,
        min_natural_support=1.0,
    )

    assert flat_loss > 0
    assert matching_loss == pytest.approx(0.0, abs=1e-8)
    assert flat_count == matching_count > 0
    assert flat_ratio == pytest.approx(0.0, abs=1e-6)
    assert matching_ratio == pytest.approx(1.0, rel=1e-4)


def test_checkerboard_noise_does_not_satisfy_low_frequency_energy_floor():
    yy, xx = torch.meshgrid(torch.arange(21), torch.arange(21), indexing="ij")
    proxy = (xx.float() * 0.1)[None, None]
    flat = torch.zeros_like(proxy)
    checkerboard = (((xx + yy) % 2) * 2 - 1).float()[None, None] * 0.2
    natural = torch.ones((1, 21, 21), dtype=torch.bool)
    land = torch.zeros_like(natural)
    kwargs = dict(
        beta=0.5,
        lowpass_window=9,
        energy_window=5,
        min_natural_support=1.0,
    )

    flat_loss = natural_low_energy_loss(flat, proxy, natural, land, **kwargs)[0]
    noisy_loss = natural_low_energy_loss(
        checkerboard, proxy, natural, land, **kwargs
    )[0]

    assert noisy_loss > 0
    assert noisy_loss == pytest.approx(float(flat_loss), rel=0.15)


def test_empty_prior_masks_return_differentiable_finite_zero():
    prediction = torch.randn((1, 1, 4, 4), requires_grad=True)
    natural = _mask(4, 4)
    land = _mask(4, 4)

    loss, count = natural_high_gradient_loss(
        prediction,
        natural,
        land,
        torch.tensor([1.0]),
        norm_std=2.0,
    )
    loss.backward()

    assert loss == 0
    assert count == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_hybrid_loss_reports_named_components_and_counts():
    raw = torch.zeros((1, 1, 9, 9), requires_grad=True)
    last_input = torch.zeros_like(raw)
    target = torch.ones_like(raw) * 0.1
    proxy = torch.arange(9, dtype=torch.float32).repeat(9, 1)[None, None] * 0.05
    natural = _mask(9, 9)
    natural[:, 2:7, 2:7] = True
    artificial = _mask(9, 9)
    artificial[:, 0:2, 0:2] = True
    observed = ~natural
    land = _mask(9, 9)

    total, components, counts, composed = hybrid_loss(
        raw_prediction=raw,
        last_input=last_input,
        target=target,
        natural_mask=natural,
        artificial_mask=artificial,
        original_observed_mask=observed,
        land_mask=land,
        monthly_upper=torch.tensor([1.0]),
        norm_std=2.0,
        weights=HybridLossWeights(),
    )

    assert set(components) == {
        "mse",
        "gradient",
        "boundary_transition",
        "seam",
    }
    assert set(counts) == set(components)
    assert composed.shape == raw.shape
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(raw.grad).all()


def test_hybrid_loss_weights_the_boundary_transition_component():
    raw = torch.zeros((1, 1, 3, 5), requires_grad=True)
    raw.data[0, 0, 1, 1] = 0.5
    last_input = torch.zeros_like(raw)
    target = torch.zeros_like(raw)
    natural = _mask(3, 5)
    artificial = _mask(3, 5)
    artificial[0, 1, 1] = True
    observed = torch.ones((1, 3, 5), dtype=torch.bool)
    land = _mask(3, 5)
    base_weights = HybridLossWeights(
        gradient=0.0,
        boundary_transition=0.0,
        seam=0.0,
    )

    base_total, base_components, base_counts, _ = hybrid_loss(
        raw_prediction=raw,
        last_input=last_input,
        target=target,
        natural_mask=natural,
        artificial_mask=artificial,
        original_observed_mask=observed,
        land_mask=land,
        monthly_upper=torch.tensor([1.0]),
        norm_std=1.0,
        weights=base_weights,
    )
    weighted_total, _, _, _ = hybrid_loss(
        raw_prediction=raw,
        last_input=last_input,
        target=target,
        natural_mask=natural,
        artificial_mask=artificial,
        original_observed_mask=observed,
        land_mask=land,
        monthly_upper=torch.tensor([1.0]),
        norm_std=1.0,
        weights=HybridLossWeights(
            gradient=0.0,
            boundary_transition=2.0,
            seam=0.0,
        ),
    )

    assert base_counts["boundary_transition"] > 0
    assert base_components["boundary_transition"] > 0
    assert (weighted_total - base_total).item() == pytest.approx(
        2.0 * base_components["boundary_transition"].item()
    )
