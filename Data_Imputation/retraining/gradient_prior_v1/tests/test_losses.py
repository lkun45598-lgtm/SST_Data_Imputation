from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v1.losses import (
    HybridLossWeights,
    hybrid_loss,
    natural_edge_masks,
    natural_high_gradient_loss,
    natural_low_energy_loss,
    natural_observation_seam_masks,
    natural_seam_continuity_loss,
    natural_seam_loss,
    output_composition,
    supervised_boundary_loss,
    supervised_gradient_loss,
    supervised_mse_loss,
)


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


def test_natural_interior_and_observation_seam_edges_are_disjoint():
    natural = _mask(3, 5)
    natural[0, 1, 2:4] = True
    observed = _mask(3, 5)
    observed[0, 1, 1] = True
    observed[0, 1, 4] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)

    natural_y, natural_x = natural_edge_masks(natural, land)
    seam_y, seam_x = natural_observation_seam_masks(
        natural, observed, artificial, land
    )

    assert natural_y.sum() == 0
    assert natural_x.sum() == 1  # edge 2--3
    assert seam_y.sum() == 0
    assert seam_x.sum() == 2  # observed 1--natural 2 and natural 3--observed 4
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


def test_seam_soft_hinge_only_penalizes_observation_natural_jump():
    natural = _mask(3, 5)
    natural[0, 1, 2] = True
    observed = _mask(3, 5)
    observed[0, 1, 1] = True
    observed[0, 1, 3] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)
    threshold = torch.tensor([1.0])
    below = torch.zeros((1, 1, 3, 5))
    below[0, 0, 1, 2] = 0.5
    above = below.clone()
    above[0, 0, 1, 2] = 2.0

    below_loss, below_count = natural_seam_loss(
        below, natural, observed, artificial, land, threshold, norm_std=1.0
    )
    above_loss, above_count = natural_seam_loss(
        above, natural, observed, artificial, land, threshold, norm_std=1.0
    )

    assert below_loss == 0
    assert above_loss > 0
    assert below_count == above_count == 2


def test_seam_continuity_matches_one_sided_preserved_gradient():
    natural = _mask(3, 5)
    natural[0, 1, 2] = True
    observed = torch.ones_like(natural)
    observed[natural] = False
    artificial = _mask(3, 5)
    land = _mask(3, 5)

    linear = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 5)
    linear = linear.repeat(1, 1, 3, 1)
    linear_loss, linear_count = natural_seam_continuity_loss(
        linear, natural, observed, artificial, land, norm_std=1.0
    )

    discontinuous = linear.clone()
    discontinuous[0, 0, 1, 2] = 5.0
    discontinuous_loss, discontinuous_count = natural_seam_continuity_loss(
        discontinuous, natural, observed, artificial, land, norm_std=1.0
    )

    assert linear_loss == pytest.approx(0.0, abs=1e-8)
    assert discontinuous_loss > 0
    assert linear_count == discontinuous_count == 2


def test_empty_seam_continuity_term_is_differentiable_zero():
    prediction = torch.randn((1, 1, 4, 4), requires_grad=True)
    natural = _mask(4, 4)
    observed = _mask(4, 4)
    artificial = _mask(4, 4)
    land = _mask(4, 4)

    loss, count = natural_seam_continuity_loss(
        prediction, natural, observed, artificial, land, norm_std=2.0
    )
    loss.backward()

    assert loss == 0
    assert count == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


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
        target_proxy=proxy,
        natural_mask=natural,
        artificial_mask=artificial,
        original_observed_mask=observed,
        land_mask=land,
        monthly_upper=torch.tensor([1.0]),
        norm_std=2.0,
        weights=HybridLossWeights(),
        lowpass_window=3,
        energy_window=3,
        min_natural_support=0.5,
    )

    assert set(components) == {
        "mse",
        "gradient",
        "boundary",
        "high",
        "seam",
        "seam_continuity",
        "low",
    }
    assert set(counts) == set(components) | {"low_energy_ratio"}
    assert composed.shape == raw.shape
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(raw.grad).all()
