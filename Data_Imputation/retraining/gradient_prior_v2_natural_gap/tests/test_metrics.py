from pathlib import Path
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.metrics import (
    EpochMetricAccumulator,
    edge_difference_values,
    local_predicted_outlier_terms,
    natural_exceedance_terms,
    seam_exceedance_terms,
    supervised_boundary_error_terms,
    supervised_error_terms,
    supervised_gradient_error_terms,
)


def _mask(height, width):
    return torch.zeros((1, height, width), dtype=torch.bool)


def test_supervised_mae_and_rmse_terms_use_physical_units():
    prediction = torch.tensor([[[[0.0, 1.0, 2.0]]]])
    target = torch.zeros_like(prediction)
    artificial = torch.tensor([[[False, True, True]]])

    terms = supervised_error_terms(prediction, target, artificial, norm_std=2.0)

    assert terms["absolute_sum"] == 6.0
    assert terms["squared_sum"] == 20.0
    assert terms["count"] == 2


def test_strict_central_gradient_mae_uses_gradient_magnitude_at_masked_centres():
    prediction = torch.arange(5, dtype=torch.float32).repeat(5, 1)[None, None]
    target = torch.zeros_like(prediction)
    artificial = _mask(5, 5)
    artificial[0, 2, 2] = True
    observed = torch.ones((1, 5, 5), dtype=torch.bool)

    gradient_sum, gradient_count = supervised_gradient_error_terms(
        prediction, target, artificial, observed, norm_std=2.0
    )

    assert gradient_count == 1
    assert gradient_sum == 2.0


def test_strict_central_gradient_mae_excludes_incomplete_reference_stencils():
    prediction = torch.arange(5, dtype=torch.float32).repeat(5, 1)[None, None]
    target = torch.zeros_like(prediction)
    artificial = _mask(5, 5)
    artificial[0, 2, 2] = True
    observed = torch.ones((1, 5, 5), dtype=torch.bool)
    observed[0, 2, 3] = False

    gradient_sum, gradient_count = supervised_gradient_error_terms(
        prediction, target, artificial, observed, norm_std=2.0
    )

    assert gradient_count == 0
    assert gradient_sum == 0.0


def test_boundary_error_uses_only_valid_observed_edges():
    prediction = torch.arange(4, dtype=torch.float32).repeat(3, 1)[None, None] * 0.5
    target = torch.zeros_like(prediction)
    artificial = _mask(3, 4)
    artificial[0, 1, 1] = True
    observed = torch.ones((1, 3, 4), dtype=torch.bool)

    boundary_sum, boundary_count = supervised_boundary_error_terms(
        prediction, target, artificial, observed, norm_std=2.0
    )

    assert boundary_count == 4
    assert boundary_sum == 2.0  # two x edges at 1 K/pixel


def test_natural_exceedance_counts_edges_above_monthly_threshold():
    field = torch.arange(5, dtype=torch.float32).repeat(3, 1)[None, None] * 2.0
    natural = torch.ones((1, 3, 5), dtype=torch.bool)
    land = torch.zeros_like(natural)

    exceed, count = natural_exceedance_terms(
        field, natural, land, torch.tensor([1.0]), norm_std=1.0
    )

    assert exceed == 12
    assert count == 22


def test_seam_exceedance_includes_temporal_proxy_preserved_pixels():
    field = torch.zeros((1, 1, 3, 5))
    field[0, 0, 1, 2] = 2.0
    natural = _mask(3, 5)
    natural[0, 1, 2] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)

    exceed, count = seam_exceedance_terms(
        field,
        natural,
        artificial,
        land,
        torch.tensor([1.0]),
        norm_std=1.0,
    )

    assert exceed == count == 4


def test_edge_difference_values_report_all_requested_physical_edges():
    field = torch.tensor([[[[0.0, 1.0, 3.0]]]])
    y_mask = torch.zeros((1, 1, 0, 3), dtype=torch.bool)
    x_mask = torch.tensor([[[[True, True]]]])

    values = edge_difference_values(field, y_mask, x_mask, norm_std=2.0)

    assert values.tolist() == pytest.approx([2.0, 4.0])


def test_local_outlier_rate_uses_natural_centres_and_robust_neighborhood():
    field = torch.zeros((1, 1, 9, 9))
    field[0, 0, 4, 4] = 3.0
    natural = _mask(9, 9)
    natural[0, 4, 4] = True
    land = _mask(9, 9)

    outliers, count = local_predicted_outlier_terms(
        field,
        natural,
        land,
        torch.tensor([1.0]),
        norm_std=1.0,
        window=9,
        min_valid=25,
    )

    assert outliers == count == 1


def test_epoch_accumulator_is_pixel_weighted_not_batch_weighted():
    metrics = EpochMetricAccumulator()
    metrics.add("absolute_error", numerator=1.0, count=1)
    metrics.add("absolute_error", numerator=9.0, count=3)
    metrics.add("squared_error", numerator=1.0, count=1)
    metrics.add("squared_error", numerator=27.0, count=3)
    metrics.add("gradient_error", numerator=8.0, count=4)
    metrics.add("boundary_error", numerator=3.0, count=2)
    metrics.add("natural_exceedance", numerator=2.0, count=10)
    metrics.add("seam_exceedance", numerator=1.0, count=4)
    metrics.add("local_outlier", numerator=1.0, count=20)
    metrics.add("low_energy_ratio", numerator=6.0, count=3)

    result = metrics.finalize()

    assert result["mae"] == pytest.approx(2.5)
    assert result["rmse"] == pytest.approx(math.sqrt(7.0))
    assert result["gradient_mae"] == pytest.approx(2.0)
    assert result["boundary_mae"] == pytest.approx(1.5)
    assert result["natural_exceedance_rate"] == pytest.approx(0.2)
    assert result["seam_exceedance_rate"] == pytest.approx(0.25)
    assert result["local_outlier_rate"] == pytest.approx(0.05)
    assert result["low_energy_ratio"] == pytest.approx(2.0)


def test_empty_metrics_finalize_to_none_not_misleading_zero():
    result = EpochMetricAccumulator().finalize()
    assert all(value is None for value in result.values())
