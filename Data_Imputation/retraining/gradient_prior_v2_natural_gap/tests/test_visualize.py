import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.visualize import (
    compose_denormalized_prediction,
    natural_gap_fraction,
    mask_categories,
)


def test_compose_denormalized_prediction_preserves_observed_and_land_pixels():
    raw = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]])
    last_input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    reconstruction = torch.tensor([[[True, False], [True, False]]])
    land = torch.tensor([[[False, False], [False, True]]])

    result = compose_denormalized_prediction(
        raw,
        last_input,
        reconstruction,
        land,
        mean=300.0,
        std=2.0,
    )

    expected = torch.tensor([[[[320.0, 304.0], [360.0, 308.0]]]])
    assert torch.equal(result, expected)


def test_mask_categories_distinguish_observed_natural_artificial_and_land():
    natural = np.array([[False, True], [False, False]])
    artificial = np.array([[False, False], [True, False]])
    land = np.array([[False, False], [False, True]])

    categories = mask_categories(natural, artificial, land)

    assert categories.tolist() == [[0, 1], [2, 3]]


def test_natural_gap_fraction_uses_ocean_pixels_only():
    natural = np.array([[True, False], [False, True]])
    land = np.array([[False, False], [False, True]])

    assert natural_gap_fraction(natural, land) == pytest.approx(1 / 3)
