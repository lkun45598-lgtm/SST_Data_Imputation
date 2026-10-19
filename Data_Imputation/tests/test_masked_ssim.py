import sys
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_IMPUTATION_DIR))

from baselines.structural_metrics import masked_window_ssim, reference_robust_range


def _field(size=15):
    y, x = np.mgrid[:size, :size]
    return (290.0 + 0.05 * x + 0.08 * y + 0.2 * np.sin(x / 2)).astype(np.float32)


def test_identical_fields_have_unit_ssim_on_eligible_centers():
    reference = _field()
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[5:10, 5:10] = True
    valid_reference = np.ones(reference.shape, dtype=bool)

    value, n_centers, n_candidates = masked_window_ssim(
        reference, reference.copy(), evaluation_mask, valid_reference,
        win_size=7, data_range=5.0,
    )

    assert value == 1.0
    assert n_centers == 25
    assert n_candidates == 25


def test_full_regular_window_contributes_to_ssim_at_extra_mask_centers():
    reference = _field()
    prediction = reference.copy()
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[5:10, 5:10] = True
    valid_reference = np.ones(reference.shape, dtype=bool)
    prediction[4, 4] += 2.0

    value, n_centers, n_candidates = masked_window_ssim(
        prediction, reference, evaluation_mask, valid_reference,
        win_size=7, data_range=5.0,
    )

    _, expected_map = structural_similarity(
        prediction.astype(np.float64), reference.astype(np.float64),
        win_size=7, data_range=5.0, full=True,
    )

    assert np.isclose(value, expected_map[evaluation_mask].mean())
    assert value < 1.0
    assert n_centers == 25
    assert n_candidates == 25


def test_single_extra_mask_center_is_eligible_with_a_complete_valid_window():
    reference = _field()
    prediction = reference.copy()
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[7, 7] = True
    valid_reference = np.ones(reference.shape, dtype=bool)

    value, n_centers, n_candidates = masked_window_ssim(
        prediction, reference, evaluation_mask, valid_reference,
        win_size=7, data_range=5.0,
    )

    assert value == 1.0
    assert n_centers == 1
    assert n_candidates == 1


def test_extra_mask_center_is_excluded_when_reference_window_is_invalid():
    reference = _field()
    prediction = reference.copy()
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[7, 7] = True
    valid_reference = np.ones(reference.shape, dtype=bool)
    valid_reference[4, 4] = False

    value, n_centers, n_candidates = masked_window_ssim(
        prediction, reference, evaluation_mask, valid_reference,
        win_size=7, data_range=5.0,
    )

    assert np.isnan(value)
    assert n_centers == 0
    assert n_candidates == 1


def test_reference_robust_range_uses_only_valid_reference_pixels():
    reference = np.arange(100, dtype=np.float32).reshape(10, 10)
    valid_reference = np.ones(reference.shape, dtype=bool)
    valid_reference[0, 0] = False
    reference[0, 0] = -10000.0

    value = reference_robust_range(reference, valid_reference)
    expected = np.percentile(reference[valid_reference], 99) - np.percentile(
        reference[valid_reference], 1
    )

    assert np.isclose(value, expected)


def test_structure_error_inside_selected_window_reduces_ssim():
    reference = _field()
    prediction = reference.copy()
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[4:11, 4:11] = True
    valid_reference = np.ones(reference.shape, dtype=bool)
    prediction[5:10, 5:10] = prediction[5:10, 5:10].mean()

    value, n_centers, n_candidates = masked_window_ssim(
        prediction, reference, evaluation_mask, valid_reference,
        win_size=7, data_range=5.0,
    )

    assert value < 1.0
    assert n_centers > 0
    assert n_candidates == 49


def test_nonfinite_prediction_excludes_every_window_that_touches_it():
    reference = _field()
    prediction = reference.copy()
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[5:10, 5:10] = True
    valid_reference = np.ones(reference.shape, dtype=bool)
    prediction[7, 7] = np.nan

    value, n_centers, n_candidates = masked_window_ssim(
        prediction, reference, evaluation_mask, valid_reference,
        win_size=7, data_range=5.0,
    )

    assert np.isnan(value)
    assert n_centers == 0
    assert n_candidates == 25


def test_result_matches_standard_ssim_map_at_extra_mask_centers():
    reference = _field()
    prediction = reference.copy()
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[5:10, 5:10] = True
    valid_reference = np.ones(reference.shape, dtype=bool)
    prediction[3:12, 3:12] += np.linspace(-0.4, 0.6, 81).reshape(9, 9)
    data_range = 5.0

    value, n_centers, n_candidates = masked_window_ssim(
        prediction, reference, evaluation_mask, valid_reference,
        win_size=7, data_range=data_range,
    )

    _, expected_map = structural_similarity(
        prediction.astype(np.float64), reference.astype(np.float64),
        win_size=7, data_range=data_range, full=True,
    )

    assert np.isclose(value, expected_map[evaluation_mask].mean())
    assert n_centers == 25
    assert n_candidates == 25


def test_rejects_even_window_size():
    reference = _field()
    mask = np.ones(reference.shape, dtype=bool)

    try:
        masked_window_ssim(reference, reference, mask, mask, win_size=6)
    except ValueError as exc:
        assert "odd" in str(exc)
    else:
        raise AssertionError("expected an even win_size to be rejected")
