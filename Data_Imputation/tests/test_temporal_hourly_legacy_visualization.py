from pathlib import Path
import sys

import numpy as np
import h5py
from scipy import ndimage


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from Data_Imputation.visualization.paper_figs.fig_temporal_hourly_legacy_4panel import (
        legacy_gaussian_fill,
        legacy_temporal_weighted_fill,
    )
except ModuleNotFoundError:
    legacy_gaussian_fill = None
    legacy_temporal_weighted_fill = None

try:
    from Data_Imputation.visualization.paper_figs.fig_temporal_hourly_legacy_4panel import (
        read_legacy_frame_batch,
    )
except ImportError:
    read_legacy_frame_batch = None


def test_legacy_temporal_fill_uses_observed_previous_days_with_inverse_day_weights():
    assert legacy_temporal_weighted_fill is not None, "legacy four-panel visualization is missing"

    raw = np.array([[np.nan, 9.0]], dtype=np.float32)
    observed = np.array([[False, True]])
    history = (
        (1, np.array([[10.0, 99.0]], dtype=np.float32), np.array([[True, False]])),
        (2, np.array([[20.0, 88.0]], dtype=np.float32), np.array([[True, True]])),
        (3, np.array([[30.0, 77.0]], dtype=np.float32), np.array([[False, True]])),
    )

    filled, fill_mask, residual_missing = legacy_temporal_weighted_fill(raw, observed, history)

    assert np.isclose(filled[0, 0], (10.0 + 20.0 / 2.0) / 1.5)
    assert filled[0, 1] == 9.0
    assert np.array_equal(fill_mask, np.array([[True, False]]))
    assert np.array_equal(residual_missing, np.array([[False, False]]))


def test_legacy_gaussian_writes_only_temporally_filled_pixels():
    assert legacy_gaussian_fill is not None, "legacy four-panel visualization is missing"

    stage1 = np.array([[10.0, 20.0], [30.0, np.nan]], dtype=np.float32)
    residual_missing = np.array([[False, False], [False, True]])
    fill_mask = np.array([[False, True], [False, False]])

    result = legacy_gaussian_fill(stage1, residual_missing, fill_mask, sigma=1.5)

    expected_input = stage1.copy()
    expected_input[residual_missing] = np.nanmean(stage1)
    expected = ndimage.gaussian_filter(expected_input, sigma=1.5)
    assert np.isclose(result[0, 1], expected[0, 1])
    assert result[0, 0] == stage1[0, 0]
    assert result[1, 0] == stage1[1, 0]
    assert np.isnan(result[1, 1])


def test_legacy_batch_reader_preserves_time_axis_for_a_single_selected_frame(tmp_path):
    assert read_legacy_frame_batch is not None, "legacy H5 batch reader is missing"

    path = tmp_path / "legacy.h5"
    expected = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("frames", data=expected)

    with h5py.File(path, "r") as handle:
        result = read_legacy_frame_batch(handle["frames"], (2,))

    assert result.shape == (1, 2, 3)
    assert np.array_equal(result[0], expected[2])
