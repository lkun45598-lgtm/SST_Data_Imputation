from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v3.algorithm import (
    causal_temporal_fill,
    gaussian_temporal_fills,
)
from Data_Imputation.preprocessing.temporal_hourly_v3.source import SourceFrame


def raw_frame(points, timestamp=None, shape=(3, 3)):
    values = np.full(shape, np.nan, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    for (y, x), value in points.items():
        values[y, x] = value
        observed[y, x] = True
    return SourceFrame(
        timestamp=timestamp or datetime(2024, 1, 1),
        values=values,
        observed=observed,
        land_mask=np.zeros(shape, dtype=bool),
    )


def missing_frame(timestamp=None, shape=(3, 3)):
    return SourceFrame(
        timestamp=timestamp or datetime(2024, 1, 1),
        values=np.full(shape, np.nan, dtype=np.float32),
        observed=np.zeros(shape, dtype=bool),
        land_mask=np.zeros(shape, dtype=bool),
    )


def test_temporal_fill_uses_same_location_and_inverse_actual_lag():
    target = raw_frame({(1, 1): 0.0})
    history = [
        (1, raw_frame({(0, 0): 10.0})),
        (3, raw_frame({(0, 0): 30.0})),
    ]

    result = causal_temporal_fill(target, history, max_hours=72)

    expected = (10.0 / 1.0 + 30.0 / 3.0) / (1.0 + 1.0 / 3.0)
    assert result.values[1, 1] == pytest.approx(0.0)
    assert result.values[0, 0] == pytest.approx(expected)
    assert result.temporal_fill_mask[0, 0]


def test_temporal_fill_does_not_use_future_or_gap_as_adjacency():
    target = missing_frame()
    history = [(3, raw_frame({(0, 0): 30.0}))]

    result = causal_temporal_fill(target, history, max_hours=2)

    assert np.isnan(result.values[0, 0])
    assert result.residual_missing[0, 0]


def test_gaussian_only_writes_temporal_fill_pixels():
    data = np.array(
        [[1.0, 2.0, np.nan], [3.0, 4.0, np.nan], [np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )
    residual_missing = np.isnan(data)
    temporal_fill_mask = np.zeros_like(data, dtype=bool)
    temporal_fill_mask[1, 1] = True

    result = gaussian_temporal_fills(
        data, residual_missing, temporal_fill_mask, sigma=1.5
    )

    assert result[0, 0] == 1.0
    assert np.isnan(result[0, 2])
    assert result[1, 1] != 4.0
