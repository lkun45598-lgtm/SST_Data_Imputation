from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v3.algorithm import (
    progressive_spatiotemporal_knn,
)
from Data_Imputation.preprocessing.temporal_hourly_v3.source import SourceFrame


def frame_from_values(values, observed=None):
    values = np.asarray(values, dtype=np.float32)
    if observed is None:
        observed = np.isfinite(values)
    return SourceFrame(
        timestamp=datetime(2024, 1, 1),
        values=values,
        observed=np.asarray(observed, dtype=bool),
        land_mask=np.zeros(values.shape, dtype=bool),
    )


def missing_center_with_neighbors(left, right):
    values = np.full((3, 3), np.nan, dtype=np.float32)
    observed = np.zeros((3, 3), dtype=bool)
    values[1, 0] = left
    values[1, 2] = right
    observed[1, 0] = observed[1, 2] = True
    return frame_from_values(values, observed)


def missing_center_frame():
    values = np.full((3, 3), np.nan, dtype=np.float32)
    return frame_from_values(values, np.zeros((3, 3), dtype=bool))


def test_knn_prefers_current_spatial_seed_when_no_history_exists():
    frame = missing_center_with_neighbors(298.0, 299.0)

    filled, knn_mask = progressive_spatiotemporal_knn(
        frame.values, frame.observed, frame.land_mask, history=[], k=2
    )

    assert np.isfinite(filled[1, 1])
    assert knn_mask[1, 1]


def test_knn_uses_actual_hour_lag_for_history():
    target = missing_center_frame()
    history = [
        (1, frame_from_values([[np.nan, np.nan, np.nan], [300.0, np.nan, np.nan], [np.nan, np.nan, np.nan]])),
        (24, frame_from_values([[np.nan, np.nan, np.nan], [np.nan, np.nan, 280.0], [np.nan, np.nan, np.nan]])),
    ]

    filled, _ = progressive_spatiotemporal_knn(
        target.values, target.observed, target.land_mask, history=history, k=2
    )

    assert filled[1, 1] > 290.0


def test_knn_never_uses_historical_synthetic_values():
    values = np.full((3, 3), np.nan, dtype=np.float32)
    observed = np.zeros((3, 3), dtype=bool)
    values[0, 0] = 300.0
    observed[0, 0] = True
    target = frame_from_values(values, observed)
    synthetic_history = frame_from_values(
        [[np.nan, 999.0, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        np.zeros((3, 3), dtype=bool),
    )

    filled, _ = progressive_spatiotemporal_knn(
        target.values,
        target.observed,
        target.land_mask,
        history=[(1, synthetic_history)],
        k=1,
    )

    assert filled[1, 1] != pytest.approx(999.0)


def test_progressive_knn_completes_connected_gap():
    values = np.full((4, 4), np.nan, dtype=np.float32)
    observed = np.zeros((4, 4), dtype=bool)
    values[0, 0] = 300.0
    observed[0, 0] = True
    frame = frame_from_values(values, observed)

    filled, knn_mask = progressive_spatiotemporal_knn(
        frame.values,
        frame.observed,
        frame.land_mask,
        history=[],
        k=1,
        rebuild_interval=2,
    )

    assert np.all(np.isfinite(filled[~frame.land_mask]))
    assert knn_mask.sum() == int((~frame.observed & ~frame.land_mask).sum())


def test_knn_rejects_frame_with_no_any_candidate():
    target = missing_center_frame()
    with pytest.raises(RuntimeError, match="no valid interpolation seed"):
        progressive_spatiotemporal_knn(
            target.values,
            target.observed,
            target.land_mask,
            history=[],
            target_timestamp=datetime(2024, 1, 2),
        )
