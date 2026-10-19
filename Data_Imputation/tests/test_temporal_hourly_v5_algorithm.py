from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4 import algorithm as v4_algorithm
from Data_Imputation.preprocessing.temporal_hourly_v4.source import RawFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.state import KNN, RAW, TEMPORAL, UNRESOLVED, ProxyFrame
from Data_Imputation.preprocessing.temporal_hourly_v5 import algorithm as v5_algorithm
from Data_Imputation.preprocessing.temporal_hourly_v5.algorithm import batch_weighted_estimate
from Data_Imputation.preprocessing.temporal_hourly_v5.cache import RollingCandidateCache


def test_batch_weighted_estimate_matches_v4_for_invalid_and_tied_candidates():
    values = np.array(
        [[10.0, 12.0, np.nan, 999.0], [5.0, 8.0, 9.0, 11.0], [np.nan] * 4]
    )
    confidence = np.array(
        [[1.0, 0.5, 0.8, 0.0], [0.6, 0.6, 0.6, np.nan], [0.0] * 4]
    )
    distance = np.array(
        [[1.0, 1.0, np.inf, 1.0], [2.0, 2.0, 2.0, 2.0], [1.0] * 4]
    )
    delta = np.array(
        [[0.0, 1.0, 2.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0] * 4]
    )

    estimates, quality, present = batch_weighted_estimate(
        values, confidence, distance, delta, limit=2
    )

    for row in range(values.shape[0]):
        expected, expected_quality = v4_algorithm._weighted_estimate(
            values[row], confidence[row], distance[row], delta[row], limit=2
        )
        assert bool(present[row]) is (expected is not None)
        if expected is not None:
            assert estimates[row] == pytest.approx(expected, abs=1e-12)
            assert quality[row] == pytest.approx(expected_quality, abs=1e-12)
        else:
            assert np.isnan(estimates[row])
            assert quality[row] == 0.0


def _raw_frame(
    points: dict[tuple[int, int], float],
    *,
    timestamp: datetime,
    shape: tuple[int, int],
) -> RawFrame:
    values = np.full(shape, np.nan, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    for (y, x), value in points.items():
        values[y, x] = value
        observed[y, x] = True
    return RawFrame(
        timestamp=timestamp,
        values=values,
        observed=observed,
        land_mask=np.zeros(shape, dtype=bool),
    )


def _proxy_frame(
    points: dict[tuple[int, int], float],
    *,
    timestamp: datetime,
    shape: tuple[int, int],
    confidence: float = 0.8,
) -> ProxyFrame:
    values = np.full(shape, np.nan, dtype=np.float32)
    source_code = np.full(shape, UNRESOLVED, dtype=np.uint8)
    confidences = np.zeros(shape, dtype=np.float32)
    for (y, x), value in points.items():
        values[y, x] = value
        source_code[y, x] = KNN
        confidences[y, x] = confidence
    return ProxyFrame(
        timestamp=timestamp,
        values=values,
        source_code=source_code,
        confidence=confidences,
        temporal_fill_mask=source_code == TEMPORAL,
        knn_fill_mask=source_code == KNN,
        land_mask=np.zeros(shape, dtype=bool),
    )


def _mixed_inputs():
    shape = (3, 4)
    values = np.array(
        [[10.0, np.nan, np.nan, 15.0], [11.0, np.nan, np.nan, 16.0], [12.0, np.nan, np.nan, 17.0]],
        dtype=np.float32,
    )
    source_code = np.full(shape, UNRESOLVED, dtype=np.uint8)
    source_code[:, (0, 3)] = RAW
    confidence = np.zeros(shape, dtype=np.float32)
    confidence[:, (0, 3)] = 1.0
    raw_history = [
        (
            1,
            _raw_frame(
                {(0, 1): 10.5, (1, 1): 11.5, (2, 1): 12.5},
                timestamp=datetime(2024, 1, 1, 1),
                shape=shape,
            ),
        ),
        (
            3,
            _raw_frame(
                {(0, 2): 13.5, (1, 2): 14.5, (2, 2): 15.5},
                timestamp=datetime(2023, 12, 31, 23),
                shape=shape,
            ),
        ),
    ]
    proxy_history = [
        (
            1,
            _proxy_frame(
                {(0, 2): 14.0, (1, 2): 15.0, (2, 2): 16.0},
                timestamp=datetime(2024, 1, 1, 1),
                shape=shape,
            ),
        )
    ]
    return dict(
        values=values,
        source_code=source_code,
        confidence=confidence,
        land_mask=np.zeros(shape, dtype=bool),
        raw_history=raw_history,
        proxy_history=proxy_history,
        query_workers=1,
        k=3,
        source_k=3,
        target_timestamp=datetime(2024, 1, 1, 2),
    )


@pytest.mark.parametrize("rebuild_interval", [1, 2])
def test_v5_progressive_knn_matches_v4_with_raw_and_proxy_history(rebuild_interval):
    kwargs = _mixed_inputs()
    expected = v4_algorithm.progressive_confidence_knn(
        **kwargs, rebuild_interval=rebuild_interval
    )
    actual = v5_algorithm.progressive_confidence_knn(
        **kwargs, rebuild_interval=rebuild_interval
    )

    np.testing.assert_array_equal(actual.source_code, expected.source_code)
    np.testing.assert_array_equal(actual.knn_fill_mask, expected.knn_fill_mask)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6, rtol=0, equal_nan=True)
    np.testing.assert_allclose(actual.confidence, expected.confidence, atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual.local_proxy_offset, expected.local_proxy_offset, atol=1e-6, rtol=0)


def test_v5_excludes_zero_distance_proxy_and_raises_without_any_seed():
    zero_distance = v5_algorithm.progressive_confidence_knn(
        values=np.array([[10.0, np.nan]], dtype=np.float32),
        source_code=np.array([[RAW, UNRESOLVED]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.0]], dtype=np.float32),
        land_mask=np.zeros((1, 2), dtype=bool),
        raw_history=[],
        proxy_history=[
            (
                1,
                _proxy_frame(
                    {(0, 1): 999.0},
                    timestamp=datetime(2024, 1, 1),
                    shape=(1, 2),
                ),
            )
        ],
        query_workers=1,
        k=1,
        rebuild_interval=1,
    )
    assert zero_distance.values[0, 1] == pytest.approx(10.0)
    assert zero_distance.values[0, 1] != pytest.approx(999.0)

    with pytest.raises(RuntimeError, match="no valid interpolation seed"):
        v5_algorithm.progressive_confidence_knn(
            values=np.full((2, 2), np.nan, dtype=np.float32),
            source_code=np.full((2, 2), UNRESOLVED, dtype=np.uint8),
            confidence=np.zeros((2, 2), dtype=np.float32),
            land_mask=np.zeros((2, 2), dtype=bool),
            raw_history=[],
            proxy_history=[],
            query_workers=1,
        )


def test_cached_history_trees_equal_freshly_built_history_trees():
    kwargs = _mixed_inputs()
    cache = RollingCandidateCache(max_history_hours=72)
    for _, frame in kwargs["raw_history"]:
        cache.add_raw(frame)
    for _, frame in kwargs["proxy_history"]:
        cache.add_proxy(frame)

    expected = v5_algorithm.progressive_confidence_knn(
        **kwargs, rebuild_interval=1
    )
    actual = v5_algorithm.progressive_confidence_knn(
        **kwargs,
        rebuild_interval=1,
        raw_tree_history=cache.raw_trees(kwargs["target_timestamp"]),
        proxy_tree_history=cache.proxy_trees(kwargs["target_timestamp"]),
    )

    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(actual.source_code, expected.source_code)
    np.testing.assert_allclose(actual.confidence, expected.confidence, atol=1e-6, rtol=0)
