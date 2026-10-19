from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4.algorithm import (
    blend_alpha,
    candidate_weights,
    evidence_quality,
    eligible_proxy_mask,
    causal_temporal_fill,
    gaussian_temporal_fills,
    progressive_confidence_knn,
)
from Data_Imputation.preprocessing.temporal_hourly_v4.source import RawFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.state import (
    KNN,
    RAW,
    TEMPORAL,
    UNRESOLVED,
    ProxyFrame,
)


def _raw_frame(points, *, timestamp=None, shape=(3, 3)):
    values = np.full(shape, np.nan, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    for (y, x), value in points.items():
        values[y, x] = value
        observed[y, x] = True
    return RawFrame(
        timestamp=timestamp or datetime(2024, 1, 1),
        values=values,
        observed=observed,
        land_mask=np.zeros(shape, dtype=bool),
    )


def _missing_frame(*, timestamp=None, shape=(3, 3)):
    return _raw_frame({}, timestamp=timestamp, shape=shape)


def _proxy_frame(points, *, timestamp=None, shape=(3, 3), confidence=0.8):
    values = np.full(shape, np.nan, dtype=np.float32)
    source_code = np.full(shape, UNRESOLVED, dtype=np.uint8)
    confidences = np.zeros(shape, dtype=np.float32)
    for (y, x), value in points.items():
        values[y, x] = value
        source_code[y, x] = KNN
        confidences[y, x] = confidence
    return ProxyFrame(
        timestamp=timestamp or datetime(2024, 1, 1),
        values=values,
        source_code=source_code,
        confidence=confidences,
        temporal_fill_mask=source_code == TEMPORAL,
        knn_fill_mask=source_code == KNN,
        land_mask=np.zeros(shape, dtype=bool),
    )


def test_temporal_fill_uses_only_raw_same_location_and_actual_lags():
    target = _raw_frame({(1, 1): 0.0})
    history = [
        (1, _raw_frame({(0, 0): 10.0})),
        (3, _raw_frame({(0, 0): 30.0})),
    ]

    result = causal_temporal_fill(target, history, max_hours=72)

    assert result.values[1, 1] == pytest.approx(0.0)
    assert result.values[0, 0] == pytest.approx(15.0)
    assert result.source_code[0, 0] == TEMPORAL
    assert result.source_code[1, 1] == RAW


def test_temporal_confidence_is_about_point_32_for_one_raw_t_minus_one():
    result = causal_temporal_fill(
        _missing_frame(), [(1, _raw_frame({(0, 0): 10.0}))], max_hours=72
    )

    assert result.confidence[0, 0] == pytest.approx(0.321, abs=0.002)


def test_temporal_fill_does_not_treat_missing_hour_as_adjacent():
    result = causal_temporal_fill(
        _missing_frame(), [(3, _raw_frame({(0, 0): 30.0}))], max_hours=2
    )

    assert np.isnan(result.values[0, 0])
    assert result.residual_missing[0, 0]


def test_gaussian_changes_only_temporal_pixels_and_preserves_residual_nan():
    values = np.array(
        [[1.0, 2.0, np.nan], [3.0, 4.0, np.nan], [np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )
    source_code = np.array(
        [[RAW, RAW, 0], [RAW, TEMPORAL, 0], [0, 0, 0]], dtype=np.uint8
    )
    temporal_fill_mask = source_code == TEMPORAL
    residual_missing = np.isnan(values)

    smoothed = gaussian_temporal_fills(
        values,
        source_code,
        temporal_fill_mask,
        residual_missing,
        sigma=1.5,
    )

    assert smoothed[0, 0] == 1.0
    assert smoothed[0, 1] == 2.0
    assert smoothed[1, 0] == 3.0
    assert smoothed[1, 1] != 4.0
    assert np.isnan(smoothed[0, 2])


def test_gaussian_temporal_fills_does_not_use_global_mean_for_missing_support():
    values = np.asarray(
        [[300.0, 310.0, 320.0], [301.0, 302.0, 321.0], [303.0, 304.0, 322.0]],
        dtype=np.float32,
    )
    codes = np.full(values.shape, RAW, dtype=np.uint8)
    codes[0, 1] = TEMPORAL
    temporal = codes == TEMPORAL
    residual = np.zeros(values.shape, dtype=bool)
    residual[2, 2] = True
    values[2, 2] = np.nan

    smoothed = gaussian_temporal_fills(values, codes, temporal, residual, sigma=0.8)

    assert np.isfinite(smoothed[0, 1])
    assert 300.0 <= smoothed[0, 1] <= 322.0


def test_proxy_candidates_require_synthetic_source_confidence_and_nonzero_distance():
    source_code = np.array([TEMPORAL, KNN, RAW, KNN], dtype=np.uint8)
    confidence = np.array([0.20, 0.19, 1.0, 0.85], dtype=np.float32)
    distance = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    eligible = eligible_proxy_mask(source_code, confidence, distance)

    assert eligible.tolist() == [True, False, False, False]


def test_candidate_weights_decay_with_space_and_actual_time_lag():
    weights = candidate_weights(
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 24.0]),
    )

    assert weights[0] > weights[1]


def test_alpha_decreases_when_base_evidence_is_strong():
    assert blend_alpha(proxy_quality=0.9, base_quality=0.9) == pytest.approx(0.0585)
    assert blend_alpha(proxy_quality=0.9, base_quality=0.0) == pytest.approx(0.585)


def test_evidence_quality_is_bounded_and_uses_candidate_confidence():
    quality = evidence_quality(
        weights=np.array([1.0, 0.5]),
        confidence=np.array([0.8, 0.4]),
        distance=np.array([1.0, 3.0]),
        delta_hours=np.array([0.0, 1.0]),
    )

    assert 0.0 < quality < 1.0


def test_progressive_knn_completes_connected_ocean_gap_and_decays_confidence():
    result = progressive_confidence_knn(
        values=np.array([[10.0, np.nan, np.nan, np.nan]], dtype=np.float32),
        source_code=np.array([[RAW, UNRESOLVED, UNRESOLVED, UNRESOLVED]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        land_mask=np.zeros((1, 4), dtype=bool),
        raw_history=[],
        proxy_history=[],
        k=1,
        rebuild_interval=1,
        query_workers=1,
    )

    assert np.all(np.isfinite(result.values))
    assert result.source_code[0, 0] == RAW
    assert np.all(result.source_code[0, 1:] == KNN)
    assert result.confidence[0, 3] < result.confidence[0, 1]


def test_historical_proxy_can_change_only_residual_missing_pixels():
    values = np.array([[10.0, 20.0, np.nan]], dtype=np.float32)
    source_code = np.array([[RAW, TEMPORAL, UNRESOLVED]], dtype=np.uint8)
    result = progressive_confidence_knn(
        values=values,
        source_code=source_code,
        confidence=np.array([[1.0, 0.5, 0.0]], dtype=np.float32),
        land_mask=np.zeros((1, 3), dtype=bool),
        raw_history=[],
        proxy_history=[(1, _proxy_frame({(0, 2): 99.0}, shape=(1, 3)))],
        k=1,
        rebuild_interval=1,
        query_workers=1,
    )

    assert result.values[0, 0] == pytest.approx(10.0)
    assert result.values[0, 1] == pytest.approx(20.0)
    assert result.source_code[0, 0] == RAW
    assert result.source_code[0, 1] == TEMPORAL
    assert result.source_code[0, 2] == KNN


def test_zero_distance_historical_synthetic_value_is_not_copied():
    result = progressive_confidence_knn(
        values=np.array([[10.0, np.nan]], dtype=np.float32),
        source_code=np.array([[RAW, UNRESOLVED]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.0]], dtype=np.float32),
        land_mask=np.zeros((1, 2), dtype=bool),
        raw_history=[],
        proxy_history=[(1, _proxy_frame({(0, 1): 999.0}, shape=(1, 2)))],
        k=1,
        rebuild_interval=1,
        query_workers=1,
    )

    assert result.values[0, 1] == pytest.approx(10.0)
    assert result.values[0, 1] != pytest.approx(999.0)


def test_proxy_offset_is_zero_without_five_local_raw_anchors():
    result = progressive_confidence_knn(
        values=np.array([[10.0, np.nan]], dtype=np.float32),
        source_code=np.array([[RAW, UNRESOLVED]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.0]], dtype=np.float32),
        land_mask=np.zeros((1, 2), dtype=bool),
        raw_history=[],
        proxy_history=[(1, _proxy_frame({(0, 0): 30.0}, shape=(1, 2)))],
        k=1,
        rebuild_interval=1,
        query_workers=1,
    )

    assert result.local_proxy_offset[0, 1] == pytest.approx(0.0)


def test_no_raw_or_proxy_seed_raises_instead_of_writing_mean_or_nan():
    with pytest.raises(RuntimeError, match="no valid interpolation seed"):
        progressive_confidence_knn(
            values=np.full((2, 2), np.nan, dtype=np.float32),
            source_code=np.full((2, 2), UNRESOLVED, dtype=np.uint8),
            confidence=np.zeros((2, 2), dtype=np.float32),
            land_mask=np.zeros((2, 2), dtype=bool),
            raw_history=[],
            proxy_history=[],
            query_workers=1,
        )
