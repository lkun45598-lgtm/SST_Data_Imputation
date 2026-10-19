from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4.source import RawFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.source import SourceIndex
from Data_Imputation.preprocessing.temporal_hourly_v4.state import (
    KNN,
    RAW,
    UNRESOLVED,
    ProxyFrame,
)
from Data_Imputation.preprocessing.temporal_hourly_v5.algorithm import (
    progressive_confidence_knn as run_v5_knn,
)
from Data_Imputation.preprocessing.temporal_hourly_v5.pipeline import run_series as run_v5_series
from Data_Imputation.preprocessing.temporal_hourly_v6.algorithm import (
    progressive_confidence_knn as run_v6_knn,
)
from Data_Imputation.preprocessing.temporal_hourly_v6.writer import (
    ResumableSeriesH5Writer,
    reserve_output_root,
)


def _raw_frame(timestamp: datetime, values: np.ndarray) -> RawFrame:
    values = np.asarray(values, dtype=np.float32)
    return RawFrame(
        timestamp=timestamp,
        values=values,
        observed=np.isfinite(values),
        land_mask=np.zeros(values.shape, dtype=bool),
    )


def _proxy_frame(timestamp: datetime, values: np.ndarray) -> ProxyFrame:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    codes = np.full(values.shape, UNRESOLVED, dtype=np.uint8)
    codes[finite] = KNN
    confidence = np.zeros(values.shape, dtype=np.float32)
    confidence[finite] = 0.6
    return ProxyFrame(
        timestamp=timestamp,
        values=values,
        source_code=codes,
        confidence=confidence,
        temporal_fill_mask=np.zeros(values.shape, dtype=bool),
        knn_fill_mask=finite,
        land_mask=np.zeros(values.shape, dtype=bool),
    )


def test_v6_parallel_knn_is_exactly_v5_equivalent():
    target = datetime(2024, 1, 4, 0)
    values = np.array(
        [
            [20.0, np.nan, np.nan, 23.0],
            [20.5, np.nan, np.nan, 23.5],
            [21.0, np.nan, np.nan, 24.0],
        ],
        dtype=np.float32,
    )
    codes = np.full(values.shape, UNRESOLVED, dtype=np.uint8)
    codes[:, (0, 3)] = RAW
    confidence = np.zeros(values.shape, dtype=np.float32)
    confidence[:, (0, 3)] = 1.0
    raw_history = []
    proxy_history = []
    for lag in range(1, 5):
        raw_values = np.full(values.shape, np.nan, dtype=np.float32)
        raw_values[:, 1] = np.array([20.2, 20.7, 21.2], dtype=np.float32) + lag / 10
        raw_history.append((lag, _raw_frame(target - timedelta(hours=lag), raw_values)))
        proxy_values = np.full(values.shape, np.nan, dtype=np.float32)
        proxy_values[:, 2] = np.array([21.8, 22.3, 22.8], dtype=np.float32) + lag / 20
        proxy_history.insert(0, (lag, _proxy_frame(target - timedelta(hours=lag), proxy_values)))

    kwargs = dict(
        values=values,
        source_code=codes,
        confidence=confidence,
        land_mask=np.zeros(values.shape, dtype=bool),
        raw_history=raw_history,
        proxy_history=proxy_history,
        query_workers=4,
        k=3,
        source_k=3,
        rebuild_interval=2,
        target_timestamp=target,
    )
    expected = run_v5_knn(**kwargs)
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = run_v6_knn(**kwargs, query_executor=executor)

    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.source_code, expected.source_code)
    np.testing.assert_array_equal(actual.confidence, expected.confidence)
    np.testing.assert_array_equal(actual.knn_fill_mask, expected.knn_fill_mask)
    np.testing.assert_array_equal(actual.local_proxy_offset, expected.local_proxy_offset)


def _write_source_series(root: Path, timestamps: list[datetime]) -> None:
    grouped: dict[int, list[datetime]] = {}
    for timestamp in timestamps:
        grouped.setdefault(timestamp.hour, []).append(timestamp)
    for hour, hour_times in grouped.items():
        path = root / f"h{hour:02d}" / "jaxa_knn_filled_00.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(
            [[[20.0 + index, np.nan], [21.0 + index, np.nan]] for index in range(len(hour_times))],
            dtype=np.float32,
        )
        observed = np.isfinite(values).astype(np.uint8)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("sst_data", data=values)
            handle.create_dataset("original_obs_mask", data=observed)
            handle.create_dataset("land_mask", data=np.zeros((2, 2), dtype=np.uint8))
            handle.create_dataset(
                "timestamps",
                data=np.asarray([item.isoformat().encode("ascii") for item in hour_times], dtype="S32"),
            )


def _completed_proxy(timestamp: datetime, offset: float) -> ProxyFrame:
    return ProxyFrame(
        timestamp=timestamp,
        values=np.array([[20.0 + offset, 20.5 + offset], [21.0 + offset, 21.5 + offset]], dtype=np.float32),
        source_code=np.array([[RAW, KNN], [RAW, KNN]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.4], [1.0, 0.3]], dtype=np.float32),
        temporal_fill_mask=np.zeros((2, 2), dtype=bool),
        knn_fill_mask=np.array([[0, 1], [0, 1]], dtype=bool),
        land_mask=np.zeros((2, 2), dtype=bool),
    )


def _write_proxy(writer: ResumableSeriesH5Writer, timestamp: datetime, offset: float) -> None:
    writer.write(
        _completed_proxy(timestamp, offset),
        raw_mask=np.array([[1, 0], [1, 0]], dtype=bool),
        original_missing_mask=np.array([[0, 1], [0, 1]], dtype=bool),
    )


def test_v6_writer_resumes_only_a_strict_consistent_prefix(tmp_path):
    timestamps = [datetime(2024, 1, day, 0) for day in (1, 2, 3)]
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_source_series(source_root, timestamps)
    index = SourceIndex.from_hour_dirs(source_root, series_id=0)
    reserve_output_root(output_root)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        with ResumableSeriesH5Writer.create(
            output_root,
            index,
            series_id=0,
            timestamps=timestamps,
        ) as writer:
            _write_proxy(writer, timestamps[0], 0.0)
            raise RuntimeError("simulated interruption")

    output_path = output_root / "h00" / "jaxa_knn_filled_00.h5"
    with h5py.File(output_path, "r+") as handle:
        handle["sst_data"][1] = _completed_proxy(timestamps[1], 1.0).values

    with ResumableSeriesH5Writer.resume(
        output_root,
        index,
        series_id=0,
        timestamps=timestamps,
    ) as writer:
        assert writer.completed_timestamps == (timestamps[0],)
        restored = writer.load_proxy_frame(timestamps[0])
        np.testing.assert_array_equal(restored.values, _completed_proxy(timestamps[0], 0.0).values)
        np.testing.assert_array_equal(
            restored.confidence,
            _completed_proxy(timestamps[0], 0.0).confidence,
        )
        _write_proxy(writer, timestamps[1], 1.0)
        _write_proxy(writer, timestamps[2], 2.0)

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["preprocess_version"] == "temporal_hourly_v6_parallel_resume"
        assert handle["resume_confidence_f32"].dtype == np.dtype("float32")
        assert np.all(np.asarray(handle["source_code"][:]) != UNRESOLVED)
        assert not np.asarray(handle["unresolved_mask"][:], dtype=bool).any()


def test_v6_resumed_pipeline_matches_uninterrupted_v5(tmp_path):
    from Data_Imputation.preprocessing.temporal_hourly_v6.pipeline import run_series as run_v6_series

    timestamps = [datetime(2024, 1, 1, hour) for hour in range(4)]
    source_root = tmp_path / "source"
    expected_root = tmp_path / "expected"
    resumed_root = tmp_path / "resumed"
    _write_source_series(source_root, timestamps)

    run_v5_series(source_root, expected_root, series_id=0, query_workers=1)
    run_v6_series(
        source_root,
        resumed_root,
        series_id=0,
        query_workers=4,
        stop_after=timestamps[1],
        preallocate_all=True,
    )

    summary = run_v6_series(
        source_root,
        resumed_root,
        series_id=0,
        query_workers=4,
        resume=True,
    )

    assert summary.processed_timestamps == tuple(timestamps)
    for timestamp in timestamps:
        expected_path = expected_root / f"h{timestamp.hour:02d}" / "jaxa_knn_filled_00.h5"
        actual_path = resumed_root / f"h{timestamp.hour:02d}" / "jaxa_knn_filled_00.h5"
        with h5py.File(expected_path, "r") as expected, h5py.File(actual_path, "r") as actual:
            for field in (
                "sst_data",
                "original_obs_mask",
                "temporal_fill_mask",
                "original_missing_mask",
                "knn_fill_mask",
                "unresolved_mask",
                "source_code",
                "proxy_confidence",
                "timestamps",
            ):
                np.testing.assert_array_equal(actual[field][:], expected[field][:])


def test_v6_pipeline_can_apply_bounded_proxy_smoothing_and_audit_it(tmp_path):
    from Data_Imputation.preprocessing.temporal_hourly_v6.pipeline import run_series as run_v6_series

    timestamps = [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1)]
    source_root = tmp_path / "source"
    for timestamp in timestamps:
        path = source_root / f"h{timestamp.hour:02d}" / "jaxa_knn_filled_00.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.array([[20.0, 50.0, 30.0]], dtype=np.float32)
        observed = np.ones((1, 3), dtype=np.uint8)
        if timestamp.hour == 1:
            values[0, 1] = np.nan
            observed[0, 1] = 0
        with h5py.File(path, "w") as handle:
            handle.create_dataset("sst_data", data=values[None])
            handle.create_dataset("original_obs_mask", data=observed[None])
            handle.create_dataset("land_mask", data=np.zeros((1, 3), dtype=np.uint8))
            handle.create_dataset(
                "timestamps",
                data=np.asarray([timestamp.isoformat().encode("ascii")], dtype="S32"),
            )

    baseline_root = tmp_path / "baseline"
    run_v6_series(source_root, baseline_root, series_id=0, query_workers=1)
    output_root = tmp_path / "smoothed"
    run_v6_series(
        source_root,
        output_root,
        series_id=0,
        query_workers=1,
        smooth_synthetic=True,
        max_change_k=0.10,
        stop_after=timestamps[0],
        preallocate_all=True,
    )

    run_v6_series(
        source_root,
        output_root,
        series_id=0,
        query_workers=1,
        smooth_synthetic=True,
        max_change_k=0.10,
        resume=True,
    )

    with h5py.File(output_root / "h01" / "jaxa_knn_filled_00.h5", "r") as handle:
        assert "proxy_smooth_mask" in handle
        assert "proxy_smooth_layer" in handle
        assert int(handle.attrs["proxy_smoothing_enabled"]) == 1
        assert int(handle["proxy_smooth_mask"][0, 0, 1]) == 1
        assert float(handle.attrs["proxy_smoothing_max_change_k"]) == 0.10
        assert np.isfinite(handle["sst_data"][:]).all()
        with h5py.File(baseline_root / "h01" / "jaxa_knn_filled_00.h5", "r") as baseline:
            before = float(baseline["sst_data"][0, 0, 1])
        assert abs(float(handle["sst_data"][0, 0, 1]) - before) <= 0.10 + 1e-6
