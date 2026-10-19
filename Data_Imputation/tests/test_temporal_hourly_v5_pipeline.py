from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4.pipeline import run_series as run_v4
from Data_Imputation.preprocessing.temporal_hourly_v4.algorithm import KNNResult
from Data_Imputation.preprocessing.temporal_hourly_v4.state import KNN, RAW
from Data_Imputation.eval_diag.benchmark_temporal_hourly_v5 import compare_completed_frames
from Data_Imputation.preprocessing.temporal_hourly_v5.pipeline import run_series as run_v5


def _write_hour(path: Path, timestamp: datetime, values: np.ndarray, observed: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sst_data", data=np.asarray(values, dtype=np.float32)[None, ...])
        handle.create_dataset(
            "original_obs_mask", data=np.asarray(observed, dtype=np.uint8)[None, ...]
        )
        handle.create_dataset("land_mask", data=np.zeros(values.shape, dtype=np.uint8))
        handle.create_dataset("latitude", data=np.arange(values.shape[0], dtype=np.float32))
        handle.create_dataset("longitude", data=np.arange(values.shape[1], dtype=np.float32))
        handle.create_dataset(
            "timestamps", data=np.asarray([timestamp.isoformat().encode("ascii")], dtype="S32")
        )


def _assert_h5_fields_equal(left: Path, right: Path) -> None:
    exact_fields = {
        "original_obs_mask",
        "temporal_fill_mask",
        "original_missing_mask",
        "knn_fill_mask",
        "unresolved_mask",
        "source_code",
        "timestamps",
    }
    float_fields = {"sst_data", "proxy_confidence"}
    with h5py.File(left, "r") as expected, h5py.File(right, "r") as actual:
        assert set(expected) == set(actual)
        for field in exact_fields:
            np.testing.assert_array_equal(actual[field][:], expected[field][:])
        for field in float_fields:
            np.testing.assert_allclose(actual[field][:], expected[field][:], atol=1e-6, rtol=0)


def test_v5_series_output_matches_v4_across_real_timestamp_gap(tmp_path):
    source_root = tmp_path / "source"
    values = np.array([[10.0, np.nan, np.nan, 13.0]], dtype=np.float32)
    observed = np.array([[1, 0, 0, 1]], dtype=np.uint8)
    for hour in (0, 1, 3):
        _write_hour(
            source_root / f"h{hour:02d}" / "jaxa_knn_filled_00.h5",
            datetime(2024, 1, 1, hour),
            values + np.float32(hour),
            observed,
        )
    v4_root = tmp_path / "v4"
    v5_root = tmp_path / "v5"

    expected = run_v4(
        source_root,
        v4_root,
        series_id=0,
        query_workers=1,
        stop_after=datetime(2024, 1, 1, 3),
    )
    actual = run_v5(
        source_root,
        v5_root,
        series_id=0,
        query_workers=1,
        stop_after=datetime(2024, 1, 1, 3),
    )

    assert actual.processed_timestamps == expected.processed_timestamps
    for hour in (0, 1, 3):
        _assert_h5_fields_equal(
            v4_root / f"h{hour:02d}" / "jaxa_knn_filled_00.h5",
            v5_root / f"h{hour:02d}" / "jaxa_knn_filled_00.h5",
        )
    with h5py.File(v5_root / "h03" / "jaxa_knn_filled_00.h5", "r") as handle:
        assert handle.attrs["preprocess_version"] == "temporal_hourly_v5_equivalent_acceleration"


def _completed_frame(value: float) -> KNNResult:
    return KNNResult(
        values=np.array([[value, value + 1.0]], dtype=np.float32),
        source_code=np.array([[RAW, KNN]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.4]], dtype=np.float32),
        knn_fill_mask=np.array([[False, True]], dtype=bool),
        local_proxy_offset=np.zeros((1, 2), dtype=np.float32),
    )


def test_compare_results_rejects_masks_and_accepts_small_float_rounding():
    left = _completed_frame(20.0)
    right = _completed_frame(20.0 + 5e-7)

    report = compare_completed_frames(left, right, atol=1e-6)

    assert report.passes
    right.source_code[0, 0] = KNN
    assert not compare_completed_frames(left, right, atol=1e-6).passes
