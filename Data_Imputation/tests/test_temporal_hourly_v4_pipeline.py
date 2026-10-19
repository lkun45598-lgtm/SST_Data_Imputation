from datetime import datetime
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4.pipeline import run_series
from Data_Imputation.preprocessing.temporal_hourly_v4.cli import main


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


def test_pipeline_reconstructs_all_present_hours_in_timestamp_order(tmp_path):
    source_root = tmp_path / "source"
    _write_hour(
        source_root / "h00" / "jaxa_knn_filled_00.h5",
        datetime(2024, 1, 1, 0),
        np.array([[10.0, np.nan, np.nan]], dtype=np.float32),
        np.array([[1, 0, 0]], dtype=np.uint8),
    )
    _write_hour(
        source_root / "h01" / "jaxa_knn_filled_00.h5",
        datetime(2024, 1, 1, 1),
        np.array([[11.0, np.nan, np.nan]], dtype=np.float32),
        np.array([[1, 0, 0]], dtype=np.uint8),
    )
    output_root = tmp_path / "output"

    summary = run_series(source_root, output_root, series_id=0, query_workers=1)

    assert summary.processed_timestamps == tuple(sorted(summary.processed_timestamps))
    assert summary.processed_timestamps == (
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 1),
    )
    assert (output_root / "h00" / "jaxa_knn_filled_00.h5").exists()
    assert (output_root / "h01" / "jaxa_knn_filled_00.h5").exists()
    with h5py.File(output_root / "h01" / "jaxa_knn_filled_00.h5", "r") as handle:
        assert np.isfinite(handle["sst_data"][:]).all()


def test_pipeline_never_places_low_confidence_knn_in_next_hour_proxy_history(tmp_path):
    source_root = tmp_path / "source"
    first_values = np.array([[10.0] + [np.nan] * 7], dtype=np.float32)
    first_observed = np.array([[1] + [0] * 7], dtype=np.uint8)
    _write_hour(
        source_root / "h00" / "jaxa_knn_filled_00.h5",
        datetime(2024, 1, 1, 0),
        first_values,
        first_observed,
    )
    _write_hour(
        source_root / "h01" / "jaxa_knn_filled_00.h5",
        datetime(2024, 1, 1, 1),
        first_values,
        first_observed,
    )

    summary = run_series(source_root, tmp_path / "output", series_id=0, query_workers=1)

    assert summary.proxy_history_rejected_low_confidence > 0


def test_pipeline_smoke_endpoint_writes_only_processed_timestamp_slots(tmp_path):
    source_root = tmp_path / "source"
    for hour in (0, 1):
        _write_hour(
            source_root / f"h{hour:02d}" / "jaxa_knn_filled_00.h5",
            datetime(2024, 1, 1, hour),
            np.array([[10.0, np.nan]], dtype=np.float32),
            np.array([[1, 0]], dtype=np.uint8),
        )
    output_root = tmp_path / "output"

    summary = run_series(
        source_root,
        output_root,
        series_id=0,
        query_workers=1,
        stop_after=datetime(2024, 1, 1, 0),
    )

    assert summary.processed_timestamps == (datetime(2024, 1, 1, 0),)
    assert (output_root / "h00" / "jaxa_knn_filled_00.h5").exists()
    assert not (output_root / "h01" / "jaxa_knn_filled_00.h5").exists()


def test_smoke_command_writes_summary_with_required_audit_fields(tmp_path):
    source_root = tmp_path / "source"
    _write_hour(
        source_root / "h00" / "jaxa_knn_filled_00.h5",
        datetime(2024, 1, 1, 0),
        np.array([[10.0, np.nan]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.uint8),
    )
    output_root = tmp_path / "output"

    exit_code = main(
        [
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--series",
            "0",
            "--smoke-until",
            "2024-01-01T00:00:00",
            "--query-workers",
            "1",
        ]
    )

    summary = json.loads((output_root / "diagnostics" / "series_00.json").read_text())
    assert {
        "raw_fraction",
        "temporal_fraction",
        "knn_fraction",
        "proxy_history_fraction",
        "mean_confidence",
        "unresolved_count",
    } <= set(summary["frames"][0])
    assert exit_code == 0
