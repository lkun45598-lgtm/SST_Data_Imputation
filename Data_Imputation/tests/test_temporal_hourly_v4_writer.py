from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4.source import SourceIndex
from Data_Imputation.preprocessing.temporal_hourly_v4.state import KNN, RAW, ProxyFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.writer import (
    SeriesH5Writer,
    reserve_output_root,
)


def _write_source(path: Path, timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.array([[[20.0, np.nan], [21.0, np.nan]]], dtype=np.float32)
    observed = np.array([[[1, 0], [1, 0]]], dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sst_data", data=values)
        handle.create_dataset("original_obs_mask", data=observed)
        handle.create_dataset("land_mask", data=np.zeros((2, 2), dtype=np.uint8))
        handle.create_dataset("latitude", data=np.array([1.0, 2.0], dtype=np.float32))
        handle.create_dataset("longitude", data=np.array([3.0, 4.0], dtype=np.float32))
        handle.create_dataset("timestamps", data=np.asarray([timestamp.encode("ascii")], dtype="S32"))


def _completed_frame(timestamp: datetime) -> ProxyFrame:
    return ProxyFrame(
        timestamp=timestamp,
        values=np.array([[20.0, 20.5], [21.0, 21.5]], dtype=np.float32),
        source_code=np.array([[RAW, KNN], [RAW, KNN]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.4], [1.0, 0.3]], dtype=np.float32),
        temporal_fill_mask=np.zeros((2, 2), dtype=bool),
        knn_fill_mask=np.array([[0, 1], [0, 1]], dtype=bool),
        land_mask=np.zeros((2, 2), dtype=bool),
    )


def test_writer_matches_legacy_hour_directory_and_core_dataset_schema(tmp_path):
    source_root = tmp_path / "source"
    timestamp = datetime(2024, 1, 1, 12)
    _write_source(
        source_root / "h12" / "jaxa_knn_filled_00.h5",
        timestamp.isoformat(),
    )
    index = SourceIndex.from_hour_dirs(source_root, series_id=0)
    output_root = tmp_path / "output"
    reserve_output_root(output_root)

    writer = SeriesH5Writer.create(output_root, index, series_id=0)
    writer.write(
        _completed_frame(timestamp),
        raw_mask=np.array([[1, 0], [1, 0]], dtype=bool),
        original_missing_mask=np.array([[0, 1], [0, 1]], dtype=bool),
    )
    writer.close()

    with h5py.File(output_root / "h12" / "jaxa_knn_filled_00.h5", "r") as handle:
        assert handle["sst_data"].dtype == np.dtype("float32")
        assert handle["timestamps"].dtype == np.dtype("S32")
        assert handle["sst_data"].compression == "gzip"
        assert handle["sst_data"].compression_opts == 4
        assert handle["proxy_confidence"].dtype == np.dtype("float16")
        assert {
            "source_code",
            "proxy_confidence",
            "knn_fill_mask",
            "unresolved_mask",
        } <= set(handle)
        assert handle.attrs["hour_offset"] == 12
        assert handle.attrs["series_id"] == 0
        assert handle["timestamps"][0] == timestamp.isoformat().encode("ascii")
        assert not handle["unresolved_mask"][0].any()


def test_writer_refuses_existing_hour_directory(tmp_path):
    (tmp_path / "h00").mkdir()

    with pytest.raises(FileExistsError, match="non-empty output root"):
        reserve_output_root(tmp_path)


def test_writer_rejects_unresolved_ocean_pixels_on_close(tmp_path):
    source_root = tmp_path / "source"
    timestamp = datetime(2024, 1, 1, 12)
    _write_source(
        source_root / "h12" / "jaxa_knn_filled_00.h5",
        timestamp.isoformat(),
    )
    index = SourceIndex.from_hour_dirs(source_root, series_id=0)
    output_root = tmp_path / "output"
    reserve_output_root(output_root)
    frame = _completed_frame(timestamp)
    frame = ProxyFrame(
        timestamp=frame.timestamp,
        values=np.array([[20.0, np.nan], [21.0, 21.5]], dtype=np.float32),
        source_code=frame.source_code,
        confidence=frame.confidence,
        temporal_fill_mask=frame.temporal_fill_mask,
        knn_fill_mask=frame.knn_fill_mask,
        land_mask=frame.land_mask,
    )

    writer = SeriesH5Writer.create(output_root, index, series_id=0)
    writer.write(
        frame,
        raw_mask=np.array([[1, 0], [1, 0]], dtype=bool),
        original_missing_mask=np.array([[0, 1], [0, 1]], dtype=bool),
    )

    with pytest.raises(RuntimeError, match="unresolved ocean pixels"):
        writer.close()
