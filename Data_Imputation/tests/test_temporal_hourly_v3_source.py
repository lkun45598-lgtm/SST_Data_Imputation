from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v3.source import SourceIndex


def write_fixture(path, rows, shape=(1, 1)):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((len(rows), *shape), np.nan, dtype=np.float32)
    observed = np.zeros((len(rows), *shape), dtype=np.uint8)
    timestamps = []
    for i, (timestamp, value, is_observed) in enumerate(rows):
        timestamps.append(timestamp.encode("ascii"))
        values[i, 0, 0] = value
        observed[i, 0, 0] = int(is_observed)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sst_data", data=values)
        handle.create_dataset("original_obs_mask", data=observed)
        handle.create_dataset("land_mask", data=np.zeros(shape, dtype=np.uint8))
        handle.create_dataset("timestamps", data=np.asarray(timestamps, dtype="S32"))


def test_index_uses_timestamps_not_array_positions(tmp_path):
    write_fixture(tmp_path / "h00" / "jaxa_knn_filled_00.h5", [
        ("2024-01-01T00:00:00", 10.0, True),
        ("2024-01-01T03:00:00", 30.0, True),
    ])

    index = SourceIndex.from_hour_dirs(tmp_path, series_id=0)

    assert index.available_times == (
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 3),
    )
    assert index.lookup(datetime(2024, 1, 1, 1)) is None
    assert index.lookup(datetime(2024, 1, 1, 3)).timestamp.hour == 3


def test_only_original_observations_are_returned(tmp_path):
    write_fixture(tmp_path / "h00" / "jaxa_knn_filled_00.h5", [
        ("2024-01-01T00:00:00", 99.0, False),
    ])

    frame = SourceIndex.from_hour_dirs(tmp_path, series_id=0).read(
        datetime(2024, 1, 1, 0)
    )

    assert np.isnan(frame.values[0, 0])
    assert not frame.observed[0, 0]


def test_history_skips_missing_actual_hours(tmp_path):
    write_fixture(tmp_path / "h00" / "jaxa_knn_filled_00.h5", [
        ("2024-01-01T00:00:00", 10.0, True),
        ("2024-01-01T03:00:00", 30.0, True),
    ])
    index = SourceIndex.from_hour_dirs(tmp_path, series_id=0)

    history = index.history(datetime(2024, 1, 1, 4), max_hours=4)

    assert [delta for delta, _ in history] == [1, 4]
    assert [frame.timestamp.hour for _, frame in history] == [3, 0]
