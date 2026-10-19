from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4.source import SourceIndex


def _write_hour(path, rows, *, land_mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = land_mask.shape
    values = np.full((len(rows), height, width), np.nan, dtype=np.float32)
    observed = np.zeros_like(values, dtype=np.uint8)
    timestamps = []
    for row_index, (timestamp, points) in enumerate(rows):
        timestamps.append(timestamp.encode("ascii"))
        for (y, x), (value, is_observed) in points.items():
            values[row_index, y, x] = value
            observed[row_index, y, x] = int(is_observed)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sst_data", data=values)
        handle.create_dataset("original_obs_mask", data=observed)
        handle.create_dataset("land_mask", data=np.asarray(land_mask, dtype=np.uint8))
        handle.create_dataset("timestamps", data=np.asarray(timestamps, dtype="S32"))


def test_source_masks_legacy_fills_outside_original_observations(tmp_path):
    _write_hour(
        tmp_path / "h00" / "jaxa_knn_filled_00.h5",
        [("2024-01-01T00:00:00", {(0, 0): (300.0, True), (0, 1): (999.0, False)})],
        land_mask=np.zeros((2, 2), dtype=np.uint8),
    )

    frame = SourceIndex.from_hour_dirs(tmp_path, series_id=0).read(
        datetime(2024, 1, 1, 0)
    )

    assert frame.values[0, 0] == 300.0
    assert frame.observed[0, 0]
    assert np.isnan(frame.values[0, 1])
    assert not frame.observed[0, 1]


def test_raw_history_uses_real_timestamp_deltas_and_skips_missing_hours(tmp_path):
    _write_hour(
        tmp_path / "h00" / "jaxa_knn_filled_00.h5",
        [
            ("2024-01-01T00:00:00", {(0, 0): (10.0, True)}),
            ("2024-01-01T03:00:00", {(0, 0): (30.0, True)}),
        ],
        land_mask=np.zeros((1, 1), dtype=np.uint8),
    )
    index = SourceIndex.from_hour_dirs(tmp_path, series_id=0)

    history = index.raw_history(datetime(2024, 1, 1, 4), max_hours=72)

    assert [delta for delta, _ in history] == [1, 4]
    assert [frame.timestamp.hour for _, frame in history] == [3, 0]


def test_source_uses_conservative_land_mask_union_across_hours(tmp_path):
    _write_hour(
        tmp_path / "h00" / "jaxa_knn_filled_00.h5",
        [("2024-01-01T00:00:00", {(0, 0): (10.0, True)})],
        land_mask=np.zeros((1, 2), dtype=np.uint8),
    )
    _write_hour(
        tmp_path / "h01" / "jaxa_knn_filled_00.h5",
        [("2024-01-01T01:00:00", {(0, 0): (11.0, True)})],
        land_mask=np.array([[0, 1]], dtype=np.uint8),
    )

    index = SourceIndex.from_hour_dirs(tmp_path, series_id=0)

    assert index.land_mask.tolist() == [[False, True]]
