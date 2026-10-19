from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v3.pipeline import run_series


def write_hour(path, timestamp, values, observed):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float32)[None, ...]
    observed = np.asarray(observed, dtype=np.uint8)[None, ...]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sst_data", data=values)
        handle.create_dataset("original_obs_mask", data=observed)
        handle.create_dataset("original_missing_mask", data=1 - observed)
        handle.create_dataset("temporal_fill_mask", data=np.zeros_like(observed))
        handle.create_dataset("land_mask", data=np.zeros(values.shape[1:], dtype=np.uint8))
        handle.create_dataset("latitude", data=np.arange(values.shape[1], dtype=np.float32))
        handle.create_dataset("longitude", data=np.arange(values.shape[2], dtype=np.float32))
        handle.create_dataset("timestamps", data=np.asarray([timestamp.encode()], dtype="S32"))


def make_hourly_fixture(root):
    shape = (4, 4)
    target_values = np.full(shape, np.nan, dtype=np.float32)
    target_observed = np.zeros(shape, dtype=bool)
    target_values[0, 0] = 300.0
    target_observed[0, 0] = True
    write_hour(
        root / "h12" / "jaxa_knn_filled_00.h5",
        "2024-01-02T12:00:00",
        target_values,
        target_observed,
    )

    history_values = np.full(shape, np.nan, dtype=np.float32)
    history_observed = np.zeros(shape, dtype=bool)
    history_values[0, 0] = 299.0
    history_observed[0, 0] = True
    write_hour(
        root / "h11" / "jaxa_knn_filled_00.h5",
        "2024-01-02T11:00:00",
        history_values,
        history_observed,
    )
    return root


def test_pipeline_writes_isolated_compatible_h5(tmp_path):
    source_root = make_hourly_fixture(tmp_path / "source")
    output_root = tmp_path / "output"

    run_series(
        source_root=source_root,
        output_root=output_root,
        series_id=0,
        target_hour=12,
        frame_limit=1,
        workers=2,
    )

    output_path = output_root / "h12" / "jaxa_knn_filled_00.h5"
    with h5py.File(output_path, "r") as handle:
        assert {
            "sst_data",
            "original_obs_mask",
            "original_missing_mask",
            "temporal_fill_mask",
            "knn_fill_mask",
            "unresolved_mask",
            "timestamps",
        }.issubset(handle.keys())
        sst = handle["sst_data"][:]
        land = handle["land_mask"][:].astype(bool)
        assert not np.any(np.isnan(sst[:, ~land]))
        assert handle.attrs["preprocess_version"] == "v3_temporal_hourly"
