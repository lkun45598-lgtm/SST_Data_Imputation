from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v6.boundary_smooth import (
    _normalized_gaussian_estimate,
    process_h5_file,
    process_synthetic_h5_file,
    process_hourly_root,
    smooth_boundary_frame,
    smooth_synthetic_frame,
)


def _frame_inputs():
    height, width = 7, 7
    land = np.zeros((height, width), dtype=bool)
    land[0, 0] = True
    observed = np.ones((height, width), dtype=bool)
    natural = np.zeros((height, width), dtype=bool)
    natural[2:5, 2:5] = True
    observed[natural] = False
    observed[land] = False
    values = np.full((height, width), 999.0, dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    values[observed] = (300.0 + 0.1 * yy[observed] + 0.2 * xx[observed]).astype(np.float32)
    values[land] = np.nan
    return values, observed, natural, land


def test_normalized_gaussian_uses_only_supported_values():
    values = np.asarray(
        [[10.0, np.nan, 30.0], [20.0, 25.0, 35.0], [15.0, 22.0, 32.0]],
        dtype=np.float32,
    )
    support = np.isfinite(values)

    estimate, valid = _normalized_gaussian_estimate(
        values, support, sigma=0.8
    )

    assert valid[0, 1]
    assert np.isfinite(estimate[0, 1])
    assert 10.0 <= estimate[0, 1] <= 35.0


def test_normalized_gaussian_does_not_create_values_without_support():
    values = np.full((3, 3), np.nan, dtype=np.float32)
    estimate, valid = _normalized_gaussian_estimate(
        values, np.zeros_like(values, dtype=bool), sigma=0.8
    )

    assert not valid.any()
    assert np.isnan(estimate).all()


def test_smooth_synthetic_frame_is_bounded_and_protects_observations():
    values, observed, natural, land = _frame_inputs()
    source_code = np.zeros(values.shape, dtype=np.uint8)
    source_code[observed] = 1
    source_code[natural] = 3
    source_code[3, 3] = 2

    smoothed, changed, layers = smooth_synthetic_frame(
        values,
        observed,
        natural,
        source_code,
        land,
        sigma=0.8,
        first_layer_weight=0.25,
        second_layer_weight=0.12,
        interior_weight=0.08,
        max_change_k=0.10,
    )

    synthetic = (source_code == 2) | (source_code == 3)
    assert np.array_equal(smoothed[observed], values[observed])
    assert np.array_equal(smoothed[land], values[land], equal_nan=True)
    assert np.all(changed <= synthetic)
    delta = np.abs(smoothed - values)
    assert np.nanmax(delta[synthetic]) <= 0.10
    assert changed.any()
    assert layers[2, 2] == 1
    assert layers[3, 3] == 2


def test_smooth_synthetic_frame_respects_cap_after_float32_rounding():
    values = np.full((5, 5), 300.0, dtype=np.float32)
    observed = np.ones((5, 5), dtype=bool)
    natural = np.zeros((5, 5), dtype=bool)
    observed[2, 2] = False
    natural[2, 2] = True
    values[2, 2] = 250.0
    source_code = np.ones((5, 5), dtype=np.uint8)
    source_code[2, 2] = 3
    land = np.zeros((5, 5), dtype=bool)

    smoothed, changed, _ = smooth_synthetic_frame(
        values,
        observed,
        natural,
        source_code,
        land,
        sigma=0.8,
        max_change_k=0.10,
    )

    assert changed[2, 2]
    assert abs(float(smoothed[2, 2]) - float(values[2, 2])) <= 0.10


def test_smooth_boundary_frame_changes_only_first_two_natural_layers():
    values, observed, natural, land = _frame_inputs()

    smoothed, changed, layers = smooth_boundary_frame(values, observed, natural, land)

    assert np.array_equal(layers == 1, np.array(
        [[False] * 7,
         [False] * 7,
         [False, False, True, True, True, False, False],
         [False, False, True, False, True, False, False],
         [False, False, True, True, True, False, False],
         [False] * 7,
         [False] * 7], dtype=bool
    ))
    assert layers[3, 3] == 2
    assert changed[3, 3]
    assert changed[2, 2]
    assert np.all(smoothed[observed] == values[observed])
    assert np.isnan(smoothed[land]).all()
    assert np.all(smoothed[natural & (layers == 0)] == values[natural & (layers == 0)])
    assert np.all(np.isfinite(smoothed[~land]))


def test_smooth_boundary_frame_does_not_create_values_without_observed_anchors():
    values = np.full((5, 5), 7.0, dtype=np.float32)
    observed = np.zeros((5, 5), dtype=bool)
    natural = np.ones((5, 5), dtype=bool)
    land = np.zeros((5, 5), dtype=bool)
    values[:] = np.nan

    smoothed, changed, layers = smooth_boundary_frame(values, observed, natural, land)

    assert np.array_equal(smoothed, values, equal_nan=True)
    assert not changed.any()
    assert not (layers > 0).any()


def test_smooth_boundary_frame_uses_local_estimate_when_old_proxy_is_nan():
    values, observed, natural, land = _frame_inputs()
    values[natural] = np.nan

    smoothed, changed, layers = smooth_boundary_frame(values, observed, natural, land)

    eligible = (layers > 0) & ~land
    assert eligible.any()
    assert changed[eligible].all()
    assert np.isfinite(smoothed[eligible]).all()
    assert np.isnan(smoothed[natural & (layers == 0)]).all()


def test_process_h5_file_preserves_schema_and_records_changes(tmp_path):
    source = tmp_path / "jaxa_knn_filled_00.h5"
    output_root = tmp_path / "out"
    output_root.mkdir()
    output = output_root / source.name
    values, observed, natural, land = _frame_inputs()
    with h5py.File(source, "w") as handle:
        handle.create_dataset("sst_data", data=values[None], compression="gzip")
        handle.create_dataset("original_obs_mask", data=observed[None].astype(np.uint8))
        handle.create_dataset("original_missing_mask", data=natural[None].astype(np.uint8))
        handle.create_dataset("land_mask", data=land.astype(np.uint8))
        handle.create_dataset("timestamps", data=np.asarray([b"2024-01-01T12:00:00"]))
        handle.create_dataset("latitude", data=np.arange(7, dtype=np.float32))
        handle.create_dataset("longitude", data=np.arange(7, dtype=np.float32))
        handle.attrs["series_id"] = 0

    stats = process_h5_file(source, output)

    assert stats["frames"] == 1
    assert stats["modified_pixels"] > 0
    with h5py.File(source, "r") as before, h5py.File(output, "r") as after:
        assert set(before.keys()).issubset(after.keys())
        assert np.array_equal(before["original_obs_mask"][:], after["original_obs_mask"][:])
        assert np.array_equal(before["original_missing_mask"][:], after["original_missing_mask"][:])
        assert np.array_equal(before["land_mask"][:], after["land_mask"][:])
        before_values = before["sst_data"][:][0]
        after_values = after["sst_data"][:][0]
        assert np.array_equal(before_values[observed], after_values[observed], equal_nan=True)
        assert np.array_equal(after["boundary_smooth_mask"][:].dtype, np.dtype("uint8"))
        assert after.attrs["boundary_smooth_version"] == "mask_aware_cubic_boundary_v1"
        assert int(after.attrs["boundary_smooth_complete"]) == 1


def test_process_synthetic_h5_file_preserves_observations_and_bounds_changes(tmp_path):
    source = tmp_path / "jaxa_knn_filled_08.h5"
    output = tmp_path / "proxy" / source.name
    values, observed, natural, land = _frame_inputs()
    source_code = np.zeros(values.shape, dtype=np.uint8)
    source_code[observed] = 1
    source_code[natural] = 3
    source_code[land] = 0
    with h5py.File(source, "w") as handle:
        handle.create_dataset("sst_data", data=values[None], compression="gzip")
        handle.create_dataset("original_obs_mask", data=observed[None].astype(np.uint8))
        handle.create_dataset("original_missing_mask", data=natural[None].astype(np.uint8))
        handle.create_dataset("source_code", data=source_code[None])
        handle.create_dataset("land_mask", data=land.astype(np.uint8))
        handle.create_dataset("timestamps", data=np.asarray([b"2024-01-01T12:00:00"]))
        handle.create_dataset("latitude", data=np.arange(7, dtype=np.float32))
        handle.create_dataset("longitude", data=np.arange(7, dtype=np.float32))

    stats = process_synthetic_h5_file(source, output, max_change_k=0.10)

    assert stats["frames"] == 1
    assert stats["modified_pixels"] > 0
    with h5py.File(source, "r") as before, h5py.File(output, "r") as after:
        before_values = before["sst_data"][:]
        after_values = after["sst_data"][:]
        assert np.array_equal(
            before_values[:, observed], after_values[:, observed], equal_nan=True
        )
        assert np.array_equal(
            before_values[:, land], after_values[:, land], equal_nan=True
        )
        delta = np.abs(after_values - before_values)
        assert np.nanmax(delta) <= 0.10 + 1e-6
        assert np.array_equal(after["original_obs_mask"][:], before["original_obs_mask"][:])
        assert np.array_equal(after["source_code"][:], before["source_code"][:])
        assert after["proxy_smooth_mask"].dtype == np.dtype("uint8")
        assert after["proxy_smooth_layer"].dtype == np.dtype("uint8")
        assert after.attrs["proxy_smoothing_version"] == "mask_aware_normalized_gaussian_v1"
        assert float(after.attrs["proxy_smoothing_max_change_k"]) == 0.10


def test_process_synthetic_h5_file_rejects_inconsistent_provenance(tmp_path):
    source = tmp_path / "bad.h5"
    output = tmp_path / "out.h5"
    values = np.array([[[20.0, 21.0]]], dtype=np.float32)
    observed = np.array([[[1, 0]]], dtype=np.uint8)
    source_code = np.array([[[3, 3]]], dtype=np.uint8)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("sst_data", data=values)
        handle.create_dataset("original_obs_mask", data=observed)
        handle.create_dataset("original_missing_mask", data=1 - observed)
        handle.create_dataset("source_code", data=source_code)
        handle.create_dataset("land_mask", data=np.zeros((1, 2), dtype=np.uint8))
        handle.create_dataset("timestamps", data=np.asarray([b"2024-01-01T00:00:00"]))

    with pytest.raises(ValueError, match="source_code.*RAW"):
        process_synthetic_h5_file(source, output)
    assert not output.exists()


def test_process_hourly_root_preserves_hour_layout_and_writes_manifest(tmp_path):
    source_root = tmp_path / "source"
    source = source_root / "h12" / "jaxa_knn_filled_08.h5"
    source.parent.mkdir(parents=True)
    values, observed, natural, land = _frame_inputs()
    source_code = np.zeros(values.shape, dtype=np.uint8)
    source_code[observed] = 1
    source_code[natural] = 3
    with h5py.File(source, "w") as handle:
        handle.create_dataset("sst_data", data=values[None])
        handle.create_dataset("original_obs_mask", data=observed[None].astype(np.uint8))
        handle.create_dataset("original_missing_mask", data=natural[None].astype(np.uint8))
        handle.create_dataset("source_code", data=source_code[None])
        handle.create_dataset("land_mask", data=land.astype(np.uint8))
        handle.create_dataset("timestamps", data=np.asarray([b"2024-01-01T12:00:00"]))

    output_root = tmp_path / "output"
    manifest = process_hourly_root(source_root, output_root, series_ids=(8,), workers=1)

    output = output_root / "h12" / source.name
    assert output.is_file()
    assert manifest["file_count"] == 1
    assert manifest["series_ids"] == [8]
    assert (output_root / "proxy_smoothing_manifest.json").is_file()
    with h5py.File(output, "r") as handle:
        assert int(handle.attrs["proxy_smoothing_complete"]) == 1
        assert handle["proxy_smooth_mask"].shape == handle["sst_data"].shape
