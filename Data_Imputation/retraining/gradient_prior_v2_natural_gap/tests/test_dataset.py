from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.dataset import (
    BalancedMaskTrainingDataset,
    DeploymentMatchedJAXADataset,
    high_observation_global_indices,
)


def _write_series(root: Path, series_id: int, base: float = 300.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"jaxa_knn_filled_{series_id:02d}.h5"
    sst = np.stack(
        [
            base + np.arange(30, dtype=np.float32).reshape(5, 6) * 0.01,
            base + 1 + np.arange(30, dtype=np.float32).reshape(5, 6) * 0.01,
        ]
    )
    land = np.zeros((5, 6), dtype=np.uint8)
    land[0, 0] = 1
    sst[:, 0, 0] = np.nan

    observed = np.ones((2, 5, 6), dtype=np.uint8)
    observed[:, 0, 0] = 0
    observed[1, 2, 2] = 0
    missing = np.zeros_like(observed)
    missing[1, 2, 2] = 1
    sst[1, 2, 2] = base + 8.0  # KNN proxy, never target-day model input.

    timestamps = np.asarray(
        [b"2020-01-31T12:00:00", b"2020-02-01T12:00:00"], dtype="S32"
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sst_data", data=sst)
        handle.create_dataset("original_obs_mask", data=observed)
        handle.create_dataset("original_missing_mask", data=missing)
        handle.create_dataset("land_mask", data=land)
        handle.create_dataset("timestamps", data=timestamps)
    return path


def _fixed_artificial_mask(valid_mask: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    result = np.zeros(valid_mask.shape, dtype=bool)
    assert valid_mask[1, 1]
    result[1, 1] = True
    return result


def test_target_hides_natural_and_artificial_gaps_without_proxy_leakage(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        window_size=2,
        mean=300.0,
        std=2.0,
        seed=7,
        artificial_mask_provider=_fixed_artificial_mask,
    )

    sample = dataset[1]
    natural = sample["natural_mask"]
    artificial = sample["artificial_mask"]
    reconstruction = sample["reconstruction_mask"]

    assert natural[2, 2]
    assert artificial[1, 1]
    assert not np.any(natural & artificial)
    assert np.array_equal(reconstruction, natural | artificial)
    assert np.array_equal(sample["mask_seq"][-1].astype(bool), reconstruction)
    assert np.all(sample["input_sst_seq"][-1][reconstruction] == 0.0)
    assert sample["target_proxy_sst"][2, 2] == 4.0


def test_supervision_is_strict_subset_of_finite_original_ocean_observations(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        window_size=2,
        mean=300.0,
        std=2.0,
        artificial_mask_provider=_fixed_artificial_mask,
    )

    sample = dataset[1]
    supervised = sample["supervised_mask"]
    eligible = (
        sample["original_obs_mask"]
        & ~sample["land_mask"]
        & np.isfinite(sample["target_proxy_sst"])
    )

    assert np.array_equal(supervised, sample["artificial_mask"])
    assert np.all(~supervised | eligible)
    assert not supervised[2, 2]
    assert not supervised[0, 0]


def test_previous_proxy_frame_remains_available_and_masked_as_observed_history(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        window_size=2,
        mean=300.0,
        std=2.0,
        artificial_mask_provider=_fixed_artificial_mask,
    )

    sample = dataset[1]
    assert np.isclose(
        sample["input_sst_seq"][0, 2, 2],
        np.float32((300.14 - 300.0) / 2.0),
        atol=1e-5,
    )
    assert not sample["mask_seq"][0, 2, 2]


def test_early_window_padding_never_crosses_annual_series(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0, base=300.0)
    _write_series(data_dir, 1, base=310.0)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0, 1),
        window_size=3,
        mean=300.0,
        std=2.0,
        artificial_mask_provider=_fixed_artificial_mask,
    )

    first_series_one = dataset[2]
    ocean = ~first_series_one["land_mask"]
    assert np.all(first_series_one["input_sst_seq"][0][ocean] >= 5.0)
    assert np.all(first_series_one["input_sst_seq"][1][ocean] >= 5.0)
    assert first_series_one["series_id"] == 1
    assert first_series_one["local_index"] == 0


def test_validation_masks_are_fixed_while_training_masks_are_epoch_addressable(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)

    def random_one(valid_mask, rng):
        result = np.zeros(valid_mask.shape, dtype=bool)
        candidates = np.flatnonzero(valid_mask)
        result.flat[int(rng.choice(candidates))] = True
        return result

    fixed = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        seed=11,
        vary_masks_by_epoch=False,
        artificial_mask_provider=random_one,
    )
    fixed_before = fixed[1]["artificial_mask"]
    fixed.set_epoch(4)
    assert np.array_equal(fixed_before, fixed[1]["artificial_mask"])

    varying = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        seed=11,
        vary_masks_by_epoch=True,
        artificial_mask_provider=random_one,
    )
    varying_before = varying[1]["artificial_mask"]
    varying.set_epoch(4)
    assert not np.array_equal(varying_before, varying[1]["artificial_mask"])


def test_month_and_timestamp_come_from_h5_metadata(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        artificial_mask_provider=_fixed_artificial_mask,
    )

    sample = dataset[1]
    assert sample["month"] == 2
    assert sample["timestamp"] == "2020-02-01T12:00:00"


def test_high_observation_indices_use_only_finite_original_ocean_pixels(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)

    indices = high_observation_global_indices(
        data_dir, series_ids=(0,), min_fraction=0.99
    )

    assert indices.tolist() == [0]


def test_large_masks_remain_a_subset_of_original_observations(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        active_global_indices=np.array([0]),
        mask_stream="large",
        artificial_mask_provider=_fixed_artificial_mask,
    )

    sample = dataset[0]

    assert sample["mask_stream"] == "large"
    assert np.all(~sample["artificial_mask"] | sample["original_obs_mask"])
    assert not np.any(sample["artificial_mask"] & sample["natural_mask"])


def test_balanced_training_dataset_alternates_streams_and_propagates_epoch(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    standard = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        mask_stream="standard",
        artificial_mask_provider=_fixed_artificial_mask,
    )
    large = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        active_global_indices=np.array([0]),
        mask_stream="large",
        artificial_mask_provider=_fixed_artificial_mask,
    )
    mixed = BalancedMaskTrainingDataset(standard, large)

    assert len(mixed) == 2 * max(len(standard), len(large))
    assert mixed[0]["mask_stream"] == "standard"
    assert mixed[1]["mask_stream"] == "large"
    mixed.set_epoch(3)
    assert standard.epoch == large.epoch == 3


def test_active_index_is_mapped_before_frame_lookup(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=(0,),
        active_global_indices=np.array([1]),
        artificial_mask_provider=_fixed_artificial_mask,
    )

    sample = dataset[0]

    assert sample["global_index"] == 1
    assert sample["local_index"] == 1


def test_active_indices_must_be_strictly_sorted_and_in_range(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)

    for invalid in (np.array([1, 0]), np.array([0, 0]), np.array([2])):
        try:
            DeploymentMatchedJAXADataset(
                data_dir,
                series_ids=(0,),
                active_global_indices=invalid,
                artificial_mask_provider=_fixed_artificial_mask,
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid active indices accepted: {invalid}")
