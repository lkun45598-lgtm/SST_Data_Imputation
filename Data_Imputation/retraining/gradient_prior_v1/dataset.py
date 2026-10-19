"""Deployment-matched JAXA dataset with strictly separated supervision masks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import os

import h5py
import numpy as np
from torch.utils.data import Dataset


ArtificialMaskProvider = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def build_training_cloud_bank(
    data_dir: Path | str,
    series_ids: Iterable[int],
    *,
    max_donors: int = 400,
    min_cloud_fraction: float = 0.05,
    max_cloud_fraction: float = 0.80,
    seed: int = 0,
) -> np.ndarray:
    """Reservoir-sample real ocean-cloud footprints without retaining all frames."""
    if max_donors <= 0:
        raise ValueError("max_donors must be positive")
    rng = np.random.default_rng(seed)
    reservoir: list[np.ndarray] = []
    eligible_seen = 0
    expected_shape = None
    for series_id in tuple(series_ids):
        path = Path(data_dir) / f"jaxa_knn_filled_{series_id:02d}.h5"
        with h5py.File(path, "r") as handle:
            land = np.asarray(handle["land_mask"][:], dtype=bool)
            ocean = ~land
            ocean_count = int(ocean.sum())
            expected_shape = land.shape if expected_shape is None else expected_shape
            if land.shape != expected_shape:
                raise ValueError("all cloud-bank series must share one spatial shape")
            for frame_index in range(handle["original_obs_mask"].shape[0]):
                observed = np.asarray(
                    handle["original_obs_mask"][frame_index], dtype=bool
                )
                cloud = (~observed) & ocean
                fraction = float(cloud.sum() / max(ocean_count, 1))
                if not min_cloud_fraction <= fraction <= max_cloud_fraction:
                    continue
                eligible_seen += 1
                if len(reservoir) < max_donors:
                    reservoir.append(cloud)
                else:
                    replacement = int(rng.integers(eligible_seen))
                    if replacement < max_donors:
                        reservoir[replacement] = cloud
    if not reservoir:
        raise ValueError("no eligible real-cloud masks found in training series")
    return np.stack(reservoir).astype(bool, copy=False)


class DeploymentMatchedJAXADataset(Dataset):
    """Return causal proxy history and a target with every reconstruction gap hidden."""

    def __init__(
        self,
        data_dir: Path | str,
        *,
        series_ids: Iterable[int],
        window_size: int = 30,
        mean: float = 299.9221,
        std: float = 2.6919,
        seed: int = 42,
        vary_masks_by_epoch: bool = True,
        cloud_bank: np.ndarray | None = None,
        mask_ratio_range: tuple[float, float] = (0.2, 0.3),
        artificial_mask_provider: ArtificialMaskProvider | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if std <= 0:
            raise ValueError("std must be positive")
        if artificial_mask_provider is None and cloud_bank is None:
            raise ValueError("cloud_bank is required when no artificial mask provider is supplied")

        self.data_dir = Path(data_dir)
        self.series_ids = tuple(int(value) for value in series_ids)
        self.window_size = int(window_size)
        self.mean = float(mean)
        self.std = float(std)
        self.seed = int(seed)
        self.vary_masks_by_epoch = bool(vary_masks_by_epoch)
        self.mask_ratio_range = tuple(float(value) for value in mask_ratio_range)
        self.artificial_mask_provider = artificial_mask_provider
        self.cloud_bank = None if cloud_bank is None else np.asarray(cloud_bank, dtype=bool)
        self.epoch = 0
        self._handles: dict[int, h5py.File] = {}
        self._handle_pid: int | None = None

        self.series_info: list[dict[str, object]] = []
        self.cumulative_frames = [0]
        total = 0
        expected_shape = None
        for series_id in self.series_ids:
            path = self.data_dir / f"jaxa_knn_filled_{series_id:02d}.h5"
            if not path.is_file():
                raise FileNotFoundError(f"missing JAXA series: {path}")
            with h5py.File(path, "r") as handle:
                required = {
                    "sst_data",
                    "original_obs_mask",
                    "original_missing_mask",
                    "land_mask",
                    "timestamps",
                }
                missing = required.difference(handle.keys())
                if missing:
                    raise KeyError(f"{path} is missing H5 datasets: {sorted(missing)}")
                frame_count, height, width = handle["sst_data"].shape
                shape = (height, width)
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    raise ValueError("all requested series must share one spatial shape")
                timestamps = tuple(
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in handle["timestamps"][:]
                )
                if len(timestamps) != frame_count:
                    raise ValueError(f"timestamp count mismatch in {path}")
            self.series_info.append(
                {
                    "series_id": series_id,
                    "path": path,
                    "frame_count": frame_count,
                    "shape": shape,
                    "timestamps": timestamps,
                }
            )
            total += frame_count
            self.cumulative_frames.append(total)

        if not self.series_info:
            raise ValueError("series_ids must not be empty")
        self.total_frames = total
        self.shape = expected_shape
        if self.cloud_bank is not None:
            if self.cloud_bank.ndim != 3 or tuple(self.cloud_bank.shape[1:]) != self.shape:
                raise ValueError("cloud_bank must have shape (N, H, W) matching the data")
            if self.cloud_bank.shape[0] == 0:
                raise ValueError("cloud_bank must not be empty")

    def __len__(self) -> int:
        return self.total_frames

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _locate(self, global_index: int) -> tuple[int, int]:
        if global_index < 0:
            global_index += self.total_frames
        if not 0 <= global_index < self.total_frames:
            raise IndexError(global_index)
        series_position = int(np.searchsorted(self.cumulative_frames, global_index, side="right") - 1)
        return series_position, global_index - self.cumulative_frames[series_position]

    def _close_handles(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _handle(self, series_position: int) -> h5py.File:
        current_pid = os.getpid()
        if self._handle_pid != current_pid:
            self._close_handles()
            self._handle_pid = current_pid
        if series_position not in self._handles:
            path = self.series_info[series_position]["path"]
            self._handles[series_position] = h5py.File(path, "r")
        return self._handles[series_position]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __del__(self):
        try:
            self._close_handles()
        except Exception:
            pass

    def _rng(self, series_id: int, local_index: int) -> np.random.Generator:
        epoch = self.epoch if self.vary_masks_by_epoch else 0
        seed_sequence = np.random.SeedSequence(
            [self.seed, epoch, int(series_id), int(local_index)]
        )
        return np.random.default_rng(seed_sequence)

    def _artificial_mask(
        self,
        eligible: np.ndarray,
        series_id: int,
        local_index: int,
    ) -> np.ndarray:
        rng = self._rng(series_id, local_index)
        if self.artificial_mask_provider is not None:
            raw = self.artificial_mask_provider(eligible.copy(), rng)
        else:
            from Data_Imputation.inference.real_cloud_mask import RealCloudMaskGenerator

            generated_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            generator = RealCloudMaskGenerator(
                self.cloud_bank,
                seed=generated_seed,
                ratio_range=self.mask_ratio_range,
            )
            raw = generator.generate(eligible)
        artificial = np.asarray(raw, dtype=bool)
        if artificial.shape != eligible.shape:
            raise ValueError("artificial mask provider returned the wrong shape")
        if np.any(artificial & ~eligible):
            raise ValueError("artificial mask selected a non-observed, land, or invalid pixel")
        return artificial

    def __getitem__(self, global_index: int) -> dict[str, object]:
        series_position, local_index = self._locate(global_index)
        info = self.series_info[series_position]
        series_id = int(info["series_id"])
        handle = self._handle(series_position)

        start = max(0, local_index - self.window_size + 1)
        sst_sequence = np.asarray(
            handle["sst_data"][start : local_index + 1], dtype=np.float32
        )
        mask_sequence = np.asarray(
            handle["original_missing_mask"][start : local_index + 1], dtype=bool
        )
        padding = self.window_size - sst_sequence.shape[0]
        if padding:
            sst_sequence = np.concatenate(
                [np.repeat(sst_sequence[:1], padding, axis=0), sst_sequence], axis=0
            )
            mask_sequence = np.concatenate(
                [np.repeat(mask_sequence[:1], padding, axis=0), mask_sequence], axis=0
            )

        target_proxy_kelvin = np.asarray(handle["sst_data"][local_index], dtype=np.float32)
        land = np.asarray(handle["land_mask"][:], dtype=bool)
        ocean = ~land
        original_observed = (
            np.asarray(handle["original_obs_mask"][local_index], dtype=bool)
            & ocean
            & np.isfinite(target_proxy_kelvin)
        )
        natural = (
            np.asarray(handle["original_missing_mask"][local_index], dtype=bool)
            & ocean
        )
        eligible = original_observed & ~natural
        artificial = self._artificial_mask(eligible, series_id, local_index)
        reconstruction = natural | artificial
        supervised = artificial.copy()

        if np.any(natural & artificial):
            raise RuntimeError("natural and artificial masks overlap")
        if np.any(supervised & ~original_observed):
            raise RuntimeError("supervision escaped original observations")

        target_input = sst_sequence[-1].copy()
        target_input[reconstruction] = self.mean
        sst_sequence = sst_sequence.copy()
        sst_sequence[-1] = target_input
        mask_sequence = mask_sequence.copy()
        mask_sequence[-1] = reconstruction

        normalized_input = (sst_sequence - self.mean) / self.std
        normalized_target_proxy = (target_proxy_kelvin - self.mean) / self.std
        normalized_input = np.nan_to_num(normalized_input, nan=0.0, posinf=0.0, neginf=0.0)
        normalized_target_proxy = np.nan_to_num(
            normalized_target_proxy, nan=0.0, posinf=0.0, neginf=0.0
        )

        timestamp = str(info["timestamps"][local_index])
        month = int(timestamp[5:7])
        if not 1 <= month <= 12:
            raise ValueError(f"invalid timestamp month: {timestamp}")

        return {
            "input_sst_seq": normalized_input.astype(np.float32, copy=False),
            "mask_seq": mask_sequence.astype(bool, copy=False),
            "target_sst": normalized_target_proxy.astype(np.float32, copy=False),
            "target_proxy_sst": normalized_target_proxy.astype(np.float32, copy=False),
            "natural_mask": natural,
            "artificial_mask": artificial,
            "reconstruction_mask": reconstruction,
            "supervised_mask": supervised,
            "original_obs_mask": original_observed,
            "land_mask": land,
            "month": month,
            "timestamp": timestamp,
            "series_id": series_id,
            "local_index": int(local_index),
            "global_index": int(global_index),
        }
