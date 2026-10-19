"""Isolated H5 pipeline for the causal hourly v3 preprocessing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

from .algorithm import (
    causal_temporal_fill,
    gaussian_temporal_fills,
    progressive_spatiotemporal_knn,
)
from .source import SourceIndex


def _selected_positions(
    count: int,
    *,
    frame_limit: int | None,
    selected_indices: Sequence[int] | None,
) -> list[int]:
    if selected_indices is not None:
        positions = sorted(set(int(index) for index in selected_indices))
        invalid = [index for index in positions if index < 0 or index >= count]
        if invalid:
            raise IndexError(f"Target frame indices out of range: {invalid}")
        return positions
    if frame_limit is None:
        return list(range(count))
    if frame_limit < 1:
        raise ValueError("frame_limit must be positive")
    return list(range(min(count, int(frame_limit))))


def _process_frame(index: SourceIndex, timestamp: datetime, *, max_history_hours: int, gaussian_sigma: float) -> dict:
    target = index.read(timestamp)
    history = index.history(timestamp, max_history_hours)
    temporal = causal_temporal_fill(target, history, max_hours=max_history_hours)
    smoothed = gaussian_temporal_fills(
        temporal.values,
        temporal.residual_missing,
        temporal.temporal_fill_mask,
        sigma=gaussian_sigma,
    )
    ocean = ~index.land_mask
    residual_before_knn = ocean & ~np.isfinite(smoothed)
    filled, knn_mask = progressive_spatiotemporal_knn(
        smoothed,
        target.observed,
        target.land_mask,
        history,
        target_timestamp=timestamp,
        temporal_fill_mask=temporal.temporal_fill_mask,
    )
    unresolved = ocean & ~np.isfinite(filled)
    if np.any(unresolved):
        raise RuntimeError(
            f"{timestamp.isoformat()}: refusing to save {int(unresolved.sum())} unresolved ocean pixels"
        )
    filled[target.land_mask] = np.nan
    return {
        "timestamp": timestamp.isoformat().encode("ascii"),
        "sst_data": filled.astype(np.float32),
        "original_obs_mask": target.observed.astype(np.uint8),
        "temporal_fill_mask": temporal.temporal_fill_mask.astype(np.uint8),
        "original_missing_mask": residual_before_knn.astype(np.uint8),
        "knn_fill_mask": knn_mask.astype(np.uint8),
        "unresolved_mask": unresolved.astype(np.uint8),
    }


def _process_frame_worker(args: tuple[Path, int, datetime, int, float]) -> dict:
    source_root, series_id, timestamp, max_history_hours, gaussian_sigma = args
    index = SourceIndex.from_hour_dirs(source_root, series_id=series_id)
    return _process_frame(
        index,
        timestamp,
        max_history_hours=max_history_hours,
        gaussian_sigma=gaussian_sigma,
    )


def run_series(
    source_root: Path,
    output_root: Path,
    *,
    series_id: int,
    target_hour: int = 12,
    max_history_hours: int = 72,
    gaussian_sigma: float = 1.5,
    frame_limit: int | None = None,
    selected_indices: Sequence[int] | None = None,
    workers: int = 1,
) -> Path:
    """Process one target-hour series and write a new compatible H5 file."""

    if not 0 <= int(target_hour) <= 23:
        raise ValueError("target_hour must be between 0 and 23")
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    index = SourceIndex.from_hour_dirs(source_root, series_id=series_id)
    target_times = tuple(
        timestamp for timestamp in index.available_times if timestamp.hour == int(target_hour)
    )
    if not target_times:
        raise FileNotFoundError(
            f"No timestamps for target hour {int(target_hour):02d} and series {series_id:02d}"
        )
    positions = _selected_positions(
        len(target_times), frame_limit=frame_limit, selected_indices=selected_indices
    )

    final_frames: list[np.ndarray] = []
    raw_masks: list[np.ndarray] = []
    temporal_masks: list[np.ndarray] = []
    original_missing_masks: list[np.ndarray] = []
    knn_masks: list[np.ndarray] = []
    unresolved_masks: list[np.ndarray] = []
    output_timestamps: list[bytes] = []
    ocean = ~index.land_mask

    selected_timestamps = [target_times[position] for position in positions]
    if int(workers) > 1 and len(selected_timestamps) > 1:
        worker_args = [
            (source_root, int(series_id), timestamp, int(max_history_hours), float(gaussian_sigma))
            for timestamp in selected_timestamps
        ]
        with ProcessPoolExecutor(max_workers=min(int(workers), len(worker_args))) as executor:
            processed = list(executor.map(_process_frame_worker, worker_args))
    else:
        processed = [
            _process_frame(
                index,
                timestamp,
                max_history_hours=max_history_hours,
                gaussian_sigma=gaussian_sigma,
            )
            for timestamp in selected_timestamps
        ]

    for result in processed:
        final_frames.append(result["sst_data"])
        raw_masks.append(result["original_obs_mask"])
        temporal_masks.append(result["temporal_fill_mask"])
        original_missing_masks.append(result["original_missing_mask"])
        knn_masks.append(result["knn_fill_mask"])
        unresolved_masks.append(result["unresolved_mask"])
        output_timestamps.append(result["timestamp"])

    output_dir = output_root / f"h{int(target_hour):02d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / f"jaxa_knn_filled_{int(series_id):02d}.h5"
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("sst_data", data=np.stack(final_frames), compression="gzip", compression_opts=4)
        handle.create_dataset("land_mask", data=index.land_mask.astype(np.uint8), compression="gzip", compression_opts=4)
        if index.latitude is not None:
            handle.create_dataset("latitude", data=index.latitude)
        if index.longitude is not None:
            handle.create_dataset("longitude", data=index.longitude)
        handle.create_dataset("original_obs_mask", data=np.stack(raw_masks), compression="gzip", compression_opts=4)
        handle.create_dataset("temporal_fill_mask", data=np.stack(temporal_masks), compression="gzip", compression_opts=4)
        handle.create_dataset("original_missing_mask", data=np.stack(original_missing_masks), compression="gzip", compression_opts=4)
        handle.create_dataset("knn_fill_mask", data=np.stack(knn_masks), compression="gzip", compression_opts=4)
        handle.create_dataset("unresolved_mask", data=np.stack(unresolved_masks), compression="gzip", compression_opts=4)
        handle.create_dataset("timestamps", data=np.asarray(output_timestamps, dtype="S32"))
        handle.attrs["preprocess_version"] = "v3_temporal_hourly"
        handle.attrs["series_id"] = int(series_id)
        handle.attrs["target_hour"] = int(target_hour)
        handle.attrs["max_history_hours"] = int(max_history_hours)
        handle.attrs["gaussian_sigma"] = float(gaussian_sigma)
        handle.attrs["knn_k"] = 20
        handle.attrs["knn_source_k"] = 8
        handle.attrs["knn_power_space"] = 2.0
        handle.attrs["knn_power_time"] = 1.0
        handle.attrs["source_policy"] = "original_obs_mask_only"
        handle.attrs["frame_selection"] = "full_series" if selected_indices is None and frame_limit is None else "selected"

    return output_path
