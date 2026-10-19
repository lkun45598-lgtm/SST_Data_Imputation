"""Streaming, legacy-compatible H5 output for the v4 proxy pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Dict, Iterable

import h5py
import numpy as np

from .source import SourceIndex
from .state import ProxyFrame


_COMPRESSION = "gzip"
_COMPRESSION_OPTS = 4


def reserve_output_root(root: Path) -> None:
    """Create the final h00--h23 layout only when it cannot overwrite data."""

    root = Path(root)
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError(f"Refusing to write into non-empty output root: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)
    for hour in range(24):
        (root / f"h{hour:02d}").mkdir(exist_ok=False)


def _timestamps_by_hour(timestamps: Iterable[datetime]) -> Dict[int, tuple[datetime, ...]]:
    grouped: Dict[int, list[datetime]] = {}
    for timestamp in timestamps:
        grouped.setdefault(int(timestamp.hour), []).append(timestamp)
    return {hour: tuple(sorted(values)) for hour, values in grouped.items()}


class SeriesH5Writer:
    """Preallocated per-hour H5 handles for one chronological series run."""

    def __init__(
        self,
        *,
        handles: Dict[int, h5py.File],
        positions: Dict[datetime, tuple[int, int]],
        written: Dict[int, np.ndarray],
        land_mask: np.ndarray,
    ) -> None:
        self._handles = handles
        self._positions = positions
        self._written = written
        self._land_mask = np.asarray(land_mask, dtype=bool).copy()
        self._closed = False

    @classmethod
    def create(
        cls,
        root: Path,
        index: SourceIndex,
        series_id: int,
        *,
        timestamps: Iterable[datetime] | None = None,
    ) -> "SeriesH5Writer":
        """Open fresh fixed-size output files for every hour present in ``index``."""

        root = Path(root)
        selected = tuple(index.available_times if timestamps is None else sorted(timestamps))
        unknown = set(selected).difference(index.available_times)
        if unknown:
            first_unknown = min(unknown)
            raise KeyError(f"Timestamp is not present in the source index: {first_unknown.isoformat()}")
        grouped = _timestamps_by_hour(selected)
        handles: Dict[int, h5py.File] = {}
        positions: Dict[datetime, tuple[int, int]] = {}
        written: Dict[int, np.ndarray] = {}
        try:
            for hour, timestamps in grouped.items():
                output_path = root / f"h{hour:02d}" / f"jaxa_knn_filled_{int(series_id):02d}.h5"
                if output_path.exists():
                    raise FileExistsError(f"Output already exists: {output_path}")
                handle = h5py.File(output_path, "x")
                handles[hour] = handle
                frame_shape = (len(timestamps), *index.shape)
                handle.create_dataset(
                    "sst_data",
                    shape=frame_shape,
                    dtype=np.float32,
                    compression=_COMPRESSION,
                    compression_opts=_COMPRESSION_OPTS,
                )
                handle.create_dataset(
                    "land_mask",
                    data=index.land_mask.astype(np.uint8),
                    compression=_COMPRESSION,
                    compression_opts=_COMPRESSION_OPTS,
                )
                if index.latitude is not None:
                    handle.create_dataset("latitude", data=index.latitude)
                if index.longitude is not None:
                    handle.create_dataset("longitude", data=index.longitude)
                for dataset_name in (
                    "original_obs_mask",
                    "temporal_fill_mask",
                    "original_missing_mask",
                    "knn_fill_mask",
                    "unresolved_mask",
                    "source_code",
                ):
                    handle.create_dataset(
                        dataset_name,
                        shape=frame_shape,
                        dtype=np.uint8,
                        compression=_COMPRESSION,
                        compression_opts=_COMPRESSION_OPTS,
                    )
                handle.create_dataset(
                    "proxy_confidence",
                    shape=frame_shape,
                    dtype=np.float16,
                    compression=_COMPRESSION,
                    compression_opts=_COMPRESSION_OPTS,
                )
                handle.create_dataset(
                    "timestamps",
                    data=np.asarray(
                        [timestamp.isoformat().encode("ascii") for timestamp in timestamps],
                        dtype="S32",
                    ),
                )
                handle.attrs["series_id"] = int(series_id)
                handle.attrs["hour_offset"] = int(hour)
                handle.attrs["num_frames"] = int(len(timestamps))
                handle.attrs["creation_date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                handle.attrs["preprocess_version"] = "temporal_hourly_v4_confidence"
                handle.attrs["max_history_hours"] = 72
                handle.attrs["gaussian_sigma"] = 1.5
                handle.attrs["spatial_weight_power"] = 2.0
                handle.attrs["proxy_blend_maximum"] = 0.65
                handle.attrs["proxy_minimum_confidence"] = 0.20
                handle.attrs["proxy_policy"] = (
                    "synthetic-only; confidence>=0.20; nonzero spatial distance"
                )
                written[hour] = np.zeros(len(timestamps), dtype=bool)
                for frame_index, timestamp in enumerate(timestamps):
                    positions[timestamp] = (hour, frame_index)
        except Exception:
            for handle in handles.values():
                handle.close()
            raise
        return cls(
            handles=handles,
            positions=positions,
            written=written,
            land_mask=index.land_mask,
        )

    def __enter__(self) -> "SeriesH5Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def write(
        self,
        frame: ProxyFrame,
        *,
        raw_mask: np.ndarray,
        original_missing_mask: np.ndarray,
    ) -> None:
        """Write one completed timestamp once across all legacy and v4 datasets."""

        if self._closed:
            raise RuntimeError("Cannot write through a closed SeriesH5Writer")
        position = self._positions.get(frame.timestamp)
        if position is None:
            raise KeyError(f"Timestamp was not preallocated: {frame.timestamp.isoformat()}")
        hour, frame_index = position
        if self._written[hour][frame_index]:
            raise RuntimeError(f"Timestamp was already written: {frame.timestamp.isoformat()}")
        values = np.asarray(frame.values, dtype=np.float32)
        codes = np.asarray(frame.source_code, dtype=np.uint8)
        confidence = np.asarray(frame.confidence, dtype=np.float32)
        temporal = np.asarray(frame.temporal_fill_mask, dtype=bool)
        knn = np.asarray(frame.knn_fill_mask, dtype=bool)
        raw = np.asarray(raw_mask, dtype=bool)
        original_missing = np.asarray(original_missing_mask, dtype=bool)
        expected_shape = self._land_mask.shape
        fields = (values, codes, confidence, temporal, knn, raw, original_missing)
        if any(field.shape != expected_shape for field in fields):
            raise ValueError("All output fields must match the source spatial shape")
        frame_land = np.asarray(frame.land_mask, dtype=bool)
        if frame_land.shape != expected_shape or not np.array_equal(frame_land, self._land_mask):
            raise ValueError("Proxy-frame land mask must equal the source land-mask union")

        unresolved = ~self._land_mask & ~np.isfinite(values)
        handle = self._handles[hour]
        handle["sst_data"][frame_index] = values
        handle["original_obs_mask"][frame_index] = raw.astype(np.uint8)
        handle["temporal_fill_mask"][frame_index] = temporal.astype(np.uint8)
        handle["original_missing_mask"][frame_index] = original_missing.astype(np.uint8)
        handle["knn_fill_mask"][frame_index] = knn.astype(np.uint8)
        handle["unresolved_mask"][frame_index] = unresolved.astype(np.uint8)
        handle["source_code"][frame_index] = codes
        handle["proxy_confidence"][frame_index] = confidence.astype(np.float16)
        self._written[hour][frame_index] = True

    def close(self) -> None:
        """Flush and close handles after enforcing complete, finite ocean output."""

        if self._closed:
            return
        errors: list[str] = []
        for hour, handle in self._handles.items():
            unwritten = int((~self._written[hour]).sum())
            if unwritten:
                errors.append(f"h{hour:02d} has {unwritten} unwritten frames")
            unresolved_data = np.asarray(handle["unresolved_mask"][:], dtype=np.uint8)
            unresolved = int(unresolved_data[:, ~self._land_mask].sum())
            if unresolved:
                errors.append(f"h{hour:02d} has {unresolved} unresolved ocean pixels")
            handle.flush()
            handle.close()
        self._closed = True
        if errors:
            raise RuntimeError("; ".join(errors))
