"""Timestamp-indexed access to raw hourly observations stored in legacy H5 files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


def _decode_timestamp(value: object) -> datetime:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.bytes_):
        value = value.tobytes().decode("utf-8")
    elif isinstance(value, np.datetime64):
        value = str(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported timestamp value: {type(value)!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


@dataclass(frozen=True)
class RawFrame:
    """One source frame containing finite SST only at original observations."""

    timestamp: datetime
    values: np.ndarray
    observed: np.ndarray
    land_mask: np.ndarray


@dataclass(frozen=True)
class SourceLocator:
    path: Path
    frame_index: int
    timestamp: datetime


class SourceIndex:
    """Lazy index across the present h00--h23 files for one series."""

    def __init__(
        self,
        locators: Dict[datetime, SourceLocator],
        *,
        shape: Tuple[int, int],
        land_mask: np.ndarray,
        latitude: Optional[np.ndarray],
        longitude: Optional[np.ndarray],
    ) -> None:
        self._locators = dict(locators)
        self._available_times = tuple(sorted(self._locators))
        self.shape = tuple(shape)
        self.land_mask = np.asarray(land_mask, dtype=bool).copy()
        self.latitude = None if latitude is None else np.asarray(latitude).copy()
        self.longitude = None if longitude is None else np.asarray(longitude).copy()

    @classmethod
    def from_hour_dirs(cls, root: Path, series_id: int) -> "SourceIndex":
        root = Path(root)
        file_name = f"jaxa_knn_filled_{int(series_id):02d}.h5"
        locators: Dict[datetime, SourceLocator] = {}
        shape: Optional[Tuple[int, int]] = None
        land_mask: Optional[np.ndarray] = None
        latitude: Optional[np.ndarray] = None
        longitude: Optional[np.ndarray] = None

        for hour in range(24):
            path = root / f"h{hour:02d}" / file_name
            if not path.exists():
                continue
            with h5py.File(path, "r") as handle:
                required = {"sst_data", "original_obs_mask", "land_mask", "timestamps"}
                missing = required.difference(handle.keys())
                if missing:
                    raise ValueError(f"{path} is missing datasets: {sorted(missing)}")
                sst = handle["sst_data"]
                observed = handle["original_obs_mask"]
                current_shape = tuple(sst.shape[1:])
                if len(current_shape) != 2 or observed.shape != sst.shape:
                    raise ValueError(f"Invalid frame/mask shape in {path}")
                if shape is None:
                    shape = current_shape
                elif current_shape != shape:
                    raise ValueError(f"Spatial shape mismatch in {path}: {current_shape} != {shape}")

                current_land = np.asarray(handle["land_mask"][:], dtype=bool)
                if current_land.shape != current_shape:
                    raise ValueError(f"Land-mask shape mismatch in {path}")
                if land_mask is None:
                    land_mask = current_land
                    latitude = np.asarray(handle["latitude"][:]) if "latitude" in handle else None
                    longitude = np.asarray(handle["longitude"][:]) if "longitude" in handle else None
                else:
                    land_mask |= current_land

                timestamps = handle["timestamps"][:]
                if len(timestamps) != sst.shape[0]:
                    raise ValueError(f"Timestamp/frame count mismatch in {path}")
                for frame_index, raw_timestamp in enumerate(timestamps):
                    timestamp = _decode_timestamp(raw_timestamp)
                    if timestamp in locators:
                        previous = locators[timestamp].path
                        raise ValueError(
                            f"Duplicate timestamp {timestamp.isoformat()} in {previous} and {path}"
                        )
                    locators[timestamp] = SourceLocator(path, frame_index, timestamp)

        if shape is None or land_mask is None or not locators:
            raise FileNotFoundError(
                f"No hourly H5 files found for series {int(series_id):02d} below {root}"
            )
        return cls(
            locators,
            shape=shape,
            land_mask=land_mask,
            latitude=latitude,
            longitude=longitude,
        )

    @property
    def available_times(self) -> Tuple[datetime, ...]:
        return self._available_times

    def lookup(self, timestamp: datetime) -> Optional[SourceLocator]:
        return self._locators.get(timestamp)

    def read(self, timestamp: datetime) -> RawFrame:
        locator = self.lookup(timestamp)
        if locator is None:
            raise KeyError(f"Timestamp is not available: {timestamp.isoformat()}")
        with h5py.File(locator.path, "r") as handle:
            values = np.asarray(handle["sst_data"][locator.frame_index], dtype=np.float32)
            observed = np.asarray(
                handle["original_obs_mask"][locator.frame_index], dtype=bool
            )
        if values.shape != self.shape or observed.shape != self.shape:
            raise ValueError(f"Frame shape mismatch at {timestamp.isoformat()}")
        observed &= ~self.land_mask
        values = np.where(observed & np.isfinite(values), values, np.nan).astype(np.float32)
        return RawFrame(
            timestamp=timestamp,
            values=values,
            observed=observed,
            land_mask=self.land_mask.copy(),
        )

    def raw_history(self, target: datetime, max_hours: int) -> List[Tuple[int, RawFrame]]:
        if max_hours < 1:
            return []
        history: List[Tuple[int, RawFrame]] = []
        for delta in range(1, int(max_hours) + 1):
            timestamp = target - timedelta(hours=delta)
            if self.lookup(timestamp) is not None:
                history.append((delta, self.read(timestamp)))
        return history
