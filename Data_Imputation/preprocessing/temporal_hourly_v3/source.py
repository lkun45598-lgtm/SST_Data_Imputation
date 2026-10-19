"""Timestamp-indexed access to raw observations in the hourly H5 files."""

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
    if isinstance(value, np.bytes_):
        value = value.tobytes().decode("utf-8")
    if isinstance(value, np.datetime64):
        value = str(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported timestamp value: {type(value)!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


@dataclass(frozen=True)
class SourceFrame:
    """One source frame with finite values only at original observations."""

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
    """Lazy index over h00-h23 files for one series.

    The index keeps metadata and file locations in memory, while frame arrays
    are loaded only when ``read`` is called. Historical values are always
    masked by ``original_obs_mask`` at read time.
    """

    def __init__(
        self,
        locators: Dict[datetime, SourceLocator],
        *,
        shape: Tuple[int, int],
        land_mask: np.ndarray,
        latitude: Optional[np.ndarray] = None,
        longitude: Optional[np.ndarray] = None,
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
        locators: Dict[datetime, SourceLocator] = {}
        shape: Optional[Tuple[int, int]] = None
        land_mask: Optional[np.ndarray] = None
        latitude: Optional[np.ndarray] = None
        longitude: Optional[np.ndarray] = None
        file_name = f"jaxa_knn_filled_{series_id:02d}.h5"

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
                    if "latitude" in handle:
                        latitude = np.asarray(handle["latitude"][:])
                    if "longitude" in handle:
                        longitude = np.asarray(handle["longitude"][:])
                else:
                    # The legacy hourly files contain small, frame-dependent
                    # coast/ocean differences. Use the conservative union so
                    # a pixel marked land in any source file is never treated
                    # as an ocean interpolation target.
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

        if not locators or shape is None or land_mask is None:
            raise FileNotFoundError(
                f"No hourly H5 files found for series {series_id:02d} below {root}"
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

    def read(self, timestamp: datetime) -> SourceFrame:
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
        return SourceFrame(
            timestamp=timestamp,
            values=values,
            observed=observed,
            land_mask=self.land_mask.copy(),
        )

    def history(self, target: datetime, max_hours: int) -> List[Tuple[int, SourceFrame]]:
        if max_hours < 1:
            return []
        result: List[Tuple[int, SourceFrame]] = []
        for delta in range(1, max_hours + 1):
            timestamp = target - timedelta(hours=delta)
            if self.lookup(timestamp) is not None:
                result.append((delta, self.read(timestamp)))
        return result
