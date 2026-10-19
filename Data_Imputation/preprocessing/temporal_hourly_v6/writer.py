"""Strict-prefix resume support for V5-compatible hourly H5 output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from Data_Imputation.preprocessing.temporal_hourly_v4.source import SourceIndex
from Data_Imputation.preprocessing.temporal_hourly_v4.state import (
    KNN,
    LAND,
    RAW,
    TEMPORAL,
    ProxyFrame,
)
from Data_Imputation.preprocessing.temporal_hourly_v4.writer import reserve_output_root
from Data_Imputation.preprocessing.temporal_hourly_v5.writer import SeriesH5Writer


_VERSION = "temporal_hourly_v6_parallel_resume"
_FRAME_FIELDS = {
    "sst_data",
    "original_obs_mask",
    "temporal_fill_mask",
    "original_missing_mask",
    "knn_fill_mask",
    "unresolved_mask",
    "source_code",
    "proxy_confidence",
    "resume_confidence_f32",
}


def _decode_timestamp(value: object) -> datetime:
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("ascii")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported timestamp value: {type(value)!r}")
    return datetime.fromisoformat(value.strip().removesuffix("Z")).replace(tzinfo=None)


def _timestamps_by_hour(timestamps: Iterable[datetime]) -> dict[int, tuple[datetime, ...]]:
    grouped: dict[int, list[datetime]] = {}
    for timestamp in timestamps:
        grouped.setdefault(int(timestamp.hour), []).append(timestamp)
    return {hour: tuple(sorted(items)) for hour, items in grouped.items()}


def _completion_flags(handle: h5py.File, land_mask: np.ndarray) -> np.ndarray:
    """Conservatively classify frames whose complete provenance was committed."""

    values = np.asarray(handle["sst_data"][:], dtype=np.float32)
    raw = np.asarray(handle["original_obs_mask"][:], dtype=bool)
    temporal = np.asarray(handle["temporal_fill_mask"][:], dtype=bool)
    original_missing = np.asarray(handle["original_missing_mask"][:], dtype=bool)
    knn = np.asarray(handle["knn_fill_mask"][:], dtype=bool)
    unresolved = np.asarray(handle["unresolved_mask"][:], dtype=bool)
    codes = np.asarray(handle["source_code"][:], dtype=np.uint8)
    confidence = np.asarray(handle["resume_confidence_f32"][:], dtype=np.float32)
    frame_count = values.shape[0]
    expected_shape = (frame_count, *land_mask.shape)
    fields = (raw, temporal, original_missing, knn, unresolved, codes, confidence)
    if values.shape != expected_shape or any(field.shape != expected_shape for field in fields):
        raise ValueError(f"Frame dataset shape mismatch in {handle.filename}")

    ocean = ~np.asarray(land_mask, dtype=bool).ravel()
    land = ~ocean
    flat_values = values.reshape(frame_count, -1)
    flat_raw = raw.reshape(frame_count, -1)
    flat_temporal = temporal.reshape(frame_count, -1)
    flat_original_missing = original_missing.reshape(frame_count, -1)
    flat_knn = knn.reshape(frame_count, -1)
    flat_unresolved = unresolved.reshape(frame_count, -1)
    flat_codes = codes.reshape(frame_count, -1)
    flat_confidence = confidence.reshape(frame_count, -1)

    complete = np.isfinite(flat_values[:, ocean]).all(axis=1)
    complete &= np.isfinite(flat_confidence[:, ocean]).all(axis=1)
    complete &= (flat_confidence[:, ocean] > 0.0).all(axis=1)
    complete &= np.isin(flat_codes[:, ocean], (RAW, TEMPORAL, KNN)).all(axis=1)
    complete &= ~flat_unresolved[:, ocean].any(axis=1)
    complete &= (flat_raw[:, ocean] == (flat_codes[:, ocean] == RAW)).all(axis=1)
    complete &= (flat_temporal[:, ocean] == (flat_codes[:, ocean] == TEMPORAL)).all(axis=1)
    complete &= (flat_knn[:, ocean] == (flat_codes[:, ocean] == KNN)).all(axis=1)
    complete &= (flat_original_missing[:, ocean] == flat_knn[:, ocean]).all(axis=1)
    raw_ocean = flat_codes[:, ocean] == RAW
    complete &= (~raw_ocean | (flat_confidence[:, ocean] == 1.0)).all(axis=1)
    if np.any(land):
        complete &= np.isnan(flat_values[:, land]).all(axis=1)
        complete &= (flat_codes[:, land] == LAND).all(axis=1)
        complete &= ~flat_raw[:, land].any(axis=1)
        complete &= ~flat_temporal[:, land].any(axis=1)
        complete &= ~flat_knn[:, land].any(axis=1)
        complete &= ~flat_unresolved[:, land].any(axis=1)
    return complete


class ResumableSeriesH5Writer(SeriesH5Writer):
    """Create fresh V6 files or resume one strictly chronological V5/V6 prefix."""

    @classmethod
    def create(
        cls,
        root: Path,
        index: SourceIndex,
        series_id: int,
        *,
        timestamps: Iterable[datetime] | None = None,
        proxy_smoothing: bool = False,
        smoothing_sigma: float = 0.8,
        smoothing_first_layer_weight: float = 0.25,
        smoothing_second_layer_weight: float = 0.12,
        smoothing_interior_weight: float = 0.08,
        smoothing_max_change_k: float = 0.10,
    ) -> "ResumableSeriesH5Writer":
        writer = super().create(
            root,
            index,
            series_id,
            timestamps=timestamps,
            proxy_smoothing=proxy_smoothing,
            smoothing_sigma=smoothing_sigma,
            smoothing_first_layer_weight=smoothing_first_layer_weight,
            smoothing_second_layer_weight=smoothing_second_layer_weight,
            smoothing_interior_weight=smoothing_interior_weight,
            smoothing_max_change_k=smoothing_max_change_k,
        )
        for handle in writer._handles.values():
            handle.create_dataset(
                "resume_confidence_f32",
                shape=handle["proxy_confidence"].shape,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            handle.attrs["preprocess_version"] = _VERSION
        return writer

    @classmethod
    def resume(
        cls,
        root: Path,
        index: SourceIndex,
        series_id: int,
        *,
        timestamps: Iterable[datetime] | None = None,
        proxy_smoothing: bool = False,
        smoothing_sigma: float = 0.8,
        smoothing_first_layer_weight: float = 0.25,
        smoothing_second_layer_weight: float = 0.12,
        smoothing_interior_weight: float = 0.08,
        smoothing_max_change_k: float = 0.10,
    ) -> "ResumableSeriesH5Writer":
        root = Path(root)
        selected = tuple(index.available_times if timestamps is None else sorted(timestamps))
        unknown = set(selected).difference(index.available_times)
        if unknown:
            raise KeyError(f"Timestamp is not present in the source index: {min(unknown).isoformat()}")
        grouped = _timestamps_by_hour(selected)
        handles: dict[int, h5py.File] = {}
        positions: dict[datetime, tuple[int, int]] = {}
        written: dict[int, np.ndarray] = {}
        try:
            for hour, hour_times in grouped.items():
                path = root / f"h{hour:02d}" / f"jaxa_knn_filled_{int(series_id):02d}.h5"
                if not path.is_file():
                    raise FileNotFoundError(f"Resume output is missing: {path}")
                handle = h5py.File(path, "r+")
                handles[hour] = handle
                missing_fields = _FRAME_FIELDS.difference(handle.keys())
                if missing_fields:
                    raise ValueError(f"{path} is missing resume fields: {sorted(missing_fields)}")
                enabled = bool(int(handle.attrs.get("proxy_smoothing_enabled", 0)))
                if enabled != bool(proxy_smoothing):
                    raise ValueError(
                        f"Resume smoothing mode differs from output metadata in {path}: "
                        f"file={enabled}, requested={bool(proxy_smoothing)}"
                    )
                if proxy_smoothing:
                    for field in ("proxy_smooth_mask", "proxy_smooth_layer"):
                        if field not in handle or handle[field].shape != handle["sst_data"].shape:
                            raise ValueError(f"{path} is missing valid {field} audit data")
                    expected_attrs = {
                        "proxy_smoothing_sigma": float(smoothing_sigma),
                        "proxy_smoothing_first_layer_weight": float(smoothing_first_layer_weight),
                        "proxy_smoothing_second_layer_weight": float(smoothing_second_layer_weight),
                        "proxy_smoothing_interior_weight": float(smoothing_interior_weight),
                        "proxy_smoothing_max_change_k": float(smoothing_max_change_k),
                    }
                    for attr_name, expected_value in expected_attrs.items():
                        actual_value = float(handle.attrs.get(attr_name, np.nan))
                        if not np.isclose(actual_value, expected_value, rtol=0.0, atol=1e-12):
                            raise ValueError(
                                f"Resume smoothing parameter {attr_name} differs in {path}: "
                                f"file={actual_value}, requested={expected_value}"
                            )
                if "timestamps" not in handle or "land_mask" not in handle:
                    raise ValueError(f"{path} is missing timestamps or land_mask")
                actual_times = tuple(_decode_timestamp(value) for value in handle["timestamps"][:])
                if actual_times != hour_times:
                    raise ValueError(f"Resume timestamps differ from source index in {path}")
                current_land = np.asarray(handle["land_mask"][:], dtype=bool)
                if not np.array_equal(current_land, index.land_mask):
                    raise ValueError(f"Resume land mask differs from source index in {path}")
                written[hour] = _completion_flags(handle, index.land_mask)
                handle.attrs["preprocess_version"] = _VERSION
                for frame_index, timestamp in enumerate(hour_times):
                    positions[timestamp] = (hour, frame_index)

            ordered_flags = [written[hour][frame_index] for hour, frame_index in map(positions.get, selected)]
            first_incomplete = next(
                (index for index, is_complete in enumerate(ordered_flags) if not is_complete),
                len(selected),
            )
            if any(ordered_flags[first_incomplete:]):
                first_late = selected[
                    first_incomplete
                    + next(index for index, flag in enumerate(ordered_flags[first_incomplete:]) if flag)
                ]
                raise RuntimeError(
                    "Resume output is not a strict chronological prefix; "
                    f"found a completed frame after a gap at {first_late.isoformat()}"
                )
        except Exception:
            for handle in handles.values():
                handle.close()
            raise
        writer = cls(
            handles=handles,
            positions=positions,
            written=written,
            land_mask=index.land_mask,
        )
        writer._proxy_smoothing_enabled = bool(proxy_smoothing)
        return writer

    @property
    def completed_timestamps(self) -> tuple[datetime, ...]:
        return tuple(
            timestamp
            for timestamp in sorted(self._positions)
            if self._written[self._positions[timestamp][0]][self._positions[timestamp][1]]
        )

    def load_proxy_frame(self, timestamp: datetime) -> ProxyFrame:
        position = self._positions.get(timestamp)
        if position is None:
            raise KeyError(f"Timestamp was not preallocated: {timestamp.isoformat()}")
        hour, frame_index = position
        if not self._written[hour][frame_index]:
            raise RuntimeError(f"Cannot load incomplete proxy frame: {timestamp.isoformat()}")
        handle = self._handles[hour]
        return ProxyFrame(
            timestamp=timestamp,
            values=np.asarray(handle["sst_data"][frame_index], dtype=np.float32),
            source_code=np.asarray(handle["source_code"][frame_index], dtype=np.uint8),
            confidence=np.asarray(handle["resume_confidence_f32"][frame_index], dtype=np.float32),
            temporal_fill_mask=np.asarray(handle["temporal_fill_mask"][frame_index], dtype=bool),
            knn_fill_mask=np.asarray(handle["knn_fill_mask"][frame_index], dtype=bool),
            land_mask=self._land_mask.copy(),
        )

    def write(
        self,
        frame: ProxyFrame,
        *,
        raw_mask: np.ndarray,
        original_missing_mask: np.ndarray,
        proxy_smooth_mask: np.ndarray | None = None,
        proxy_smooth_layer: np.ndarray | None = None,
    ) -> None:
        super().write(
            frame,
            raw_mask=raw_mask,
            original_missing_mask=original_missing_mask,
            proxy_smooth_mask=proxy_smooth_mask,
            proxy_smooth_layer=proxy_smooth_layer,
        )
        hour, frame_index = self._positions[frame.timestamp]
        self._handles[hour]["resume_confidence_f32"][frame_index] = np.asarray(
            frame.confidence,
            dtype=np.float32,
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(require_complete=exc_type is None)

    def close(self, *, require_complete: bool = True) -> None:
        if self._closed:
            return
        if require_complete:
            super().close()
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self._closed = True


__all__ = ["ResumableSeriesH5Writer", "reserve_output_root"]
