"""V5-labelled writer retaining the complete legacy-compatible V4 H5 schema."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from Data_Imputation.preprocessing.temporal_hourly_v4.source import SourceIndex
from Data_Imputation.preprocessing.temporal_hourly_v4.writer import (
    SeriesH5Writer as _V4SeriesH5Writer,
    reserve_output_root,
)


class SeriesH5Writer(_V4SeriesH5Writer):
    """Write V5 output using the exact V4 schema with explicit provenance versioning."""

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
    ) -> "SeriesH5Writer":
        writer = super().create(root, index, series_id, timestamps=timestamps)
        for handle in writer._handles.values():
            handle.attrs["preprocess_version"] = "temporal_hourly_v5_equivalent_acceleration"
        writer._proxy_smoothing_enabled = bool(proxy_smoothing)
        if proxy_smoothing:
            for handle in writer._handles.values():
                shape = handle["sst_data"].shape
                handle.create_dataset(
                    "proxy_smooth_mask",
                    shape=shape,
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=4,
                )
                handle.create_dataset(
                    "proxy_smooth_layer",
                    shape=shape,
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=4,
                )
                handle.attrs["proxy_smoothing_enabled"] = np.uint8(1)
                handle.attrs["proxy_smoothing_version"] = "mask_aware_normalized_gaussian_v1"
                handle.attrs["proxy_smoothing_sigma"] = float(smoothing_sigma)
                handle.attrs["proxy_smoothing_first_layer_weight"] = float(
                    smoothing_first_layer_weight
                )
                handle.attrs["proxy_smoothing_second_layer_weight"] = float(
                    smoothing_second_layer_weight
                )
                handle.attrs["proxy_smoothing_interior_weight"] = float(
                    smoothing_interior_weight
                )
                handle.attrs["proxy_smoothing_max_change_k"] = float(smoothing_max_change_k)
                handle.attrs["proxy_smoothing_anchor_mask"] = "finite original observed ocean only"
        else:
            for handle in writer._handles.values():
                handle.attrs["proxy_smoothing_enabled"] = np.uint8(0)
        return writer

    def write(
        self,
        frame,
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
        )
        if not getattr(self, "_proxy_smoothing_enabled", False):
            if proxy_smooth_mask is not None or proxy_smooth_layer is not None:
                raise ValueError("proxy smoothing audit fields require proxy_smoothing=True")
            return
        if proxy_smooth_mask is None or proxy_smooth_layer is None:
            raise ValueError("proxy smoothing audit fields are required when smoothing is enabled")
        smooth_mask = np.asarray(proxy_smooth_mask, dtype=bool)
        smooth_layer = np.asarray(proxy_smooth_layer, dtype=np.uint8)
        if smooth_mask.shape != self._land_mask.shape or smooth_layer.shape != self._land_mask.shape:
            raise ValueError("proxy smoothing audit fields must match the spatial shape")
        hour, frame_index = self._positions[frame.timestamp]
        self._handles[hour]["proxy_smooth_mask"][frame_index] = smooth_mask.astype(np.uint8)
        self._handles[hour]["proxy_smooth_layer"][frame_index] = smooth_layer
