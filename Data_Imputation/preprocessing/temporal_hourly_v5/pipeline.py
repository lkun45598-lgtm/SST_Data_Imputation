"""Chronological V5 hourly JAXA proxy generation with rolling candidate trees."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from Data_Imputation.preprocessing.temporal_hourly_v4.algorithm import (
    causal_temporal_fill,
    gaussian_temporal_fills,
)
from Data_Imputation.preprocessing.temporal_hourly_v4.pipeline import (
    FrameSummary,
    SeriesSummary,
    _proxy_history_fraction,
)
from Data_Imputation.preprocessing.temporal_hourly_v4.source import SourceIndex
from Data_Imputation.preprocessing.temporal_hourly_v4.state import KNN, TEMPORAL, ProxyFrame
from Data_Imputation.preprocessing.temporal_hourly_v6.boundary_smooth import smooth_synthetic_frame

from .algorithm import progressive_confidence_knn
from .cache import RollingCandidateCache
from .writer import SeriesH5Writer, reserve_output_root


def _rejected_proxy_count(frame: ProxyFrame, *, minimum_confidence: float = 0.20) -> int:
    """Count the same low-confidence synthetic values V4 removes before caching."""

    codes = np.asarray(frame.source_code, dtype=np.uint8)
    confidence = np.asarray(frame.confidence, dtype=np.float32)
    synthetic = (codes == TEMPORAL) | (codes == KNN)
    return int((synthetic & (confidence < float(minimum_confidence))).sum())


def run_series(
    source_root: Path,
    output_root: Path,
    *,
    series_id: int,
    max_history_hours: int = 72,
    gaussian_sigma: float = 1.5,
    query_workers: int = 21,
    stop_after: datetime | None = None,
    output_reserved: bool = False,
    smooth_synthetic: bool = False,
    smoothing_sigma: float = 0.8,
    smoothing_first_layer_weight: float = 0.25,
    smoothing_second_layer_weight: float = 0.12,
    smoothing_interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> SeriesSummary:
    """Reconstruct one source series in strict timestamp order using rolling trees."""

    if int(max_history_hours) < 1 or int(query_workers) < 1:
        raise ValueError("max_history_hours and query_workers must be positive")
    source_root = Path(source_root)
    output_root = Path(output_root)
    index = SourceIndex.from_hour_dirs(source_root, series_id=series_id)
    selected_times = tuple(
        timestamp
        for timestamp in index.available_times
        if stop_after is None or timestamp <= stop_after
    )
    if not selected_times:
        raise ValueError("The requested smoke endpoint precedes all available source timestamps")
    if not output_reserved:
        reserve_output_root(output_root)

    cache = RollingCandidateCache(max_history_hours=max_history_hours)
    rejected_low_confidence = 0
    frame_summaries: list[FrameSummary] = []
    with SeriesH5Writer.create(
        output_root,
        index,
        series_id,
        timestamps=selected_times,
        proxy_smoothing=smooth_synthetic,
        smoothing_sigma=smoothing_sigma,
        smoothing_first_layer_weight=smoothing_first_layer_weight,
        smoothing_second_layer_weight=smoothing_second_layer_weight,
        smoothing_interior_weight=smoothing_interior_weight,
        smoothing_max_change_k=max_change_k,
    ) as writer:
        for timestamp in selected_times:
            raw = index.read(timestamp)
            raw_history = cache.raw_history(timestamp)
            proxy_history = cache.proxy_history(timestamp)
            history_fraction = _proxy_history_fraction(proxy_history, raw.land_mask)
            temporal = causal_temporal_fill(raw, raw_history, max_hours=max_history_hours)
            gaussian_values = gaussian_temporal_fills(
                temporal.values,
                temporal.source_code,
                temporal.temporal_fill_mask,
                temporal.residual_missing,
                sigma=gaussian_sigma,
            )
            completed = progressive_confidence_knn(
                gaussian_values,
                temporal.source_code,
                temporal.confidence,
                raw.land_mask,
                raw_history,
                proxy_history,
                query_workers=query_workers,
                raw_tree_history=cache.raw_trees(timestamp),
                proxy_tree_history=cache.proxy_trees(timestamp),
                target_timestamp=timestamp,
            )
            if smooth_synthetic:
                natural_for_smoothing = temporal.residual_missing.copy()
                natural_for_smoothing &= ~raw.observed
                smoothed_values, smooth_mask, smooth_layer = smooth_synthetic_frame(
                    completed.values,
                    raw.observed,
                    natural_for_smoothing,
                    completed.source_code,
                    raw.land_mask,
                    sigma=smoothing_sigma,
                    first_layer_weight=smoothing_first_layer_weight,
                    second_layer_weight=smoothing_second_layer_weight,
                    interior_weight=smoothing_interior_weight,
                    max_change_k=max_change_k,
                )
            else:
                smoothed_values = completed.values
                smooth_mask = None
                smooth_layer = None
            frame = ProxyFrame(
                timestamp=timestamp,
                values=smoothed_values,
                source_code=completed.source_code,
                confidence=completed.confidence,
                temporal_fill_mask=temporal.temporal_fill_mask,
                knn_fill_mask=completed.knn_fill_mask,
                land_mask=raw.land_mask,
            )
            writer.write(
                frame,
                raw_mask=raw.observed,
                original_missing_mask=temporal.residual_missing,
                proxy_smooth_mask=smooth_mask,
                proxy_smooth_layer=smooth_layer,
            )
            rejected_low_confidence += _rejected_proxy_count(frame)
            cache.add_raw(raw)
            cache.add_proxy(frame)
            cache.prune(timestamp)
            ocean = ~raw.land_mask
            frame_summaries.append(
                FrameSummary(
                    timestamp=timestamp,
                    raw_fraction=float(raw.observed[ocean].mean()),
                    temporal_fraction=float(temporal.temporal_fill_mask[ocean].mean()),
                    knn_fraction=float(completed.knn_fill_mask[ocean].mean()),
                    proxy_history_fraction=history_fraction,
                    mean_confidence=float(completed.confidence[ocean].mean()),
                    unresolved_count=int((~np.isfinite(completed.values[ocean])).sum()),
                )
            )
    return SeriesSummary(
        series_id=int(series_id),
        processed_timestamps=selected_times,
        proxy_history_rejected_low_confidence=rejected_low_confidence,
        frames=tuple(frame_summaries),
    )


def _run_series_worker(args: tuple) -> SeriesSummary:
    (
        source_root,
        output_root,
        series_id,
        query_workers,
        smooth_synthetic,
        smoothing_sigma,
        smoothing_first_layer_weight,
        smoothing_second_layer_weight,
        smoothing_interior_weight,
        max_change_k,
    ) = args
    return run_series(
        source_root,
        output_root,
        series_id=series_id,
        query_workers=query_workers,
        output_reserved=True,
        smooth_synthetic=smooth_synthetic,
        smoothing_sigma=smoothing_sigma,
        smoothing_first_layer_weight=smoothing_first_layer_weight,
        smoothing_second_layer_weight=smoothing_second_layer_weight,
        smoothing_interior_weight=smoothing_interior_weight,
        max_change_k=max_change_k,
    )


def run_all_series(
    source_root: Path,
    output_root: Path,
    *,
    series_ids: Sequence[int] = tuple(range(9)),
    series_workers: int = 9,
    query_workers: int = 21,
    smooth_synthetic: bool = False,
    smoothing_sigma: float = 0.8,
    smoothing_first_layer_weight: float = 0.25,
    smoothing_second_layer_weight: float = 0.12,
    smoothing_interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> list[SeriesSummary]:
    """Run independent series concurrently while preserving each series' time order."""

    series_ids = tuple(int(series_id) for series_id in series_ids)
    if not series_ids:
        raise ValueError("series_ids must not be empty")
    if int(series_workers) < 1 or int(query_workers) < 1:
        raise ValueError("series_workers and query_workers must be positive")
    if int(series_workers) * int(query_workers) > 192:
        raise ValueError("series_workers * query_workers must not exceed 192")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    reserve_output_root(Path(output_root))
    worker_count = min(int(series_workers), len(series_ids))
    work = [
        (
            Path(source_root),
            Path(output_root),
            series_id,
            int(query_workers),
            bool(smooth_synthetic),
            float(smoothing_sigma),
            float(smoothing_first_layer_weight),
            float(smoothing_second_layer_weight),
            float(smoothing_interior_weight),
            float(max_change_k),
        )
        for series_id in series_ids
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_run_series_worker, work))
