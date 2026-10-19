"""Chronological V6 processing with persistent per-series query executors."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
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
from Data_Imputation.preprocessing.temporal_hourly_v5.cache import RollingCandidateCache
from Data_Imputation.preprocessing.temporal_hourly_v6.boundary_smooth import smooth_synthetic_frame

from .algorithm import progressive_confidence_knn
from .writer import ResumableSeriesH5Writer, reserve_output_root


def _rejected_proxy_count(frame: ProxyFrame, *, minimum_confidence: float = 0.20) -> int:
    codes = np.asarray(frame.source_code, dtype=np.uint8)
    confidence = np.asarray(frame.confidence, dtype=np.float32)
    synthetic = (codes == TEMPORAL) | (codes == KNN)
    return int((synthetic & (confidence < float(minimum_confidence))).sum())


def _restore_cache(
    cache: RollingCandidateCache,
    writer: ResumableSeriesH5Writer,
    index: SourceIndex,
    completed_times: Sequence[datetime],
    next_timestamp: datetime,
    *,
    max_history_hours: int,
) -> None:
    cutoff = next_timestamp - timedelta(hours=int(max_history_hours))
    for timestamp in completed_times:
        if cutoff <= timestamp < next_timestamp:
            cache.add_raw(index.read(timestamp))
            cache.add_proxy(writer.load_proxy_frame(timestamp))
    cache.prune(next_timestamp)


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
    resume: bool = False,
    preallocate_all: bool = False,
    smooth_synthetic: bool = False,
    smoothing_sigma: float = 0.8,
    smoothing_first_layer_weight: float = 0.25,
    smoothing_second_layer_weight: float = 0.12,
    smoothing_interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> SeriesSummary:
    """Run or resume one series while preserving strict causal time order."""

    if int(max_history_hours) < 1 or int(query_workers) < 1:
        raise ValueError("max_history_hours and query_workers must be positive")
    if resume and preallocate_all:
        raise ValueError("preallocate_all is only valid when creating a fresh output")
    source_root = Path(source_root)
    output_root = Path(output_root)
    index = SourceIndex.from_hour_dirs(source_root, series_id=series_id)
    all_times = tuple(index.available_times)
    process_times = tuple(
        timestamp for timestamp in all_times if stop_after is None or timestamp <= stop_after
    )
    if not process_times:
        raise ValueError("The requested endpoint precedes all available source timestamps")
    allocation_times = all_times if preallocate_all else process_times

    if resume:
        writer = ResumableSeriesH5Writer.resume(
            output_root,
            index,
            series_id,
            timestamps=allocation_times,
            proxy_smoothing=smooth_synthetic,
            smoothing_sigma=smoothing_sigma,
            smoothing_first_layer_weight=smoothing_first_layer_weight,
            smoothing_second_layer_weight=smoothing_second_layer_weight,
            smoothing_interior_weight=smoothing_interior_weight,
            smoothing_max_change_k=max_change_k,
        )
    else:
        if not output_reserved:
            reserve_output_root(output_root)
        writer = ResumableSeriesH5Writer.create(
            output_root,
            index,
            series_id,
            timestamps=allocation_times,
            proxy_smoothing=smooth_synthetic,
            smoothing_sigma=smoothing_sigma,
            smoothing_first_layer_weight=smoothing_first_layer_weight,
            smoothing_second_layer_weight=smoothing_second_layer_weight,
            smoothing_interior_weight=smoothing_interior_weight,
            smoothing_max_change_k=max_change_k,
        )

    completed_before = writer.completed_timestamps
    completed_set = set(completed_before)
    pending_times = tuple(timestamp for timestamp in process_times if timestamp not in completed_set)
    if completed_before and pending_times:
        if completed_before != process_times[: len(completed_before)]:
            writer.close(require_complete=False)
            raise RuntimeError("Resume frames are not a prefix of the requested processing interval")

    cache = RollingCandidateCache(max_history_hours=max_history_hours)
    if pending_times:
        _restore_cache(
            cache,
            writer,
            index,
            completed_before,
            pending_times[0],
            max_history_hours=max_history_hours,
        )

    rejected_low_confidence = 0
    frame_summaries: list[FrameSummary] = []
    try:
        with ThreadPoolExecutor(
            max_workers=int(query_workers),
            thread_name_prefix=f"sst-v6-s{int(series_id):02d}",
        ) as query_executor:
            for timestamp in pending_times:
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
                    query_executor=query_executor,
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
    except BaseException:
        writer.close(require_complete=False)
        raise
    else:
        writer.close(require_complete=len(process_times) == len(allocation_times))

    return SeriesSummary(
        series_id=int(series_id),
        processed_timestamps=tuple(process_times),
        proxy_history_rejected_low_confidence=rejected_low_confidence,
        frames=tuple(frame_summaries),
    )


def _run_series_worker(args: tuple) -> SeriesSummary:
    (
        source_root,
        output_root,
        series_id,
        query_workers,
        resume,
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
        resume=resume,
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
    resume: bool = False,
    smooth_synthetic: bool = False,
    smoothing_sigma: float = 0.8,
    smoothing_first_layer_weight: float = 0.25,
    smoothing_second_layer_weight: float = 0.12,
    smoothing_interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> list[SeriesSummary]:
    """Run nine causal series with at most 189 persistent query threads."""

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
    output_root = Path(output_root)
    if resume:
        if not output_root.is_dir() or not any(output_root.iterdir()):
            raise FileNotFoundError(f"Resume requires a non-empty output root: {output_root}")
    else:
        reserve_output_root(output_root)
    worker_count = min(int(series_workers), len(series_ids))
    work = [
        (
            Path(source_root),
            output_root,
            series_id,
            int(query_workers),
            bool(resume),
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
