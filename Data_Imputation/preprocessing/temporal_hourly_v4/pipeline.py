"""Chronological causal processing for complete hourly JAXA series."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .algorithm import (
    causal_temporal_fill,
    gaussian_temporal_fills,
    progressive_confidence_knn,
)
from .source import SourceIndex
from .state import KNN, TEMPORAL, UNRESOLVED, ProxyFrame
from .writer import SeriesH5Writer, reserve_output_root


@dataclass(frozen=True)
class FrameSummary:
    """Coverage and provenance audit values for one output timestamp."""

    timestamp: datetime
    raw_fraction: float
    temporal_fraction: float
    knn_fraction: float
    proxy_history_fraction: float
    mean_confidence: float
    unresolved_count: int


@dataclass(frozen=True)
class SeriesSummary:
    """Causal audit summary returned by one completed series."""

    series_id: int
    processed_timestamps: tuple[datetime, ...]
    proxy_history_rejected_low_confidence: int
    frames: tuple[FrameSummary, ...]


def _cache_proxy_frame(frame: ProxyFrame, *, minimum_confidence: float = 0.20) -> tuple[ProxyFrame, int]:
    """Keep raw pixels but remove low-confidence synthetic pixels before caching."""

    values = np.asarray(frame.values, dtype=np.float32).copy()
    source_code = np.asarray(frame.source_code, dtype=np.uint8).copy()
    confidence = np.asarray(frame.confidence, dtype=np.float32).copy()
    temporal_mask = np.asarray(frame.temporal_fill_mask, dtype=bool).copy()
    knn_mask = np.asarray(frame.knn_fill_mask, dtype=bool).copy()
    synthetic = (source_code == TEMPORAL) | (source_code == KNN)
    rejected = synthetic & (confidence < float(minimum_confidence))
    values[rejected] = np.nan
    source_code[rejected] = UNRESOLVED
    confidence[rejected] = 0.0
    temporal_mask[rejected] = False
    knn_mask[rejected] = False
    return (
        ProxyFrame(
            timestamp=frame.timestamp,
            values=values,
            source_code=source_code,
            confidence=confidence,
            temporal_fill_mask=temporal_mask,
            knn_fill_mask=knn_mask,
            land_mask=np.asarray(frame.land_mask, dtype=bool).copy(),
        ),
        int(rejected.sum()),
    )


def rolling_proxy_history(
    cache: Dict[datetime, ProxyFrame],
    target_timestamp: datetime,
    max_history_hours: int,
) -> list[tuple[int, ProxyFrame]]:
    """Return only cached, strictly earlier frames within the real-time window."""

    history: list[tuple[int, ProxyFrame]] = []
    for timestamp in sorted(cache):
        if timestamp >= target_timestamp:
            continue
        delta_float = (target_timestamp - timestamp).total_seconds() / 3600.0
        if not delta_float.is_integer():
            continue
        delta = int(delta_float)
        if 1 <= delta <= int(max_history_hours):
            history.append((delta, cache[timestamp]))
    return history


def prune_cache(
    cache: Dict[datetime, ProxyFrame],
    newest_timestamp: datetime,
    max_history_hours: int,
) -> None:
    """Bound the causal cache to the configured time window."""

    cutoff = newest_timestamp - timedelta(hours=int(max_history_hours))
    for timestamp in tuple(cache):
        if timestamp < cutoff:
            del cache[timestamp]


def _proxy_history_fraction(
    proxy_history: Sequence[tuple[int, ProxyFrame]], land_mask: np.ndarray
) -> float:
    """Return mean usable historical synthetic coverage over available frames."""

    if not proxy_history:
        return 0.0
    ocean = ~np.asarray(land_mask, dtype=bool)
    fractions: list[float] = []
    for _, frame in proxy_history:
        codes = np.asarray(frame.source_code, dtype=np.uint8)
        confidence = np.asarray(frame.confidence, dtype=np.float32)
        usable = (
            ((codes == TEMPORAL) | (codes == KNN))
            & (confidence >= 0.20)
            & np.isfinite(frame.values)
            & ocean
        )
        fractions.append(float(usable[ocean].mean()))
    return float(np.mean(fractions))


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
) -> SeriesSummary:
    """Reconstruct one source series in strict timestamp order and stream it to H5."""

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

    cache: Dict[datetime, ProxyFrame] = {}
    rejected_low_confidence = 0
    frame_summaries: list[FrameSummary] = []
    with SeriesH5Writer.create(
        output_root,
        index,
        series_id,
        timestamps=selected_times,
    ) as writer:
        for timestamp in selected_times:
            raw = index.read(timestamp)
            raw_history = index.raw_history(timestamp, max_history_hours)
            proxy_history = rolling_proxy_history(cache, timestamp, max_history_hours)
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
                target_timestamp=timestamp,
            )
            frame = ProxyFrame(
                timestamp=timestamp,
                values=completed.values,
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
            )
            cached_frame, rejected = _cache_proxy_frame(frame)
            cache[timestamp] = cached_frame
            rejected_low_confidence += rejected
            prune_cache(cache, timestamp, max_history_hours)
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


def _run_series_worker(args: tuple[Path, Path, int, int]) -> SeriesSummary:
    source_root, output_root, series_id, query_workers = args
    return run_series(
        source_root,
        output_root,
        series_id=series_id,
        query_workers=query_workers,
        output_reserved=True,
    )


def run_all_series(
    source_root: Path,
    output_root: Path,
    *,
    series_ids: Sequence[int] = tuple(range(9)),
    series_workers: int = 9,
    query_workers: int = 21,
) -> list[SeriesSummary]:
    """Run independent annual series concurrently without parallelizing time."""

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
        (Path(source_root), Path(output_root), series_id, int(query_workers))
        for series_id in series_ids
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_run_series_worker, work))
