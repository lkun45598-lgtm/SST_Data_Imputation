"""V5-equivalent interpolation with persistent cross-tree parallelism."""

from __future__ import annotations

from concurrent.futures import Executor
from datetime import datetime
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from Data_Imputation.preprocessing.temporal_hourly_v4.algorithm import (
    KNNResult,
    _batch_local_offsets,
    _query_tree,
    _raw_history_trees,
    _proxy_history_trees,
    _weighted_estimate,
    candidate_weights,
)
from Data_Imputation.preprocessing.temporal_hourly_v4.source import RawFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.state import (
    KNN,
    LAND,
    RAW,
    UNRESOLVED,
    ProxyFrame,
)
from Data_Imputation.preprocessing.temporal_hourly_v5.cache import (
    ProxyTreeHistory,
    RawTreeHistory,
)


def _weighted_chunk(
    values: np.ndarray,
    confidence: np.ndarray,
    distance: np.ndarray,
    delta_hours: np.ndarray,
    start: int,
    end: int,
    limit: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    estimates = np.full(end - start, np.nan, dtype=np.float64)
    quality = np.zeros(end - start, dtype=np.float64)
    present = np.zeros(end - start, dtype=bool)
    for output_row, row in enumerate(range(start, end)):
        estimate, evidence_quality = _weighted_estimate(
            values[row],
            confidence[row],
            distance[row],
            delta_hours[row],
            limit=int(limit),
        )
        if estimate is not None:
            estimates[output_row] = estimate
            quality[output_row] = evidence_quality
            present[output_row] = True
    return start, end, estimates, quality, present


def batch_weighted_estimate(
    values: np.ndarray,
    confidence: np.ndarray,
    distance: np.ndarray,
    delta_hours: np.ndarray,
    *,
    limit: int,
    executor: Executor | None = None,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate independent rows concurrently without changing row arithmetic."""

    values = np.asarray(values, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    delta_hours = np.asarray(delta_hours, dtype=np.float64)
    if (
        values.ndim != 2
        or confidence.shape != values.shape
        or distance.shape != values.shape
        or delta_hours.shape != values.shape
    ):
        raise ValueError("candidate values, confidence, distance, and deltas must be matching 2D arrays")
    if int(limit) < 1 or int(workers) < 1:
        raise ValueError("limit and workers must be positive")

    rows, width = values.shape
    estimates = np.full(rows, np.nan, dtype=np.float64)
    quality = np.zeros(rows, dtype=np.float64)
    present = np.zeros(rows, dtype=bool)
    if width == 0 or rows == 0:
        return estimates, quality, present

    task_count = min(int(workers), rows) if executor is not None else 1
    boundaries = np.linspace(0, rows, task_count + 1, dtype=int)
    if executor is None:
        chunks = [
            _weighted_chunk(
                values,
                confidence,
                distance,
                delta_hours,
                0,
                rows,
                int(limit),
            )
        ]
    else:
        futures = [
            executor.submit(
                _weighted_chunk,
                values,
                confidence,
                distance,
                delta_hours,
                int(start),
                int(end),
                int(limit),
            )
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if end > start
        ]
        chunks = [future.result() for future in futures]
    for start, end, chunk_estimates, chunk_quality, chunk_present in chunks:
        estimates[start:end] = chunk_estimates
        quality[start:end] = chunk_quality
        present[start:end] = chunk_present
    return estimates, quality, present


def _queried_candidates(
    tree: cKDTree | None,
    candidate_values: np.ndarray,
    candidate_confidence: np.ndarray,
    query_coords: np.ndarray,
    count: int,
    delta_hours: int,
    *,
    query_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances, indices = _query_tree(
        tree,
        query_coords,
        count,
        query_workers=query_workers,
    )
    if distances.shape[1] == 0:
        empty = np.empty((len(query_coords), 0), dtype=np.float64)
        return empty, empty, empty, empty
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    candidate_confidence = np.asarray(candidate_confidence, dtype=np.float64)
    valid = indices < len(candidate_values)
    safe_indices = np.minimum(indices, max(0, len(candidate_values) - 1))
    values = np.where(valid, candidate_values[safe_indices], np.nan)
    confidence = np.where(valid, candidate_confidence[safe_indices], 0.0)
    deltas = np.full(distances.shape, float(delta_hours), dtype=np.float64)
    return values, confidence, distances, deltas


def _stack_candidate_parts(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not parts:
        empty = np.empty((batch_size, 0), dtype=np.float64)
        return empty, empty, empty, empty
    return tuple(np.concatenate(items, axis=1) for items in zip(*parts))


def _query_histories(
    histories: Sequence[RawTreeHistory | ProxyTreeHistory],
    query_coords: np.ndarray,
    count: int,
    *,
    executor: Executor | None,
    query_workers: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    per_tree_workers = 1 if executor is not None else int(query_workers)
    if executor is None:
        return [
            _queried_candidates(
                candidates.tree,
                candidates.values,
                candidates.confidence,
                query_coords,
                count,
                candidates.delta_hours,
                query_workers=per_tree_workers,
            )
            for candidates in histories
        ]
    futures = [
        executor.submit(
            _queried_candidates,
            candidates.tree,
            candidates.values,
            candidates.confidence,
            query_coords,
            count,
            candidates.delta_hours,
            query_workers=per_tree_workers,
        )
        for candidates in histories
    ]
    return [future.result() for future in futures]


def _parallel_local_proxy_anchor_offsets(
    raw_anchor_coords: np.ndarray,
    raw_anchor_values: np.ndarray,
    proxy_trees: Sequence[ProxyTreeHistory],
    *,
    executor: Executor | None,
    query_workers: int,
) -> tuple[cKDTree | None, np.ndarray]:
    if len(raw_anchor_coords) == 0 or not proxy_trees:
        return None, np.empty(0, dtype=np.float64)

    def query_exact(candidates: ProxyTreeHistory):
        return _query_tree(
            candidates.tree,
            raw_anchor_coords,
            1,
            query_workers=1 if executor is not None else int(query_workers),
        )

    if executor is None:
        queried = [query_exact(candidates) for candidates in proxy_trees]
    else:
        queried = [future.result() for future in [executor.submit(query_exact, item) for item in proxy_trees]]

    weighted_sum = np.zeros(len(raw_anchor_coords), dtype=np.float64)
    weight_sum = np.zeros(len(raw_anchor_coords), dtype=np.float64)
    for candidates, (distances, indices) in zip(proxy_trees, queried):
        distance = distances[:, 0]
        index = indices[:, 0]
        exact = np.isfinite(distance) & (distance == 0.0) & (index < candidates.tree.n)
        if not np.any(exact):
            continue
        selected = index[exact]
        weights = candidate_weights(
            candidates.confidence[selected],
            distance[exact],
            np.full(int(exact.sum()), candidates.delta_hours, dtype=np.float64),
        )
        weighted_sum[exact] += weights * candidates.values[selected]
        weight_sum[exact] += weights
    paired = weight_sum > 0.0
    if not np.any(paired):
        return None, np.empty(0, dtype=np.float64)
    proxy_estimate = weighted_sum[paired] / weight_sum[paired]
    offsets = np.asarray(raw_anchor_values, dtype=np.float64)[paired] - proxy_estimate
    return cKDTree(np.asarray(raw_anchor_coords, dtype=np.float64)[paired]), offsets


def progressive_confidence_knn(
    values: np.ndarray,
    source_code: np.ndarray,
    confidence: np.ndarray,
    land_mask: np.ndarray,
    raw_history: Sequence[tuple[int, RawFrame]],
    proxy_history: Sequence[tuple[int, ProxyFrame]],
    *,
    query_workers: int,
    query_executor: Executor | None = None,
    raw_tree_history: Sequence[RawTreeHistory] | None = None,
    proxy_tree_history: Sequence[ProxyTreeHistory] | None = None,
    k: int = 20,
    source_k: int = 8,
    rebuild_interval: int = 500,
    density_radius: float = 20.0,
    proxy_minimum_confidence: float = 0.20,
    target_timestamp: datetime | None = None,
) -> KNNResult:
    """Complete residual gaps with ordered, cross-tree parallel evaluation."""

    result = np.asarray(values, dtype=np.float32).copy()
    codes = np.asarray(source_code, dtype=np.uint8).copy()
    confidences = np.asarray(confidence, dtype=np.float32).copy()
    land = np.asarray(land_mask, dtype=bool)
    if result.ndim != 2 or codes.shape != result.shape or confidences.shape != result.shape:
        raise ValueError("values, source_code, and confidence must be matching 2D arrays")
    if land.shape != result.shape:
        raise ValueError("land_mask must have the same shape as values")
    if int(k) < 1 or int(source_k) < 1 or int(rebuild_interval) < 1 or int(query_workers) < 1:
        raise ValueError("k, source_k, rebuild_interval, and query_workers must be positive")

    ocean = ~land
    result[land] = np.nan
    codes[land] = LAND
    confidences[land] = 0.0
    initial_missing = ocean & ~np.isfinite(result)
    knn_fill_mask = np.zeros(result.shape, dtype=bool)
    local_proxy_offset = np.zeros(result.shape, dtype=np.float32)
    if not np.any(initial_missing):
        return KNNResult(result, codes, confidences, knn_fill_mask, local_proxy_offset)

    raw_trees = (
        list(raw_tree_history)
        if raw_tree_history is not None
        else _raw_history_trees(raw_history, shape=result.shape, target_land=land)
    )
    proxy_trees = (
        list(proxy_tree_history)
        if proxy_tree_history is not None
        else _proxy_history_trees(
            proxy_history,
            shape=result.shape,
            target_land=land,
            minimum_confidence=proxy_minimum_confidence,
        )
    )
    initial_current = ocean & np.isfinite(result) & (confidences > 0.0)
    if not np.any(initial_current) and not raw_trees and not proxy_trees:
        stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
        raise RuntimeError(f"{stamp}: no valid interpolation seed for ocean gap")

    raw_anchor_mask = ocean & np.isfinite(result) & (codes == RAW)
    anchor_y, anchor_x = np.where(raw_anchor_mask)
    anchor_coords = np.column_stack((anchor_y, anchor_x)).astype(np.float64)
    anchor_tree, anchor_offsets = _parallel_local_proxy_anchor_offsets(
        anchor_coords,
        result[anchor_y, anchor_x].astype(np.float64),
        proxy_trees,
        executor=query_executor,
        query_workers=query_workers,
    )

    missing_y, missing_x = np.where(initial_missing)
    missing_coords = np.column_stack((missing_y, missing_x)).astype(np.float64)
    missing_tree = cKDTree(missing_coords)
    density = missing_tree.query_ball_point(
        missing_coords,
        r=float(density_radius),
        return_length=True,
        workers=int(query_workers) if query_executor is not None else 1,
    )
    order = np.argsort(np.asarray(density), kind="stable")
    sorted_y = missing_y[order]
    sorted_x = missing_x[order]
    batch_size = int(rebuild_interval)

    for batch_start in range(0, len(sorted_y), batch_size):
        batch_end = min(batch_start + batch_size, len(sorted_y))
        batch_y = sorted_y[batch_start:batch_end]
        batch_x = sorted_x[batch_start:batch_end]
        batch_coords = np.column_stack((batch_y, batch_x)).astype(np.float64)
        offsets = _batch_local_offsets(
            anchor_tree,
            anchor_offsets,
            batch_coords,
            query_workers=1 if query_executor is not None else int(query_workers),
        )

        current_mask = ocean & np.isfinite(result) & (confidences > 0.0)
        current_y, current_x = np.where(current_mask)
        current_coords = np.column_stack((current_y, current_x)).astype(np.float64)
        current_tree = cKDTree(current_coords) if len(current_coords) else None
        current_values = result[current_y, current_x].astype(np.float64)
        current_confidence = confidences[current_y, current_x].astype(np.float64)
        base_parts = [
            _queried_candidates(
                current_tree,
                current_values,
                current_confidence,
                batch_coords,
                int(k),
                0,
                query_workers=1 if query_executor is not None else int(query_workers),
            )
        ]
        base_parts.extend(
            _query_histories(
                raw_trees,
                batch_coords,
                int(source_k),
                executor=query_executor,
                query_workers=query_workers,
            )
        )
        base_values, base_confidence, base_distance, base_delta = _stack_candidate_parts(
            base_parts, len(batch_coords)
        )
        base_estimate, base_quality, base_present = batch_weighted_estimate(
            base_values,
            base_confidence,
            base_distance,
            base_delta,
            limit=int(k),
            executor=query_executor,
            workers=query_workers,
        )

        proxy_parts = _query_histories(
            proxy_trees,
            batch_coords,
            int(source_k) + 1,
            executor=query_executor,
            query_workers=query_workers,
        )
        eligible_proxy_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for proxy_values, proxy_confidence, proxy_distance, proxy_delta in proxy_parts:
            eligible = (
                np.isfinite(proxy_values)
                & np.isfinite(proxy_confidence)
                & np.isfinite(proxy_distance)
                & (proxy_confidence >= float(proxy_minimum_confidence))
                & (proxy_distance > 0.0)
            )
            eligible_proxy_parts.append(
                (
                    np.where(eligible, proxy_values, np.nan),
                    np.where(eligible, proxy_confidence, 0.0),
                    np.where(eligible, proxy_distance, np.inf),
                    proxy_delta,
                )
            )
        proxy_values, proxy_confidence, proxy_distance, proxy_delta = _stack_candidate_parts(
            eligible_proxy_parts, len(batch_coords)
        )
        proxy_estimate, proxy_quality, proxy_present = batch_weighted_estimate(
            proxy_values,
            proxy_confidence,
            proxy_distance,
            proxy_delta,
            limit=int(k),
            executor=query_executor,
            workers=query_workers,
        )
        proxy_estimate[proxy_present] += offsets[proxy_present]

        unresolved_rows = ~(base_present | proxy_present)
        if np.any(unresolved_rows):
            row = int(np.flatnonzero(unresolved_rows)[0])
            stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
            raise RuntimeError(
                f"{stamp}: no valid interpolation seed for pixel ({int(batch_y[row])}, {int(batch_x[row])})"
            )

        alpha = 0.65 * np.clip(proxy_quality, 0.0, 1.0) * (1.0 - np.clip(base_quality, 0.0, 1.0))
        final_values = np.where(
            base_present & proxy_present,
            (1.0 - alpha) * base_estimate + alpha * proxy_estimate,
            np.where(base_present, base_estimate, proxy_estimate),
        )
        final_confidence = np.where(
            base_present & proxy_present,
            (1.0 - alpha) * base_quality + alpha * proxy_quality,
            np.where(base_present, base_quality, proxy_quality),
        )
        if not np.all(np.isfinite(final_values)):
            row = int(np.flatnonzero(~np.isfinite(final_values))[0])
            raise RuntimeError(
                f"Non-finite interpolation at pixel ({int(batch_y[row])}, {int(batch_x[row])})"
            )
        result[batch_y, batch_x] = final_values.astype(np.float32)
        codes[batch_y, batch_x] = KNN
        confidences[batch_y, batch_x] = np.clip(final_confidence, 0.0, 0.75).astype(np.float32)
        knn_fill_mask[batch_y, batch_x] = True
        local_proxy_offset[batch_y, batch_x] = offsets.astype(np.float32)

    unresolved = ocean & ~np.isfinite(result)
    if np.any(unresolved):
        stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
        raise RuntimeError(f"{stamp}: progressive KNN left {int(unresolved.sum())} ocean NaNs")
    result[land] = np.nan
    return KNNResult(result, codes, confidences, knn_fill_mask, local_proxy_offset)
