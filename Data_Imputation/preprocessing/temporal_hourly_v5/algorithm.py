"""Vectorized, V4-equivalent causal interpolation routines for the V5 proxy."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from Data_Imputation.preprocessing.temporal_hourly_v4.algorithm import (
    KNNResult,
    _batch_local_offsets,
    _local_proxy_anchor_offsets,
    _proxy_history_trees,
    _query_tree,
    _raw_history_trees,
    _weighted_estimate,
)
from Data_Imputation.preprocessing.temporal_hourly_v4.source import RawFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.state import KNN, LAND, RAW, UNRESOLVED, ProxyFrame
from Data_Imputation.preprocessing.temporal_hourly_v5.cache import (
    ProxyTreeHistory,
    RawTreeHistory,
)


def batch_weighted_estimate(
    values: np.ndarray,
    confidence: np.ndarray,
    distance: np.ndarray,
    delta_hours: np.ndarray,
    *,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exactly V4-equivalent weighted estimates for a candidate batch."""

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
    if int(limit) < 1:
        raise ValueError("limit must be positive")

    rows, width = values.shape
    estimates = np.full(rows, np.nan, dtype=np.float64)
    quality = np.zeros(rows, dtype=np.float64)
    present = np.zeros(rows, dtype=bool)
    if width == 0:
        return estimates, quality, present

    for row in range(rows):
        estimate, evidence_quality = _weighted_estimate(
            values[row],
            confidence[row],
            distance[row],
            delta_hours[row],
            limit=int(limit),
        )
        if estimate is not None:
            estimates[row] = estimate
            quality[row] = evidence_quality
            present[row] = True
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
    """Gather one tree's queried values into fixed-width batch candidate tensors."""

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
    """Concatenate candidate groups without changing their V4 chronological order."""

    if not parts:
        empty = np.empty((batch_size, 0), dtype=np.float64)
        return empty, empty, empty, empty
    return tuple(np.concatenate(items, axis=1) for items in zip(*parts))


def progressive_confidence_knn(
    values: np.ndarray,
    source_code: np.ndarray,
    confidence: np.ndarray,
    land_mask: np.ndarray,
    raw_history: Sequence[tuple[int, RawFrame]],
    proxy_history: Sequence[tuple[int, ProxyFrame]],
    *,
    query_workers: int,
    raw_tree_history: Sequence[RawTreeHistory] | None = None,
    proxy_tree_history: Sequence[ProxyTreeHistory] | None = None,
    k: int = 20,
    source_k: int = 8,
    rebuild_interval: int = 500,
    density_radius: float = 20.0,
    proxy_minimum_confidence: float = 0.20,
    target_timestamp: datetime | None = None,
) -> KNNResult:
    """Causally complete residual gaps with V4-equivalent batched candidate evaluation."""

    result = np.asarray(values, dtype=np.float32).copy()
    codes = np.asarray(source_code, dtype=np.uint8).copy()
    confidences = np.asarray(confidence, dtype=np.float32).copy()
    land = np.asarray(land_mask, dtype=bool)
    if result.ndim != 2 or codes.shape != result.shape or confidences.shape != result.shape:
        raise ValueError("values, source_code, and confidence must be matching 2D arrays")
    if land.shape != result.shape:
        raise ValueError("land_mask must have the same shape as values")
    if int(k) < 1 or int(source_k) < 1 or int(rebuild_interval) < 1:
        raise ValueError("k, source_k, and rebuild_interval must be positive")

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
    anchor_tree, anchor_offsets = _local_proxy_anchor_offsets(
        anchor_coords,
        result[anchor_y, anchor_x].astype(np.float64),
        proxy_trees,
        query_workers=query_workers,
    )

    missing_y, missing_x = np.where(initial_missing)
    missing_coords = np.column_stack((missing_y, missing_x)).astype(np.float64)
    missing_tree = cKDTree(missing_coords)
    density = missing_tree.query_ball_point(
        missing_coords, r=float(density_radius), return_length=True
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
            query_workers=query_workers,
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
                query_workers=query_workers,
            )
        ]
        for candidates in raw_trees:
            base_parts.append(
                _queried_candidates(
                    candidates.tree,
                    candidates.values,
                    candidates.confidence,
                    batch_coords,
                    int(source_k),
                    candidates.delta_hours,
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
        )

        proxy_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for candidates in proxy_trees:
            proxy_values, proxy_confidence, proxy_distance, proxy_delta = _queried_candidates(
                candidates.tree,
                candidates.values,
                candidates.confidence,
                batch_coords,
                int(source_k) + 1,
                candidates.delta_hours,
                query_workers=query_workers,
            )
            eligible = (
                np.isfinite(proxy_values)
                & np.isfinite(proxy_confidence)
                & np.isfinite(proxy_distance)
                & (proxy_confidence >= float(proxy_minimum_confidence))
                & (proxy_distance > 0.0)
            )
            proxy_parts.append(
                (
                    np.where(eligible, proxy_values, np.nan),
                    np.where(eligible, proxy_confidence, 0.0),
                    np.where(eligible, proxy_distance, np.inf),
                    proxy_delta,
                )
            )
        proxy_values, proxy_confidence, proxy_distance, proxy_delta = _stack_candidate_parts(
            proxy_parts, len(batch_coords)
        )
        proxy_estimate, proxy_quality, proxy_present = batch_weighted_estimate(
            proxy_values,
            proxy_confidence,
            proxy_distance,
            proxy_delta,
            limit=int(k),
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
