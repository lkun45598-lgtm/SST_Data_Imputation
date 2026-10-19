"""Causal interpolation stages for the v4 hourly SST proxy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .source import RawFrame
from .state import KNN, LAND, RAW, TEMPORAL, UNRESOLVED


@dataclass(frozen=True)
class TemporalResult:
    values: np.ndarray
    source_code: np.ndarray
    confidence: np.ndarray
    temporal_fill_mask: np.ndarray
    residual_missing: np.ndarray


@dataclass(frozen=True)
class KNNResult:
    """One fully completed frame from the causal spatial completion stage."""

    values: np.ndarray
    source_code: np.ndarray
    confidence: np.ndarray
    knn_fill_mask: np.ndarray
    local_proxy_offset: np.ndarray


@dataclass(frozen=True)
class _CandidateTree:
    """Immutable spatial candidates belonging to one historical time lag."""

    delta_hours: int
    tree: cKDTree
    values: np.ndarray
    confidence: np.ndarray


def eligible_proxy_mask(
    source_code: np.ndarray,
    confidence: np.ndarray,
    distance: np.ndarray,
    *,
    minimum_confidence: float = 0.20,
) -> np.ndarray:
    """Return synthetic historical candidates allowed to carry forward."""

    codes = np.asarray(source_code, dtype=np.uint8)
    confidences = np.asarray(confidence, dtype=np.float64)
    distances = np.asarray(distance, dtype=np.float64)
    synthetic = (codes == TEMPORAL) | (codes == KNN)
    return (
        synthetic
        & np.isfinite(confidences)
        & (confidences >= float(minimum_confidence))
        & np.isfinite(distances)
        & (distances > 0.0)
    )


def candidate_weights(
    confidence: np.ndarray,
    distance: np.ndarray,
    delta_hours: np.ndarray,
) -> np.ndarray:
    """Compute confidence, spatial, and real-time-decayed IDW weights."""

    confidences = np.asarray(confidence, dtype=np.float64)
    distances = np.asarray(distance, dtype=np.float64)
    deltas = np.asarray(delta_hours, dtype=np.float64)
    return confidences * np.power(distances + 1.0, -2.0) * np.exp(-deltas / 24.0)


def evidence_quality(
    *,
    weights: np.ndarray,
    confidence: np.ndarray,
    distance: np.ndarray,
    delta_hours: np.ndarray,
) -> float:
    """Summarize the reliability of a candidate collection in the range [0, 1]."""

    weights = np.asarray(weights, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    delta_hours = np.asarray(delta_hours, dtype=np.float64)
    valid = (
        np.isfinite(weights)
        & np.isfinite(confidence)
        & np.isfinite(distance)
        & np.isfinite(delta_hours)
        & (weights > 0.0)
    )
    if not np.any(valid):
        return 0.0
    weights = weights[valid]
    confidence = confidence[valid]
    distance = distance[valid]
    delta_hours = delta_hours[valid]
    total = float(weights.sum())
    mean_confidence = float(np.dot(weights, confidence) / total)
    mean_distance = float(np.dot(weights, distance) / total)
    mean_delta = float(np.dot(weights, delta_hours) / total)
    effective_count = float(total**2 / np.dot(weights, weights))
    quality = (
        mean_confidence
        * np.exp(-mean_distance / 20.0)
        * np.exp(-mean_delta / 24.0)
        * (1.0 - np.exp(-effective_count / 4.0))
    )
    return float(np.clip(quality, 0.0, 1.0))


def blend_alpha(*, proxy_quality: float, base_quality: float) -> float:
    """Limit historical proxy influence to at most 65 percent."""

    proxy_quality = float(np.clip(proxy_quality, 0.0, 1.0))
    base_quality = float(np.clip(base_quality, 0.0, 1.0))
    return 0.65 * proxy_quality * (1.0 - base_quality)


def _history_items(
    history: Sequence[Tuple[int, RawFrame]] | Mapping[int, RawFrame],
) -> Iterable[Tuple[int, RawFrame]]:
    items = history.items() if isinstance(history, Mapping) else history
    for delta, frame in items:
        if int(delta) >= 1:
            yield int(delta), frame


def _temporal_confidence(weights: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    weight_sum = weights.sum(axis=0)
    squared_sum = np.square(weights).sum(axis=0)
    n_eff = np.divide(
        np.square(weight_sum),
        squared_sum,
        out=np.zeros_like(weight_sum),
        where=squared_sum > 0.0,
    )
    delta_eff = np.divide(
        (weights * deltas[:, None, None]).sum(axis=0),
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 0.0,
    )
    return 0.85 * (1.0 - np.exp(-n_eff / 2.0)) * np.exp(-delta_eff / 24.0)


def causal_temporal_fill(
    target: RawFrame,
    history: Sequence[Tuple[int, RawFrame]] | Mapping[int, RawFrame],
    *,
    max_hours: int = 72,
) -> TemporalResult:
    """Fill missing ocean pixels from same-location raw historical observations."""

    values = np.asarray(target.values, dtype=np.float32).copy()
    land = np.asarray(target.land_mask, dtype=bool)
    raw_mask = np.asarray(target.observed, dtype=bool) & ~land & np.isfinite(values)
    values[~raw_mask] = np.nan
    values[land] = np.nan

    source_code = np.full(values.shape, UNRESOLVED, dtype=np.uint8)
    source_code[land] = LAND
    source_code[raw_mask] = RAW
    confidence = np.zeros(values.shape, dtype=np.float32)
    confidence[raw_mask] = 1.0

    deltas: list[int] = []
    weighted_values: list[np.ndarray] = []
    weighted_presence: list[np.ndarray] = []
    for delta, frame in _history_items(history):
        if delta > int(max_hours):
            continue
        frame_values = np.asarray(frame.values, dtype=np.float32)
        frame_observed = np.asarray(frame.observed, dtype=bool)
        frame_land = np.asarray(frame.land_mask, dtype=bool)
        if frame_values.shape != values.shape or frame_observed.shape != values.shape:
            raise ValueError("Historical frame shape does not match target frame")
        valid = frame_observed & ~frame_land & np.isfinite(frame_values) & ~land
        weight = 1.0 / float(delta)
        weighted_values.append(np.where(valid, frame_values * weight, 0.0))
        weighted_presence.append(np.where(valid, weight, 0.0))
        deltas.append(delta)

    candidate = ~raw_mask & ~land
    if weighted_presence:
        weights = np.stack(weighted_presence, axis=0).astype(np.float64)
        weighted_sum = np.stack(weighted_values, axis=0).sum(axis=0, dtype=np.float64)
        weight_sum = weights.sum(axis=0)
        fillable = candidate & (weight_sum > 0.0)
        temporal_values = np.divide(
            weighted_sum,
            weight_sum,
            out=np.zeros(values.shape, dtype=np.float64),
            where=weight_sum > 0.0,
        )
        values[fillable] = temporal_values[fillable].astype(np.float32)
        source_code[fillable] = TEMPORAL
        all_weights = np.stack(weighted_presence, axis=0).astype(np.float64)
        confidence_map = _temporal_confidence(
            all_weights, np.asarray(deltas, dtype=np.float64)
        )
        confidence[fillable] = confidence_map[fillable].astype(np.float32)
    else:
        fillable = np.zeros(values.shape, dtype=bool)

    residual_missing = ~land & ~np.isfinite(values)
    return TemporalResult(
        values=values,
        source_code=source_code,
        confidence=confidence,
        temporal_fill_mask=fillable,
        residual_missing=residual_missing,
    )


def gaussian_temporal_fills(
    values: np.ndarray,
    source_code: np.ndarray,
    temporal_fill_mask: np.ndarray,
    residual_missing: np.ndarray,
    *,
    sigma: float = 1.5,
) -> np.ndarray:
    """Apply Gaussian smoothing only to temporal fills, never raw or residual pixels."""

    result = np.asarray(values, dtype=np.float32).copy()
    codes = np.asarray(source_code, dtype=np.uint8)
    temporal = np.asarray(temporal_fill_mask, dtype=bool)
    residual = np.asarray(residual_missing, dtype=bool)
    if result.shape != codes.shape or result.shape != temporal.shape or result.shape != residual.shape:
        raise ValueError("values, source_code, and masks must have the same shape")
    valid = np.isfinite(result) & ~residual
    if not np.any(valid):
        return result
    numerator = ndimage.gaussian_filter(
        np.where(valid, result, 0.0).astype(np.float64), sigma=float(sigma), mode="nearest"
    )
    support = ndimage.gaussian_filter(
        valid.astype(np.float64), sigma=float(sigma), mode="nearest"
    )
    filtered = np.divide(
        numerator,
        support,
        out=np.full(result.shape, np.nan, dtype=np.float64),
        where=support > np.finfo(np.float64).eps,
    )
    write = temporal & (codes == TEMPORAL) & valid
    result[write] = filtered[write].astype(np.float32)
    result[residual] = np.nan
    return result


def _query_tree(
    tree: cKDTree | None,
    query_coords: np.ndarray,
    count: int,
    *,
    query_workers: int,
    distance_upper_bound: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Query a KD-tree and consistently return two-dimensional arrays."""

    count = int(count)
    if tree is None or count < 1 or tree.n < 1:
        size = len(query_coords)
        return (
            np.empty((size, 0), dtype=np.float64),
            np.empty((size, 0), dtype=np.int64),
        )
    distances, indices = tree.query(
        query_coords,
        k=min(count, tree.n),
        workers=max(1, int(query_workers)),
        distance_upper_bound=float(distance_upper_bound),
    )
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return distances, indices


def _weighted_estimate(
    values: np.ndarray,
    confidence: np.ndarray,
    distance: np.ndarray,
    delta_hours: np.ndarray,
    *,
    limit: int,
) -> tuple[float | None, float]:
    """Return the weighted estimate and its evidence quality, or no estimate."""

    values = np.asarray(values, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    delta_hours = np.asarray(delta_hours, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & np.isfinite(confidence)
        & np.isfinite(distance)
        & np.isfinite(delta_hours)
        & (confidence > 0.0)
        & (distance >= 0.0)
        & (delta_hours >= 0.0)
    )
    if not np.any(valid):
        return None, 0.0
    values = values[valid]
    confidence = confidence[valid]
    distance = distance[valid]
    delta_hours = delta_hours[valid]
    weights = candidate_weights(confidence, distance, delta_hours)
    valid_weight = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid_weight):
        return None, 0.0
    values = values[valid_weight]
    confidence = confidence[valid_weight]
    distance = distance[valid_weight]
    delta_hours = delta_hours[valid_weight]
    weights = weights[valid_weight]
    keep = min(int(limit), len(weights))
    if keep < len(weights):
        selected = np.argpartition(weights, -keep)[-keep:]
        values = values[selected]
        confidence = confidence[selected]
        distance = distance[selected]
        delta_hours = delta_hours[selected]
        weights = weights[selected]
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None, 0.0
    estimate = float(np.dot(weights, values) / total)
    if not np.isfinite(estimate):
        return None, 0.0
    return estimate, evidence_quality(
        weights=weights,
        confidence=confidence,
        distance=distance,
        delta_hours=delta_hours,
    )


def _raw_history_trees(
    history: Sequence[Tuple[int, RawFrame]],
    *,
    shape: tuple[int, int],
    target_land: np.ndarray,
) -> list[_CandidateTree]:
    """Build immutable spatial trees from historical real observations only."""

    trees: list[_CandidateTree] = []
    for delta, frame in _history_items(history):
        values = np.asarray(frame.values, dtype=np.float32)
        observed = np.asarray(frame.observed, dtype=bool)
        land = np.asarray(frame.land_mask, dtype=bool)
        if values.shape != shape or observed.shape != shape or land.shape != shape:
            raise ValueError("Historical raw frame shape does not match target frame")
        valid = observed & ~land & ~target_land & np.isfinite(values)
        y, x = np.where(valid)
        if len(y) == 0:
            continue
        trees.append(
            _CandidateTree(
                delta_hours=delta,
                tree=cKDTree(np.column_stack((y, x)).astype(np.float64)),
                values=values[y, x].astype(np.float64),
                confidence=np.ones(len(y), dtype=np.float64),
            )
        )
    return trees


def _proxy_history_trees(
    history: Sequence[Tuple[int, "ProxyFrame"]],
    *,
    shape: tuple[int, int],
    target_land: np.ndarray,
    minimum_confidence: float,
) -> list[_CandidateTree]:
    """Build immutable trees from only eligible historical synthetic values."""

    trees: list[_CandidateTree] = []
    for delta, frame in history:
        delta = int(delta)
        if delta < 1:
            raise ValueError("Historical proxy hour deltas must be positive")
        values = np.asarray(frame.values, dtype=np.float32)
        codes = np.asarray(frame.source_code, dtype=np.uint8)
        confidence = np.asarray(frame.confidence, dtype=np.float32)
        land = np.asarray(frame.land_mask, dtype=bool)
        if (
            values.shape != shape
            or codes.shape != shape
            or confidence.shape != shape
            or land.shape != shape
        ):
            raise ValueError("Historical proxy frame shape does not match target frame")
        synthetic = (codes == TEMPORAL) | (codes == KNN)
        valid = (
            synthetic
            & (confidence >= float(minimum_confidence))
            & np.isfinite(confidence)
            & np.isfinite(values)
            & ~land
            & ~target_land
        )
        y, x = np.where(valid)
        if len(y) == 0:
            continue
        trees.append(
            _CandidateTree(
                delta_hours=delta,
                tree=cKDTree(np.column_stack((y, x)).astype(np.float64)),
                values=values[y, x].astype(np.float64),
                confidence=confidence[y, x].astype(np.float64),
            )
        )
    return trees


def _local_proxy_anchor_offsets(
    raw_anchor_coords: np.ndarray,
    raw_anchor_values: np.ndarray,
    proxy_trees: Sequence[_CandidateTree],
    *,
    query_workers: int,
) -> tuple[cKDTree | None, np.ndarray]:
    """Estimate per-anchor historical proxy offsets using only collocated raw data."""

    if len(raw_anchor_coords) == 0 or not proxy_trees:
        return None, np.empty(0, dtype=np.float64)
    weighted_sum = np.zeros(len(raw_anchor_coords), dtype=np.float64)
    weight_sum = np.zeros(len(raw_anchor_coords), dtype=np.float64)
    for candidates in proxy_trees:
        distances, indices = _query_tree(
            candidates.tree,
            raw_anchor_coords,
            1,
            query_workers=query_workers,
        )
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


def _batch_local_offsets(
    anchor_tree: cKDTree | None,
    anchor_offsets: np.ndarray,
    query_coords: np.ndarray,
    *,
    query_workers: int,
    radius: float = 48.0,
    minimum_anchors: int = 5,
) -> np.ndarray:
    """Use a robust local offset only with the required number of raw anchors."""

    result = np.zeros(len(query_coords), dtype=np.float64)
    if anchor_tree is None or anchor_tree.n < int(minimum_anchors):
        return result
    distances, indices = _query_tree(
        anchor_tree,
        query_coords,
        int(minimum_anchors),
        query_workers=query_workers,
        distance_upper_bound=radius,
    )
    for row in range(len(query_coords)):
        valid = np.isfinite(distances[row]) & (indices[row] < anchor_tree.n)
        if int(valid.sum()) >= int(minimum_anchors):
            result[row] = float(np.median(anchor_offsets[indices[row, valid]]))
    return result


def progressive_confidence_knn(
    values: np.ndarray,
    source_code: np.ndarray,
    confidence: np.ndarray,
    land_mask: np.ndarray,
    raw_history: Sequence[Tuple[int, RawFrame]],
    proxy_history: Sequence[Tuple[int, "ProxyFrame"]],
    *,
    query_workers: int,
    k: int = 20,
    source_k: int = 8,
    rebuild_interval: int = 500,
    density_radius: float = 20.0,
    proxy_minimum_confidence: float = 0.20,
    target_timestamp: datetime | None = None,
) -> KNNResult:
    """Causally complete residual ocean gaps with confidence-controlled proxies.

    Raw target and raw historical observations form the primary evidence set.
    Historical synthetic values can only supplement it after their confidence,
    temporal decay, spatial decay, and nonzero query distance are all checked.
    """

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

    raw_trees = _raw_history_trees(raw_history, shape=result.shape, target_land=land)
    proxy_trees = _proxy_history_trees(
        proxy_history,
        shape=result.shape,
        target_land=land,
        minimum_confidence=proxy_minimum_confidence,
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
        current_distances, current_indices = _query_tree(
            current_tree,
            batch_coords,
            int(k),
            query_workers=query_workers,
        )

        raw_rows: list[tuple[_CandidateTree, np.ndarray, np.ndarray]] = []
        for candidates in raw_trees:
            distances, indices = _query_tree(
                candidates.tree,
                batch_coords,
                int(source_k),
                query_workers=query_workers,
            )
            raw_rows.append((candidates, distances, indices))

        proxy_rows: list[tuple[_CandidateTree, np.ndarray, np.ndarray]] = []
        for candidates in proxy_trees:
            # Ask for one extra point so a prohibited zero-distance value does
            # not hide the nearest usable historical synthetic candidate.
            distances, indices = _query_tree(
                candidates.tree,
                batch_coords,
                int(source_k) + 1,
                query_workers=query_workers,
            )
            proxy_rows.append((candidates, distances, indices))

        for row, (y, x) in enumerate(zip(batch_y, batch_x)):
            base_values: list[np.ndarray] = []
            base_confidence: list[np.ndarray] = []
            base_distance: list[np.ndarray] = []
            base_delta: list[np.ndarray] = []
            if current_distances.shape[1]:
                indices = current_indices[row]
                valid = indices < len(current_values)
                if np.any(valid):
                    base_values.append(current_values[indices[valid]])
                    base_confidence.append(current_confidence[indices[valid]])
                    base_distance.append(current_distances[row, valid])
                    base_delta.append(np.zeros(int(valid.sum()), dtype=np.float64))
            for candidates, distances, indices in raw_rows:
                row_indices = indices[row]
                valid = row_indices < len(candidates.values)
                if np.any(valid):
                    base_values.append(candidates.values[row_indices[valid]])
                    base_confidence.append(candidates.confidence[row_indices[valid]])
                    base_distance.append(distances[row, valid])
                    base_delta.append(
                        np.full(int(valid.sum()), candidates.delta_hours, dtype=np.float64)
                    )

            if base_values:
                base_estimate, base_quality = _weighted_estimate(
                    np.concatenate(base_values),
                    np.concatenate(base_confidence),
                    np.concatenate(base_distance),
                    np.concatenate(base_delta),
                    limit=int(k),
                )
            else:
                base_estimate, base_quality = None, 0.0

            proxy_values: list[np.ndarray] = []
            proxy_confidence: list[np.ndarray] = []
            proxy_distance: list[np.ndarray] = []
            proxy_delta: list[np.ndarray] = []
            for candidates, distances, indices in proxy_rows:
                row_indices = indices[row]
                valid_index = row_indices < len(candidates.values)
                if not np.any(valid_index):
                    continue
                row_distance = distances[row, valid_index]
                row_confidence = candidates.confidence[row_indices[valid_index]]
                eligible = eligible_proxy_mask(
                    np.full(len(row_distance), KNN, dtype=np.uint8),
                    row_confidence,
                    row_distance,
                    minimum_confidence=proxy_minimum_confidence,
                )
                if np.any(eligible):
                    selected = row_indices[valid_index][eligible]
                    proxy_values.append(candidates.values[selected])
                    proxy_confidence.append(row_confidence[eligible])
                    proxy_distance.append(row_distance[eligible])
                    proxy_delta.append(
                        np.full(int(eligible.sum()), candidates.delta_hours, dtype=np.float64)
                    )

            if proxy_values:
                proxy_estimate, proxy_quality = _weighted_estimate(
                    np.concatenate(proxy_values),
                    np.concatenate(proxy_confidence),
                    np.concatenate(proxy_distance),
                    np.concatenate(proxy_delta),
                    limit=int(k),
                )
                if proxy_estimate is not None:
                    proxy_estimate += float(offsets[row])
            else:
                proxy_estimate, proxy_quality = None, 0.0

            if base_estimate is None and proxy_estimate is None:
                stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
                raise RuntimeError(
                    f"{stamp}: no valid interpolation seed for pixel ({int(y)}, {int(x)})"
                )
            if base_estimate is None:
                final_value = float(proxy_estimate)
                alpha = 1.0
                final_confidence = proxy_quality
            elif proxy_estimate is None:
                final_value = float(base_estimate)
                alpha = 0.0
                final_confidence = base_quality
            else:
                alpha = blend_alpha(
                    proxy_quality=proxy_quality,
                    base_quality=base_quality,
                )
                final_value = (1.0 - alpha) * float(base_estimate) + alpha * float(proxy_estimate)
                final_confidence = (1.0 - alpha) * base_quality + alpha * proxy_quality
            if not np.isfinite(final_value):
                raise RuntimeError(f"Non-finite interpolation at pixel ({int(y)}, {int(x)})")
            result[y, x] = np.float32(final_value)
            codes[y, x] = KNN
            confidences[y, x] = np.float32(min(0.75, max(0.0, final_confidence)))
            knn_fill_mask[y, x] = True
            local_proxy_offset[y, x] = np.float32(offsets[row])

    unresolved = ocean & ~np.isfinite(result)
    if np.any(unresolved):
        stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
        raise RuntimeError(f"{stamp}: progressive KNN left {int(unresolved.sum())} ocean NaNs")
    result[land] = np.nan
    return KNNResult(result, codes, confidences, knn_fill_mask, local_proxy_offset)
