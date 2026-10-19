"""Causal temporal, Gaussian, and progressive spatial interpolation stages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .source import SourceFrame


@dataclass(frozen=True)
class TemporalFillResult:
    values: np.ndarray
    residual_missing: np.ndarray
    temporal_fill_mask: np.ndarray


def _history_items(
    history: Sequence[Tuple[int, SourceFrame]] | Mapping[int, SourceFrame],
) -> Iterable[Tuple[int, SourceFrame]]:
    items = history.items() if isinstance(history, Mapping) else history
    for delta, frame in items:
        delta = int(delta)
        if delta < 1:
            continue
        yield delta, frame


def causal_temporal_fill(
    target: SourceFrame,
    history: Sequence[Tuple[int, SourceFrame]] | Mapping[int, SourceFrame],
    *,
    max_hours: int = 72,
) -> TemporalFillResult:
    """Fill missing ocean pixels from same-location raw historical values.

    ``delta`` is supplied by the timestamp index, so missing hours are never
    silently treated as adjacent observations. Raw target observations are
    immutable and historical values are accepted only where their source frame
    marks ``observed``.
    """

    values = np.asarray(target.values, dtype=np.float32).copy()
    land = np.asarray(target.land_mask, dtype=bool)
    observed = np.asarray(target.observed, dtype=bool) & ~land & np.isfinite(values)
    values[land] = np.nan

    candidate = (~observed) & (~land) & ~np.isfinite(values)
    weighted_sum = np.zeros(values.shape, dtype=np.float64)
    weight_sum = np.zeros(values.shape, dtype=np.float64)
    for delta, frame in _history_items(history):
        if delta > max_hours:
            continue
        frame_land = np.asarray(frame.land_mask, dtype=bool)
        source_mask = (
            np.asarray(frame.observed, dtype=bool)
            & ~frame_land
            & np.isfinite(frame.values)
        )
        if source_mask.shape != values.shape:
            raise ValueError("Historical frame shape does not match target frame")
        weight = 1.0 / float(delta)
        weighted_sum[source_mask] += np.asarray(frame.values, dtype=np.float64)[source_mask] * weight
        weight_sum[source_mask] += weight

    fillable = candidate & (weight_sum > 0.0)
    if np.any(fillable):
        values[fillable] = (weighted_sum[fillable] / weight_sum[fillable]).astype(np.float32)
    temporal_fill_mask = fillable
    residual_missing = (~np.isfinite(values)) & ~land
    return TemporalFillResult(
        values=values,
        residual_missing=residual_missing,
        temporal_fill_mask=temporal_fill_mask,
    )


def gaussian_temporal_fills(
    values: np.ndarray,
    residual_missing: np.ndarray,
    temporal_fill_mask: np.ndarray,
    *,
    sigma: float = 1.5,
) -> np.ndarray:
    """Smooth only Stage-1 temporal fills using the existing Gaussian rule."""

    result = np.asarray(values, dtype=np.float32).copy()
    residual_missing = np.asarray(residual_missing, dtype=bool)
    temporal_fill_mask = np.asarray(temporal_fill_mask, dtype=bool)
    valid_input = np.isfinite(result) & ~residual_missing
    if not np.any(valid_input):
        return result

    mean_value = float(np.nanmean(result[valid_input]))
    convolution_input = result.copy()
    convolution_input[~valid_input] = mean_value
    filtered = ndimage.gaussian_filter(convolution_input, sigma=sigma)

    write = temporal_fill_mask & valid_input
    result[write] = filtered[write].astype(np.float32)
    result[residual_missing] = np.nan
    return result


def _query_tree(
    tree: cKDTree | None,
    query_coords: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tree is None or count <= 0:
        n = len(query_coords)
        return np.empty((n, 0), dtype=np.float64), np.empty((n, 0), dtype=np.int64)
    distances, indices = tree.query(query_coords, k=min(count, tree.n))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def progressive_spatiotemporal_knn(
    values: np.ndarray,
    observed_mask: np.ndarray,
    land_mask: np.ndarray,
    history: Sequence[Tuple[int, SourceFrame]],
    *,
    target_timestamp: datetime | None = None,
    temporal_fill_mask: np.ndarray | None = None,
    k: int = 20,
    source_k: int = 8,
    power_space: float = 2.0,
    power_time: float = 1.0,
    temporal_reliability: float = 0.5,
    density_radius: float = 20.0,
    rebuild_interval: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete one frame with causal spatiotemporal IDW interpolation.

    Historical candidates are built exclusively from ``SourceFrame.observed``
    and remain fixed throughout this call. Newly generated current-frame
    values are available only to later pixels in this same frame.
    """

    result = np.asarray(values, dtype=np.float32).copy()
    observed = np.asarray(observed_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if result.ndim != 2 or observed.shape != result.shape or land.shape != result.shape:
        raise ValueError("values, observed_mask, and land_mask must have the same 2D shape")
    if k < 1 or source_k < 1:
        raise ValueError("k and source_k must be positive")
    ocean = ~land
    result[land] = np.nan
    initial_missing = ocean & ~np.isfinite(result)
    knn_mask = np.zeros(result.shape, dtype=bool)
    if not np.any(initial_missing):
        return result, knn_mask

    temporal_mask = (
        np.asarray(temporal_fill_mask, dtype=bool)
        if temporal_fill_mask is not None
        else np.zeros(result.shape, dtype=bool)
    )
    if temporal_mask.shape != result.shape:
        raise ValueError("temporal_fill_mask must have the same shape as values")

    # Build the immutable historical candidate trees from raw observations.
    history_trees: list[tuple[int, cKDTree, np.ndarray]] = []
    for delta, frame in history:
        delta = int(delta)
        if delta < 1:
            raise ValueError("historical hour deltas must be positive")
        frame_values = np.asarray(frame.values, dtype=np.float32)
        frame_observed = np.asarray(frame.observed, dtype=bool)
        frame_land = np.asarray(frame.land_mask, dtype=bool)
        if frame_values.shape != result.shape or frame_observed.shape != result.shape:
            raise ValueError("Historical frame shape does not match target frame")
        valid = frame_observed & ~frame_land & np.isfinite(frame_values)
        y, x = np.where(valid)
        if len(y) == 0:
            continue
        coords = np.column_stack((y, x)).astype(np.float64)
        history_trees.append((delta, cKDTree(coords), frame_values[y, x].astype(np.float64)))

    raw_seed = ocean & observed & np.isfinite(result)
    current_seed = ocean & np.isfinite(result)
    if not np.any(current_seed) and not history_trees:
        stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
        raise RuntimeError(f"{stamp}: no valid interpolation seed for ocean gap")

    missing_y, missing_x = np.where(initial_missing)
    missing_coords = np.column_stack((missing_y, missing_x)).astype(np.float64)
    missing_tree = cKDTree(missing_coords)
    density = missing_tree.query_ball_point(
        missing_coords, r=float(density_radius), return_length=True
    )
    order = np.argsort(np.asarray(density), kind="stable")
    sorted_y = missing_y[order]
    sorted_x = missing_x[order]
    batch_size = max(1, int(rebuild_interval))

    for batch_start in range(0, len(sorted_y), batch_size):
        batch_end = min(batch_start + batch_size, len(sorted_y))
        batch_y = sorted_y[batch_start:batch_end]
        batch_x = sorted_x[batch_start:batch_end]
        batch_coords = np.column_stack((batch_y, batch_x)).astype(np.float64)

        # Rebuild the current tree at the start of each progressive batch.
        current_y, current_x = np.where(ocean & np.isfinite(result))
        current_coords = np.column_stack((current_y, current_x)).astype(np.float64)
        current_values = result[current_y, current_x].astype(np.float64)
        current_reliability = np.where(
            raw_seed[current_y, current_x],
            1.0,
            np.where(temporal_mask[current_y, current_x], temporal_reliability, temporal_reliability),
        ).astype(np.float64)
        current_tree = cKDTree(current_coords) if len(current_coords) else None
        current_distances, current_indices = _query_tree(current_tree, batch_coords, k)

        # Historical queries are vectorized per source frame, then combined
        # with current candidates for each target pixel below.
        historical_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        for delta, tree, source_values in history_trees:
            distances, indices = _query_tree(tree, batch_coords, source_k)
            historical_rows.append((delta, distances, source_values[indices]))

        pending_coords: list[np.ndarray] = []
        pending_values: list[float] = []
        pending_reliability: list[float] = []

        for row, (y, x) in enumerate(zip(batch_y, batch_x)):
            candidate_distances: list[np.ndarray] = []
            candidate_values: list[np.ndarray] = []
            candidate_reliability: list[np.ndarray] = []
            candidate_delta: list[np.ndarray] = []

            if current_distances.shape[1]:
                candidate_distances.append(current_distances[row])
                candidate_values.append(current_values[current_indices[row]])
                candidate_reliability.append(current_reliability[current_indices[row]])
                candidate_delta.append(np.zeros(current_distances.shape[1], dtype=np.float64))

            for delta, distances, source_values in historical_rows:
                candidate_distances.append(distances[row])
                candidate_values.append(source_values[row])
                candidate_reliability.append(np.ones(distances.shape[1], dtype=np.float64))
                candidate_delta.append(np.full(distances.shape[1], delta, dtype=np.float64))

            if pending_coords:
                pending_array = np.asarray(pending_coords, dtype=np.float64)
                distance = np.sqrt(np.sum((pending_array - np.asarray([y, x])) ** 2, axis=1))
                candidate_distances.append(distance)
                candidate_values.append(np.asarray(pending_values, dtype=np.float64))
                candidate_reliability.append(np.asarray(pending_reliability, dtype=np.float64))
                candidate_delta.append(np.zeros(len(pending_coords), dtype=np.float64))

            if not candidate_distances:
                stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
                raise RuntimeError(f"{stamp}: no valid interpolation seed for pixel ({y}, {x})")

            distances = np.concatenate(candidate_distances)
            candidate_values_array = np.concatenate(candidate_values)
            reliability = np.concatenate(candidate_reliability)
            delta_hours = np.concatenate(candidate_delta)
            valid = np.isfinite(distances) & np.isfinite(candidate_values_array) & (reliability > 0)
            if not np.any(valid):
                stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
                raise RuntimeError(f"{stamp}: no valid interpolation seed for pixel ({y}, {x})")
            distances = distances[valid]
            candidate_values_array = candidate_values_array[valid]
            reliability = reliability[valid]
            delta_hours = delta_hours[valid]
            weights = reliability * np.power(distances + 1e-6, -power_space)
            weights *= np.power(1.0 + delta_hours, -power_time)
            keep = min(int(k), len(weights))
            if keep < len(weights):
                selected = np.argpartition(weights, -keep)[-keep:]
                weights = weights[selected]
                candidate_values_array = candidate_values_array[selected]
            weight_sum = float(weights.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
                raise RuntimeError(f"{stamp}: invalid interpolation weights for pixel ({y}, {x})")
            interpolated = float(np.dot(weights, candidate_values_array) / weight_sum)
            if not np.isfinite(interpolated):
                raise RuntimeError(f"non-finite interpolation at pixel ({y}, {x})")
            result[y, x] = np.float32(interpolated)
            knn_mask[y, x] = True
            pending_coords.append(np.asarray([y, x], dtype=np.float64))
            pending_values.append(interpolated)
            pending_reliability.append(1.0)

    unresolved = ocean & ~np.isfinite(result)
    if np.any(unresolved):
        stamp = target_timestamp.isoformat() if target_timestamp is not None else "unknown"
        raise ValueError(f"{stamp}: progressive KNN left {int(unresolved.sum())} ocean NaNs")
    result[land] = np.nan
    return result, knn_mask
