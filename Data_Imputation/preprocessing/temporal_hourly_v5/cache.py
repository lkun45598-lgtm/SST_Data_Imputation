"""Rolling, timestamp-indexed raw and proxy KD-tree caches for the V5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy.spatial import cKDTree

from Data_Imputation.preprocessing.temporal_hourly_v4.source import RawFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.state import KNN, TEMPORAL, ProxyFrame


@dataclass(frozen=True)
class CachedRaw:
    """A raw frame and the tree of its finite original observations."""

    frame: RawFrame
    tree: cKDTree | None
    values: np.ndarray


@dataclass(frozen=True)
class CachedProxy:
    """A proxy frame and the tree of confidence-qualified synthetic pixels."""

    frame: ProxyFrame
    tree: cKDTree | None
    values: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class RawTreeHistory:
    """Cached raw candidates paired with their actual lag at a target time."""

    delta_hours: int
    tree: cKDTree
    values: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class ProxyTreeHistory:
    """Cached proxy candidates paired with their actual lag at a target time."""

    delta_hours: int
    tree: cKDTree
    values: np.ndarray
    confidence: np.ndarray


class RollingCandidateCache:
    """Keep only the causal real-time window of raw and proxy candidate trees."""

    def __init__(self, *, max_history_hours: int = 72, minimum_confidence: float = 0.20) -> None:
        if int(max_history_hours) < 1:
            raise ValueError("max_history_hours must be positive")
        self.max_history_hours = int(max_history_hours)
        self.minimum_confidence = float(minimum_confidence)
        self._raw: dict[datetime, CachedRaw] = {}
        self._proxy: dict[datetime, CachedProxy] = {}

    @staticmethod
    def _tree_and_values(valid: np.ndarray, values: np.ndarray) -> tuple[cKDTree | None, np.ndarray]:
        y, x = np.where(valid)
        if len(y) == 0:
            return None, np.empty(0, dtype=np.float64)
        coords = np.column_stack((y, x)).astype(np.float64)
        return cKDTree(coords), np.asarray(values[y, x], dtype=np.float64)

    def add_raw(self, frame: RawFrame) -> None:
        """Cache original observed-ocean candidates from one completed input frame."""

        values = np.asarray(frame.values, dtype=np.float32)
        observed = np.asarray(frame.observed, dtype=bool)
        land = np.asarray(frame.land_mask, dtype=bool)
        if values.shape != observed.shape or values.shape != land.shape:
            raise ValueError("raw frame values, observed mask, and land mask must match")
        valid = observed & ~land & np.isfinite(values)
        tree, candidate_values = self._tree_and_values(valid, values)
        self._raw[frame.timestamp] = CachedRaw(
            frame=frame,
            tree=tree,
            values=candidate_values,
        )

    def add_proxy(self, frame: ProxyFrame) -> None:
        """Cache only finite, confidence-qualified synthetic ocean candidates."""

        values = np.asarray(frame.values, dtype=np.float32)
        codes = np.asarray(frame.source_code, dtype=np.uint8)
        confidence = np.asarray(frame.confidence, dtype=np.float32)
        land = np.asarray(frame.land_mask, dtype=bool)
        if values.shape != codes.shape or values.shape != confidence.shape or values.shape != land.shape:
            raise ValueError("proxy frame values, codes, confidence, and land mask must match")
        synthetic = (codes == TEMPORAL) | (codes == KNN)
        valid = (
            synthetic
            & ~land
            & np.isfinite(values)
            & np.isfinite(confidence)
            & (confidence >= self.minimum_confidence)
        )
        tree, candidate_values = self._tree_and_values(valid, values)
        self._proxy[frame.timestamp] = CachedProxy(
            frame=frame,
            tree=tree,
            values=candidate_values,
            confidence=np.asarray(confidence[valid], dtype=np.float64),
        )

    @staticmethod
    def _delta_hours(target: datetime, timestamp: datetime) -> int | None:
        delta = (target - timestamp).total_seconds() / 3600.0
        if not float(delta).is_integer():
            return None
        return int(delta)

    def _entries_in_window(
        self,
        entries: dict[datetime, object],
        target: datetime,
        *,
        oldest_first: bool = False,
    ):
        selected: list[tuple[int, object]] = []
        for timestamp, entry in entries.items():
            if timestamp >= target:
                continue
            delta = self._delta_hours(target, timestamp)
            if delta is not None and 1 <= delta <= self.max_history_hours:
                selected.append((delta, entry))
        return sorted(selected, key=lambda item: item[0], reverse=oldest_first)

    def raw_history(self, target: datetime) -> list[tuple[int, RawFrame]]:
        """Return cached raw frames ordered by increasing actual integer lag."""

        return [
            (delta, cached.frame)
            for delta, cached in self._entries_in_window(self._raw, target)
        ]

    def raw_trees(self, target: datetime) -> list[RawTreeHistory]:
        """Return raw KD-trees ordered by increasing actual integer lag."""

        result: list[RawTreeHistory] = []
        for delta, cached in self._entries_in_window(self._raw, target):
            if cached.tree is not None:
                result.append(
                    RawTreeHistory(
                        delta_hours=delta,
                        tree=cached.tree,
                        values=cached.values,
                        confidence=np.ones(len(cached.values), dtype=np.float64),
                    )
                )
        return result

    def proxy_history(self, target: datetime) -> list[tuple[int, ProxyFrame]]:
        """Return proxy frames in V4's chronological oldest-to-newest order."""

        return [
            (delta, cached.frame)
            for delta, cached in self._entries_in_window(
                self._proxy,
                target,
                oldest_first=True,
            )
        ]

    def proxy_trees(self, target: datetime) -> list[ProxyTreeHistory]:
        """Return proxy trees in V4's chronological oldest-to-newest order."""

        result: list[ProxyTreeHistory] = []
        for delta, cached in self._entries_in_window(
            self._proxy,
            target,
            oldest_first=True,
        ):
            if cached.tree is not None:
                result.append(
                    ProxyTreeHistory(
                        delta_hours=delta,
                        tree=cached.tree,
                        values=cached.values,
                        confidence=cached.confidence,
                    )
                )
        return result

    def prune(self, newest: datetime) -> None:
        """Discard entries older than the configured causal time window."""

        cutoff = newest - timedelta(hours=self.max_history_hours)
        for entries in (self._raw, self._proxy):
            for timestamp in tuple(entries):
                if timestamp < cutoff:
                    del entries[timestamp]
