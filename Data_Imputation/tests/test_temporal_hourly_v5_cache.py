from datetime import datetime
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.preprocessing.temporal_hourly_v4.source import RawFrame
from Data_Imputation.preprocessing.temporal_hourly_v4.state import KNN, RAW, UNRESOLVED, ProxyFrame
from Data_Imputation.preprocessing.temporal_hourly_v5 import cache as cache_module
from Data_Imputation.preprocessing.temporal_hourly_v5.cache import RollingCandidateCache


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _raw_at(text: str, points: dict[tuple[int, int], float] | None = None) -> RawFrame:
    values = np.full((3, 3), np.nan, dtype=np.float32)
    observed = np.zeros((3, 3), dtype=bool)
    for (y, x), value in (points or {(1, 1): 20.0}).items():
        values[y, x] = value
        observed[y, x] = True
    return RawFrame(
        timestamp=_at(text),
        values=values,
        observed=observed,
        land_mask=np.zeros((3, 3), dtype=bool),
    )


def _proxy_at(text: str) -> ProxyFrame:
    values = np.array(
        [[20.0, 21.0, np.nan], [22.0, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )
    source_code = np.array(
        [[KNN, KNN, UNRESOLVED], [RAW, UNRESOLVED, UNRESOLVED], [UNRESOLVED] * 3],
        dtype=np.uint8,
    )
    confidence = np.array(
        [[0.20, 0.19, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    return ProxyFrame(
        timestamp=_at(text),
        values=values,
        source_code=source_code,
        confidence=confidence,
        temporal_fill_mask=source_code == 2,
        knn_fill_mask=source_code == KNN,
        land_mask=np.zeros((3, 3), dtype=bool),
    )


def test_raw_cache_returns_only_prior_frames_in_actual_72_hour_window():
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_raw(_raw_at("2024-01-01T00:00:00"))
    cache.add_raw(_raw_at("2024-01-01T03:00:00"))
    cache.add_raw(_raw_at("2023-12-28T00:00:00"))

    history = cache.raw_history(_at("2024-01-01T04:00:00"))

    assert [delta for delta, _ in history] == [1, 4]
    assert [frame.timestamp for _, frame in history] == [
        _at("2024-01-01T03:00:00"),
        _at("2024-01-01T00:00:00"),
    ]


def test_raw_tree_is_built_once_and_reused_until_timestamp_leaves_cache(monkeypatch):
    calls = 0
    original = cache_module.cKDTree

    def counting_tree(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cache_module, "cKDTree", counting_tree)
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_raw(_raw_at("2024-01-01T00:00:00"))

    first = cache.raw_trees(_at("2024-01-01T01:00:00"))
    second = cache.raw_trees(_at("2024-01-01T02:00:00"))

    assert calls == 1
    assert first[0].tree is second[0].tree
    assert first[0].delta_hours == 1
    assert second[0].delta_hours == 2
    assert first[0].tree.n == 1


def test_proxy_cache_keeps_only_eligible_synthetic_pixels():
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_proxy(_proxy_at("2024-01-01T00:00:00"))

    trees = cache.proxy_trees(_at("2024-01-01T01:00:00"))

    assert len(trees) == 1
    assert trees[0].delta_hours == 1
    assert trees[0].tree.n == 1
    np.testing.assert_array_equal(trees[0].values, np.array([20.0]))
    np.testing.assert_allclose(trees[0].confidence, np.array([0.20]), atol=1e-8, rtol=0)


def test_proxy_history_returns_cached_frames_in_actual_lag_order():
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_proxy(_proxy_at("2024-01-01T00:00:00"))
    cache.add_proxy(_proxy_at("2024-01-01T03:00:00"))

    history = cache.proxy_history(_at("2024-01-01T04:00:00"))

    assert [delta for delta, _ in history] == [1, 4]
    assert [frame.timestamp for _, frame in history] == [
        _at("2024-01-01T03:00:00"),
        _at("2024-01-01T00:00:00"),
    ]


def test_prune_discards_only_entries_before_causal_window():
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_raw(_raw_at("2024-01-01T00:00:00"))
    cache.add_raw(_raw_at("2024-01-01T01:00:00"))

    cache.prune(_at("2024-01-04T00:00:00"))

    history = cache.raw_history(_at("2024-01-04T01:00:00"))
    assert [delta for delta, _ in history] == [72]
