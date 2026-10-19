# Temporal Hourly V5 Equivalent Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated V5 hourly JAXA proxy pipeline that materially reduces V4 wall-clock time while preserving V4's causal evidence, interpolation semantics, and scientific accuracy.

**Architecture:** V5 keeps V4's real-timestamp order, 72-hour raw/proxy policy, temporal fill, Gaussian stage, and progressive 500-pixel update boundary. It holds raw and eligible proxy KD-trees in a rolling per-series cache and computes the fixed-width KNN candidate tensors for each batch with NumPy rather than one Python loop per missing pixel. V4 remains unchanged as the numerical reference.

**Tech Stack:** Python 3, NumPy, SciPy `cKDTree`, h5py, pytest, existing V4 evaluation helpers.

**Repository constraint:** `.git` is read-only and the working directory contains user changes. Do not stage, revert, commit, or create a worktree. Use scoped tests and `git diff --check` as checkpoints. Never modify old hourly H5, checkpoints, paper caches, or `temporal_hourly_v4`.

---

## File Structure

- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/__init__.py` -- V5 package marker and version.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/cache.py` -- rolling raw/proxy tree cache keyed by real timestamp.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/algorithm.py` -- vectorised fixed-width batch candidate evaluation and V5 progressive KNN.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/writer.py` -- copied, V5-versioned legacy-compatible streaming writer.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/pipeline.py` -- V5 chronological runner using raw and proxy caches.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/cli.py` -- V5 smoke, benchmark, and explicit full CLI.
- Create: `Data_Imputation/tests/test_temporal_hourly_v5_cache.py` -- cache real-time and tree reuse tests.
- Create: `Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py` -- batch-estimate and progressive KNN V4-equivalence tests.
- Create: `Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py` -- chronological V4/V5 H5 equivalence tests.
- Create: `Data_Imputation/eval_diag/benchmark_temporal_hourly_v5.py` -- real-frame V4/V5 throughput and output comparison tool.

## Task 1: Add a Real-Timestamp Raw and Proxy Tree Cache

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/__init__.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/cache.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v5_cache.py`

- [x] **Step 1: Write failing cache-contract tests.**

```python
def test_raw_cache_returns_only_prior_frames_in_actual_72_hour_window():
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_raw(raw_at("2024-01-01T00:00:00"))
    cache.add_raw(raw_at("2024-01-01T03:00:00"))
    cache.add_raw(raw_at("2023-12-28T00:00:00"))

    history = cache.raw_history(at("2024-01-01T04:00:00"))

    assert [delta for delta, _ in history] == [1, 4]


def test_raw_tree_is_built_once_and_reused_until_timestamp_leaves_cache(monkeypatch):
    calls = 0
    original = cache_module.cKDTree

    def counting_tree(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cache_module, "cKDTree", counting_tree)
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_raw(raw_at("2024-01-01T00:00:00"))
    cache.raw_trees(at("2024-01-01T01:00:00"))
    cache.raw_trees(at("2024-01-01T02:00:00"))

    assert calls == 1
```

- [x] **Step 2: Run the cache tests and verify expected RED failures.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_cache.py`

Expected: `ModuleNotFoundError: No module named 'Data_Imputation.preprocessing.temporal_hourly_v5'`.

- [x] **Step 3: Implement the minimal cache types.**

```python
@dataclass(frozen=True)
class CachedRaw:
    frame: RawFrame
    tree: cKDTree | None
    values: np.ndarray


@dataclass(frozen=True)
class CachedProxy:
    frame: ProxyFrame
    tree: cKDTree | None
    values: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class RawTreeHistory:
    delta_hours: int
    tree: cKDTree
    values: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class ProxyTreeHistory:
    delta_hours: int
    tree: cKDTree
    values: np.ndarray
    confidence: np.ndarray


class RollingCandidateCache:
    def add_raw(self, frame: RawFrame) -> None: """Store one raw frame and its eligible-observation KD-tree."""
    def add_proxy(self, frame: ProxyFrame) -> None: """Store one confidence-filtered synthetic proxy KD-tree."""
    def raw_history(self, target: datetime) -> list[tuple[int, RawFrame]]: """Return raw frames with actual integer lag from one through 72 hours."""
    def raw_trees(self, target: datetime) -> list[RawTreeHistory]: """Return cached raw KD-trees using the same chronological lag order."""
    def proxy_trees(self, target: datetime) -> list[ProxyTreeHistory]: """Return cached eligible proxy KD-trees using the same chronological lag order."""
    def prune(self, newest: datetime) -> None: """Discard cached frames strictly older than the causal time window."""
```

`add_raw` must create coordinates from only finite non-land original observations. `add_proxy` must create coordinates from only finite non-land `TEMPORAL`/`KNN` pixels whose confidence is at least `0.20`. History methods must sort actual timestamps, compute the integer hour delta, and include only `1 <= delta <= 72`; they must never infer a delta from frame position. `prune` removes only entries before `newest - 72h`.

- [x] **Step 4: Run cache tests and V4 source-history regression tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_cache.py Data_Imputation/tests/test_temporal_hourly_v4_source.py`

Expected: all pass.

- [x] **Step 5: Check the scoped diff.**

Run: `git diff --check -- Data_Imputation/preprocessing/temporal_hourly_v5 Data_Imputation/tests/test_temporal_hourly_v5_cache.py`

Expected: no output.

## Task 2: Add a Vectorised Fixed-Width Weighted-Evidence Kernel

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/algorithm.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py`

- [x] **Step 1: Write a failing V4-equivalence test for a batch kernel.**

```python
def test_batch_weighted_estimate_matches_v4_row_by_row_for_invalid_and_tied_candidates():
    values = np.array([[10.0, 12.0, np.nan], [5.0, 8.0, 9.0]])
    confidence = np.array([[1.0, 0.5, 0.8], [0.6, 0.6, 0.0]])
    distance = np.array([[1.0, 1.0, np.inf], [2.0, 2.0, 3.0]])
    delta = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])

    estimates, quality, present = batch_weighted_estimate(
        values, confidence, distance, delta, limit=2
    )

    for row in range(2):
        expected, expected_quality = v4_algorithm._weighted_estimate(
            values[row], confidence[row], distance[row], delta[row], limit=2
        )
        assert present[row] is (expected is not None)
        if expected is not None:
            assert estimates[row] == pytest.approx(expected, abs=1e-6)
            assert quality[row] == pytest.approx(expected_quality, abs=1e-6)
```

- [x] **Step 2: Run the focused test and verify RED.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py -k batch_weighted`

Expected: import failure because `batch_weighted_estimate` is absent.

- [x] **Step 3: Implement a row-independent vectorised kernel.**

```python
def batch_weighted_estimate(
    values: np.ndarray,
    confidence: np.ndarray,
    distance: np.ndarray,
    delta_hours: np.ndarray,
    *,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return estimate, evidence quality, and valid-estimate flag per batch row."""
```

Preserve V4 validity requirements: finite arrays, positive confidence, nonnegative distance/delta, and positive finite candidate weight. Use V4's `candidate_weights` equations. If candidates exceed `limit`, call `np.argpartition(weights, -keep, axis=1)` and gather the selected candidate entries before computing the estimate and evidence terms. Keep no-valid-candidate rows marked false rather than substituting a mean or zero.

- [x] **Step 4: Run the focused test and then all V5 algorithm tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py`

Expected: all pass.

## Task 3: Implement V5 Progressive KNN Without Changing Batch Causality

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v5/algorithm.py`
- Modify: `Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py`

- [x] **Step 1: Write failing end-to-end equivalence tests.**

```python
@pytest.mark.parametrize("rebuild_interval", [1, 2])
def test_v5_progressive_knn_matches_v4_with_raw_and_proxy_history(rebuild_interval):
    inputs = mixed_raw_temporal_proxy_fixture()
    expected = v4_algorithm.progressive_confidence_knn(
        *inputs.v4_args, query_workers=1, rebuild_interval=rebuild_interval
    )
    actual = v5_algorithm.progressive_confidence_knn(
        *inputs.v5_args, query_workers=1, rebuild_interval=rebuild_interval
    )

    np.testing.assert_array_equal(actual.source_code, expected.source_code)
    np.testing.assert_array_equal(actual.knn_fill_mask, expected.knn_fill_mask)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6, rtol=0, equal_nan=True)
    np.testing.assert_allclose(actual.confidence, expected.confidence, atol=1e-6, rtol=0)


def test_v5_excludes_zero_distance_proxy_and_raises_without_any_seed():
    zero_distance = v5_algorithm.progressive_confidence_knn(
        values=np.array([[10.0, np.nan]], dtype=np.float32),
        source_code=np.array([[RAW, UNRESOLVED]], dtype=np.uint8),
        confidence=np.array([[1.0, 0.0]], dtype=np.float32),
        land_mask=np.zeros((1, 2), dtype=bool),
        raw_history=[],
        proxy_history=[(1, proxy_frame({(0, 1): 999.0}, shape=(1, 2)))],
        query_workers=1,
        k=1,
        rebuild_interval=1,
    )
    assert zero_distance.values[0, 1] == pytest.approx(10.0)
    assert zero_distance.values[0, 1] != pytest.approx(999.0)

    with pytest.raises(RuntimeError, match="no valid interpolation seed"):
        v5_algorithm.progressive_confidence_knn(
            values=np.full((2, 2), np.nan, dtype=np.float32),
            source_code=np.full((2, 2), UNRESOLVED, dtype=np.uint8),
            confidence=np.zeros((2, 2), dtype=np.float32),
            land_mask=np.zeros((2, 2), dtype=bool),
            raw_history=[],
            proxy_history=[],
            query_workers=1,
        )
```

- [x] **Step 2: Run equivalence tests and verify RED.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py -k 'progressive or zero_distance'`

Expected: failure because V5 progressive KNN is absent.

- [x] **Step 3: Implement `progressive_confidence_knn`.**

```python
def progressive_confidence_knn(
    values, source_code, confidence, land_mask, raw_history, proxy_history,
    *, query_workers, raw_tree_history=None, proxy_tree_history=None,
    k=20, source_k=8, rebuild_interval=500, density_radius=20.0,
    proxy_minimum_confidence=0.20, target_timestamp=None,
) -> KNNResult:
    """Complete residual ocean pixels with V4-equivalent cached or fresh trees."""
```

Use V4's deterministic density sort. At the start of each batch, rebuild only the current-frame tree; use cached historical trees when supplied or build V4-equivalent trees otherwise. Query every tree once for the whole batch. Assemble base tensors in `current -> raw lag ascending` order, proxy tensors in `proxy lag ascending` order, and apply the vectorised kernel. Retain V4's raw-anchor offset policy, including a zero offset without five anchors inside radius 48. Write each completed batch only after every row has been computed. Preserve V4's exception if no base or proxy candidate exists.

- [x] **Step 4: Run V4 and V5 algorithm suites.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py`

Expected: all pass.

- [x] **Step 5: Add and run an equivalence test for cached versus uncached V5 trees.**

```python
def test_cached_history_trees_equal_freshly_built_history_trees():
    target = datetime(2024, 1, 1, 2)
    raw_history = [(1, raw_frame({(0, 0): 10.0}, timestamp=datetime(2024, 1, 1, 1)))]
    proxy_history = [(1, proxy_frame({(0, 2): 14.0}, timestamp=datetime(2024, 1, 1, 1)))]
    cache = RollingCandidateCache(max_history_hours=72)
    cache.add_raw(raw_history[0][1])
    cache.add_proxy(proxy_history[0][1])
    kwargs = dict(
        values=np.array([[np.nan, np.nan, np.nan]], dtype=np.float32),
        source_code=np.full((1, 3), UNRESOLVED, dtype=np.uint8),
        confidence=np.zeros((1, 3), dtype=np.float32),
        land_mask=np.zeros((1, 3), dtype=bool),
        raw_history=raw_history,
        proxy_history=proxy_history,
        query_workers=1,
        rebuild_interval=1,
        target_timestamp=target,
    )
    expected = v5_algorithm.progressive_confidence_knn(
        **kwargs, raw_tree_history=None, proxy_tree_history=None
    )
    actual = v5_algorithm.progressive_confidence_knn(
        **kwargs,
        raw_tree_history=cache.raw_trees(target),
        proxy_tree_history=cache.proxy_trees(target),
    )
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(actual.source_code, expected.source_code)
```

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py`

Expected: all pass.

## Task 4: Add V5 Chronological Pipeline and Versioned Writer

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/writer.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/pipeline.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v5/cli.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py`

- [ ] **Step 1: Write a failing pipeline equivalence test.**

```python
def test_v5_series_output_matches_v4_for_three_real_timestamps(tmp_path, hourly_fixture):
    v4_root = tmp_path / "v4"
    v5_root = tmp_path / "v5"
    run_v4(hourly_fixture, v4_root, series_id=0, stop_after=at("2024-01-01T02:00:00"))
    run_v5(hourly_fixture, v5_root, series_id=0, stop_after=at("2024-01-01T02:00:00"))

    assert_h5_fields_equal(v4_root, v5_root, fields=(
        "sst_data", "original_obs_mask", "temporal_fill_mask", "original_missing_mask",
        "knn_fill_mask", "unresolved_mask", "source_code", "proxy_confidence", "timestamps",
    ), float_atol=1e-6)
```

- [ ] **Step 2: Run pipeline test and verify RED.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py`

Expected: import failure because the V5 pipeline is absent.

- [ ] **Step 3: Implement writer and pipeline.**

Copy only the V4 writer's legacy H5 schema into V5 and set
`preprocess_version = "temporal_hourly_v5_equivalent_acceleration"`. In the
pipeline, construct `SourceIndex` from V4's reader, iterate its sorted
`available_times`, fetch the current raw frame, obtain all raw history from
`RollingCandidateCache`, run V4-equivalent temporal/Gaussian stages, run V5 KNN
with cached trees, write the completed frame, then add current raw/proxy frames
to the respective caches and prune both. Do not call `SourceIndex.raw_history`
inside V5's run loop.

The CLI must preserve V4's `--series`, `--all-series`, `--smoke-until`,
`--full`, `--query-workers`, and `--series-workers` guards. It must refuse a
non-empty output root.

- [ ] **Step 4: Run pipeline and schema tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py Data_Imputation/tests/test_temporal_hourly_v4_writer.py`

Expected: all pass.

- [ ] **Step 5: Check V5 package syntax and diff.**

Run: `python -m compileall -q Data_Imputation/preprocessing/temporal_hourly_v5 && git diff --check -- Data_Imputation/preprocessing/temporal_hourly_v5 Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py`

Expected: no output and exit code 0.

## Task 5: Add Real-Frame Benchmark and Gate Production Generation

**Files:**
- Create: `Data_Imputation/eval_diag/benchmark_temporal_hourly_v5.py`
- Modify: `Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py`

- [ ] **Step 1: Write a failing benchmark-comparison test.**

```python
def test_compare_results_rejects_mask_difference_and_accepts_small_float_rounding():
    left = synthetic_completed_frame(value=20.0)
    right = synthetic_completed_frame(value=20.0 + 5e-7)
    report = compare_completed_frames(left, right, atol=1e-6)
    assert report.passes

    right.source_code[0, 0] = KNN
    assert not compare_completed_frames(left, right, atol=1e-6).passes
```

- [ ] **Step 2: Run the test and verify RED.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py -k compare_results`

Expected: import failure because `compare_completed_frames` is absent.

- [ ] **Step 3: Implement comparison and benchmark CLI.**

```python
def compare_completed_frames(left: KNNResult, right: KNNResult, *, atol: float = 1e-6) -> ComparisonReport:
    """Check masks exactly and SST/confidence numerical differences explicitly."""

def benchmark_series(source_root: Path, *, series_id: int, until: datetime, query_workers: int) -> BenchmarkReport:
    """Run V4 and V5 into separate temporary roots and compare each written field."""
```

The CLI must print frame count, V4 seconds, V5 seconds, speedup, SST maximum
absolute error, confidence maximum absolute error, mismatched source/mask pixel
counts, unresolved-ocean counts, and pass/fail. It must exit nonzero if source
codes/fill masks differ, any ocean unresolved count is nonzero, an SST/
confidence difference exceeds `1e-6`, or V5 does not improve elapsed time on a
ten-frame benchmark.

- [ ] **Step 4: Run all V5 tests and full unit regression.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v5_cache.py Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py Data_Imputation/tests/test_temporal_hourly_v4_source.py Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py Data_Imputation/tests/test_temporal_hourly_v4_writer.py Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py Data_Imputation/tests/test_temporal_hourly_v4_diagnostics.py`

Expected: all pass.

- [ ] **Step 5: Perform the real-data ten-frame benchmark without touching final output.**

Run: `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m Data_Imputation.eval_diag.benchmark_temporal_hourly_v5 --source-root Data_Imputation/experiments/hourly_data --series 8 --until 2024-07-04T09:00:00 --query-workers 21 --work-root /tmp/temporal_hourly_v5_benchmark`

Expected: source code and fill-mask mismatch counts are zero; SST and confidence
max errors are at most `1e-6`; both unresolved-ocean counts are zero; V5 elapsed
time is lower than V4. Keep benchmark outputs under `/tmp`, never final `/data`.

- [ ] **Step 6: Check all created-file diffs.**

Run: `git diff --check -- Data_Imputation/preprocessing/temporal_hourly_v5 Data_Imputation/tests/test_temporal_hourly_v5_cache.py Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py Data_Imputation/eval_diag/benchmark_temporal_hourly_v5.py docs/superpowers/plans/2026-08-24-temporal-hourly-v5-equivalent-acceleration.md`

Expected: no output.

## Task 6: Production Decision, Cleanup, and Relaunch

**Files:**
- No code changes.

- [ ] **Step 1: Record benchmark evidence in the handoff message.**

Report exact V4/V5 times, speedup, maximum output differences, all mismatch
counts, and evaluation metrics. Do not claim equivalence without this fresh
evidence.

- [ ] **Step 2: Ask for confirmation before removing incomplete V4 output.**

The exact approved target is only:

```text
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h00
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h01
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h02
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h03
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h04
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h05
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h06
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h07
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h08
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h09
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h10
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h11
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h12
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h13
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h14
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h15
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h16
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h17
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h18
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h19
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h20
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h21
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h22
/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h23
```

Also remove `/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/diagnostics`
only if it was created by the interrupted V4 run. Do not touch any sibling
directory or source H5.

- [ ] **Step 3: After explicit cleanup approval, remove only the confirmed partial V4 directories and verify the root is empty.**

Run: `find /data/sst_data/sst_missing_value_imputation/jaxa_processed_data -mindepth 1 -maxdepth 1 -printf '%p\n'`

Expected before removal: only the 24 `hXX` directories and optional `diagnostics`.

Expected after removal: no output.

- [ ] **Step 4: After explicit production approval, launch V5 full processing.**

Run: `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m Data_Imputation.preprocessing.temporal_hourly_v5.cli --source-root Data_Imputation/experiments/hourly_data --output-root /data/sst_data/sst_missing_value_imputation/jaxa_processed_data --all-series --full --series-workers 9 --query-workers 21`

Expected: every series processes chronological real timestamps with no future
history and writes all 216 V5 H5 outputs plus diagnostics.
