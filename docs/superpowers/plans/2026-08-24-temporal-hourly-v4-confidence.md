# Temporal Hourly V4 Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated causal hourly JAXA SST proxy pipeline that writes every hour and every series in the existing H5 layout while allowing only confidence-controlled historical reconstruction structure to propagate.

**Architecture:** `temporal_hourly_v4` has independent raw-source reading, in-memory causal frame state, interpolation algorithms, stream H5 writers, and a CLI. A series advances one real timestamp at a time and retains only a 72-hour rolling `ProxyFrame` cache; all completed frames are written immediately to `/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/h00` through `h23`.

**Tech Stack:** Python 3, NumPy, SciPy `ndimage` and `cKDTree`, h5py, pytest, ProcessPoolExecutor, JSON/CSV, matplotlib.

**Repository constraint:** `.git` is read-only in this checkout and the worktree already contains user changes. Do not stage, revert, or commit anything. Each task uses scoped tests, `git diff --check`, and `git status --short -- <paths>` as its checkpoint.

---

## File Structure

- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/__init__.py` -- public package exports and version constant.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/source.py` -- timestamp-indexed raw H5 reader; never returns legacy filled values as observations.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/state.py` -- immutable `ProxyFrame`, provenance constants, and rolling-cache validation.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/algorithm.py` -- temporal fill/confidence, Gaussian stage, candidate quality, and progressive causal KNN.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/writer.py` -- output-root reservation and streaming compatible H5 writer.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/pipeline.py` -- chronological per-series processing and all-series process orchestration.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/diagnostics.py` -- per-frame CSV/JSON diagnostics and manifest.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/cli.py` -- smoke, benchmark, and full-regeneration commands.
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/README.md` -- input/output contract and commands.
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_source.py` -- raw source and timestamp-gap tests.
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py` -- temporal/confidence and causal KNN unit tests.
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_writer.py` -- output schema/no-overwrite tests.
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py` -- chronological, streaming, and all-hour H5 integration tests.
- Create: `Data_Imputation/visualization/paper_figs/fig_temporal_hourly_v4_4panel.py` -- v4 four-panel diagnostic plot matching the existing SST plot style.
- Create: `Data_Imputation/eval_diag/eval_temporal_hourly_v4_proxy.py` -- masked-pixel proxy evaluation for MAE, Gradient MAE, gradient correlation, and SSIM.

## Task 1: Create the V4 Data Model and Raw Source Reader

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/__init__.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/state.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/source.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_source.py`

- [ ] **Step 1: Write failing source-contract tests.**

```python
def test_read_masks_legacy_filled_values_outside_original_observations(tmp_path):
    # H5 contains 999.0 where original_obs_mask is zero.
    frame = SourceIndex.from_hour_dirs(tmp_path, series_id=0).read(timestamp)
    assert np.isnan(frame.values[0, 0])
    assert not frame.observed[0, 0]

def test_history_uses_real_timestamp_deltas_and_skips_gaps(tmp_path):
    index = SourceIndex.from_hour_dirs(tmp_path, series_id=0)
    assert [delta for delta, _ in index.raw_history(target, 72)] == [1, 4]
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_source.py`

Expected: FAIL because `temporal_hourly_v4` is absent.

- [ ] **Step 3: Add fixed provenance constants and frame types.**

```python
LAND = np.uint8(0)
RAW = np.uint8(1)
TEMPORAL = np.uint8(2)
KNN = np.uint8(3)
UNRESOLVED = np.uint8(4)

@dataclass(frozen=True)
class RawFrame:
    timestamp: datetime
    values: np.ndarray
    observed: np.ndarray
    land_mask: np.ndarray

@dataclass(frozen=True)
class ProxyFrame:
    timestamp: datetime
    values: np.ndarray
    source_code: np.ndarray
    confidence: np.ndarray
    temporal_fill_mask: np.ndarray
    knn_fill_mask: np.ndarray
    land_mask: np.ndarray
```

Implement `SourceIndex.from_hour_dirs`, `read`, `raw_history`, and timestamp decoding from v3's behavior, retaining only finite raw pixels. Build the land-mask union across present `h00`--`h23` files, reject duplicate timestamps and shape mismatches, and expose `available_times` in sorted order.

- [ ] **Step 4: Run the source tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_source.py`

Expected: PASS.

- [ ] **Step 5: Check the scoped diff.**

Run: `git diff --check -- Data_Imputation/preprocessing/temporal_hourly_v4 Data_Imputation/tests/test_temporal_hourly_v4_source.py`

Expected: no output.

## Task 2: Implement Causal Temporal Fill, Gaussian Smoothing, and Confidence

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v4/algorithm.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py`

- [ ] **Step 1: Write failing temporal-stage tests.**

```python
def test_temporal_fill_uses_only_raw_same_location_and_actual_lags():
    result = causal_temporal_fill(target, [(1, raw_10), (3, raw_30)])
    assert result.values[0, 0] == pytest.approx(15.0)
    assert result.source_code[0, 0] == TEMPORAL

def test_temporal_confidence_is_about_point_32_for_one_raw_t_minus_one():
    result = causal_temporal_fill(target, [(1, raw_10)])
    assert result.confidence[0, 0] == pytest.approx(0.321, abs=0.002)

def test_gaussian_changes_only_temporal_pixels_and_never_raw_or_residual_nan():
    assert smoothed[raw_y, raw_x] == original[raw_y, raw_x]
    assert np.isnan(smoothed[residual_y, residual_x])
```

- [ ] **Step 2: Run the temporal tests to verify failure.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py -k 'temporal or gaussian'`

Expected: FAIL because the temporal functions are absent.

- [ ] **Step 3: Implement the temporal result and formulas.**

```python
@dataclass(frozen=True)
class TemporalResult:
    values: np.ndarray
    source_code: np.ndarray
    confidence: np.ndarray
    temporal_fill_mask: np.ndarray
    residual_missing: np.ndarray

def temporal_confidence(weights: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    weight_sum = weights.sum(axis=0)
    n_eff = np.divide(weight_sum ** 2, (weights ** 2).sum(axis=0),
                      out=np.zeros_like(weight_sum), where=weight_sum > 0)
    delta_eff = np.divide((weights * deltas[:, None, None]).sum(axis=0), weight_sum,
                          out=np.zeros_like(weight_sum), where=weight_sum > 0)
    return 0.85 * (1.0 - np.exp(-n_eff / 2.0)) * np.exp(-delta_eff / 24.0)
```

Initialize raw finite pixels with `RAW` and confidence `1.0`; only fill otherwise missing ocean pixels from raw historical values using `1/delta`. Apply Gaussian with `sigma=1.5` only where `source_code == TEMPORAL`, preserving source code and confidence.

- [ ] **Step 4: Run temporal tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py -k 'temporal or gaussian'`

Expected: PASS.

## Task 3: Implement Candidate Quality and Proxy Eligibility

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v4/algorithm.py`
- Modify: `Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py`

- [ ] **Step 1: Add failing evidence-quality tests.**

```python
def test_proxy_candidate_requires_confidence_threshold_and_nonzero_distance():
    eligible = select_proxy_candidates(history, query=(1, 1), min_confidence=0.20)
    assert (1, 1) not in {tuple(point) for point in eligible.coords}
    assert np.all(eligible.confidence >= 0.20)

def test_alpha_decreases_when_raw_evidence_is_strong():
    assert blend_alpha(proxy_quality=0.9, base_quality=0.9) < 0.1
    assert blend_alpha(proxy_quality=0.9, base_quality=0.0) == pytest.approx(0.585)
```

- [ ] **Step 2: Run quality tests to verify failure.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py -k 'proxy_candidate or alpha'`

Expected: FAIL because eligibility and alpha functions are absent.

- [ ] **Step 3: Implement the fixed candidate equations.**

```python
def candidate_weights(confidence, distance, delta_hours):
    return confidence * np.power(distance + 1.0, -2.0) * np.exp(-delta_hours / 24.0)

def evidence_quality(weights, confidence, distance, delta_hours):
    total = weights.sum()
    if total <= 0:
        return 0.0
    c = np.dot(weights, confidence) / total
    d = np.dot(weights, distance) / total
    t = np.dot(weights, delta_hours) / total
    n = total ** 2 / np.dot(weights, weights)
    return float(c * np.exp(-d / 20.0) * np.exp(-t / 24.0) * (1.0 - np.exp(-n / 4.0)))

def blend_alpha(proxy_quality, base_quality):
    return float(0.65 * proxy_quality * (1.0 - base_quality))
```

`select_proxy_candidates` must accept only prior `TEMPORAL` or `KNN` values with confidence at least `0.20`, finite SST, non-land location, and Euclidean distance strictly greater than zero. Raw historical values are handled by the base candidate path.

- [ ] **Step 4: Run quality tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py -k 'proxy_candidate or alpha'`

Expected: PASS.

## Task 4: Implement Progressive Causal KNN with Controlled Proxy Blending

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v4/algorithm.py`
- Modify: `Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py`

- [ ] **Step 1: Add failing progressive-fill tests.**

```python
def test_progressive_knn_completes_connected_ocean_gap_and_decays_confidence():
    result = progressive_confidence_knn(seed_frame, raw_history=[], proxy_history=[], k=1)
    assert np.all(np.isfinite(result.values[~land]))
    assert result.confidence[3, 3] < result.confidence[0, 1]

def test_historical_proxy_can_change_only_residual_missing_pixels():
    result = progressive_confidence_knn(frame, raw_history, proxy_history)
    assert result.values[raw_y, raw_x] == frame.values[raw_y, raw_x]
    assert result.values[temporal_y, temporal_x] == frame.values[temporal_y, temporal_x]

def test_zero_distance_historical_synthetic_value_is_not_copied():
    assert result.values[1, 1] != pytest.approx(999.0)

def test_proxy_offset_is_zero_without_five_local_raw_anchors():
    result = progressive_confidence_knn(frame, raw_history, proxy_history)
    assert result.local_proxy_offset[1, 1] == pytest.approx(0.0)

def test_no_raw_or_proxy_seed_raises_instead_of_writing_mean_or_nan():
    with pytest.raises(RuntimeError, match='no valid interpolation seed'):
        progressive_confidence_knn(empty_frame, [], [])
```

- [ ] **Step 2: Run the KNN tests to verify failure.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py -k 'progressive or historical_proxy or zero_distance or no_raw'`

Expected: FAIL because `progressive_confidence_knn` is absent.

- [ ] **Step 3: Implement batch KNN.**

```python
@dataclass(frozen=True)
class KNNResult:
    values: np.ndarray
    source_code: np.ndarray
    confidence: np.ndarray
    knn_fill_mask: np.ndarray

def progressive_confidence_knn(values, source_code, confidence, land_mask,
                               raw_history, proxy_history, *, query_workers, k=20,
                               source_k=8, rebuild_interval=500,
                               target_timestamp=None) -> KNNResult:
    return _fill_density_sorted_batches(
        values=values, source_code=source_code, confidence=confidence,
        land_mask=land_mask, raw_history=raw_history, proxy_history=proxy_history,
        query_workers=query_workers, k=k, source_k=source_k,
        rebuild_interval=rebuild_interval, target_timestamp=target_timestamp,
    )
```

Use `tree.query(query_coords, k=k, workers=query_workers)` for batched current, raw-history, and proxy-history queries. `_fill_density_sorted_batches` must build immutable raw/proxy trees once per target frame, then rebuild the current-frame tree before each deterministic density-sorted batch. It computes `B`, `P`, `R_B`, `R_P`, `alpha`, and `F` for each residual pixel and writes source code `KNN` plus `min(0.75, (1 - alpha) * R_B + alpha * R_P)`. Current-frame synthetic KNN pixels may be used only by later batches in the same frame. For proxy blending, estimate a local robust offset only when at least five current raw anchors lie within 48 pixels; otherwise use zero offset. If a base candidate set is empty but proxy candidates exist, use the proxy estimate as a seed. If both are empty, raise `RuntimeError` with the timestamp.

- [ ] **Step 4: Run algorithm tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py`

Expected: PASS.

## Task 5: Implement Streaming Compatible H5 Writer

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/writer.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_writer.py`

- [ ] **Step 1: Write failing schema and no-overwrite tests.**

```python
def test_writer_matches_legacy_hour_directory_and_core_dataset_schema(tmp_path):
    reserve_output_root(tmp_path)
    writer = SeriesH5Writer.create(tmp_path, index, series_id=0)
    writer.write(proxy_frame)
    with h5py.File(tmp_path / 'h12' / 'jaxa_knn_filled_00.h5') as handle:
        assert handle['sst_data'].dtype == np.dtype('float32')
        assert handle['timestamps'].dtype == np.dtype('S32')
        assert {'source_code', 'proxy_confidence', 'knn_fill_mask', 'unresolved_mask'} <= set(handle)
        assert handle.attrs['hour_offset'] == 12

def test_writer_refuses_existing_hour_directory(tmp_path):
    (tmp_path / 'h00').mkdir()
    with pytest.raises(FileExistsError):
        reserve_output_root(tmp_path)
```

- [ ] **Step 2: Run writer tests to verify failure.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_writer.py`

Expected: FAIL because the writer module is absent.

- [ ] **Step 3: Implement root reservation and fixed-size writers.**

```python
def reserve_output_root(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to write into non-empty output root: {root}')
    root.mkdir(parents=True, exist_ok=True)
    for hour in range(24):
        (root / f'h{hour:02d}').mkdir(exist_ok=False)

class SeriesH5Writer:
    @classmethod
    def create(cls, root: Path, index: SourceIndex, series_id: int) -> 'SeriesH5Writer':
        timestamps_by_hour = group_timestamps_by_hour(index.available_times)
        return cls(root, index, series_id, timestamps_by_hour)
```

`SeriesH5Writer.__init__` opens one new H5 file per present hour, preallocates every dataset using that hour's exact timestamp count, and records `timestamp -> frame_index`. `write(frame, raw_mask, original_missing_mask)` finds the target H5 from `frame.timestamp.hour`, writes exactly one index across all core and v4 datasets, and raises on a duplicate timestamp. `close()` verifies every preallocated index was written, sets `unresolved_mask` from non-finite ocean pixels, rejects a nonzero unresolved ocean count, flushes handles, and closes all files. Preserve legacy datasets and attributes exactly; append v4 fields with gzip compression, using `float16` only for `proxy_confidence`.

- [ ] **Step 4: Run writer tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_writer.py`

Expected: PASS.

## Task 6: Implement Chronological Per-Series Pipeline and All-Series Orchestration

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/pipeline.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py`

- [ ] **Step 1: Add failing causality and integration tests.**

```python
def test_pipeline_reconstructs_all_present_hours_in_timestamp_order(tmp_path):
    result = run_series(source_root, output_root, series_id=0, query_workers=1)
    assert result.processed_timestamps == sorted(result.processed_timestamps)
    assert (output_root / 'h00' / 'jaxa_knn_filled_00.h5').exists()
    assert (output_root / 'h01' / 'jaxa_knn_filled_00.h5').exists()

def test_pipeline_never_places_low_confidence_knn_in_next_hour_proxy_history(tmp_path):
    diagnostics = run_series(source_root, output_root, series_id=0, query_workers=1)
    assert diagnostics['proxy_history_rejected_low_confidence'] > 0
```

- [ ] **Step 2: Run pipeline tests to verify failure.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py`

Expected: FAIL because `run_series` is absent.

- [ ] **Step 3: Implement per-series chronological processing.**

```python
def run_series(source_root: Path, output_root: Path, *, series_id: int,
               max_history_hours: int = 72, gaussian_sigma: float = 1.5,
               query_workers: int = 21, stop_after: datetime | None = None,
               output_reserved: bool = False) -> SeriesSummary:
    if not output_reserved:
        reserve_output_root(output_root)
    index = SourceIndex.from_hour_dirs(source_root, series_id)
    cache: dict[datetime, ProxyFrame] = {}
    with SeriesH5Writer.create(output_root, index, series_id) as writer:
        for timestamp in index.available_times:
            if stop_after is not None and timestamp > stop_after:
                break
            raw = index.read(timestamp)
            raw_history = index.raw_history(timestamp, max_history_hours)
            proxy_history = rolling_proxy_history(cache, timestamp, max_history_hours)
            temporal = causal_temporal_fill(raw, raw_history)
            final = progressive_confidence_knn(
                temporal.values, temporal.source_code, temporal.confidence,
                raw.land_mask, raw_history, proxy_history,
                query_workers=query_workers, target_timestamp=timestamp,
            )
            writer.write(final, raw.observed, temporal.residual_missing)
            cache[timestamp] = final
            prune_cache(cache, timestamp, max_history_hours)
```

`rolling_proxy_history` must return only strictly prior timestamps, keep `RAW` values regardless of confidence, retain synthetic values only where confidence is at least `0.20`, and never mutate cached frames. A selected smoke endpoint still processes every earlier timestamp from the start of that series.

- [ ] **Step 4: Add all-series parent orchestration.**

```python
def run_all_series(source_root: Path, output_root: Path, *, series_ids=range(9),
                   series_workers: int = 9, query_workers: int = 21) -> list[SeriesSummary]:
    reserve_output_root(output_root)
    with ProcessPoolExecutor(max_workers=min(series_workers, len(series_ids))) as pool:
        work = [(source_root, output_root, sid, query_workers, True) for sid in series_ids]
        return list(pool.map(_run_series_worker, work))
```

Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before the pool is created. Validate `series_workers * query_workers <= 192`; default full execution is `9 * 21 = 189` query workers.

- [ ] **Step 5: Run pipeline tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py`

Expected: PASS.

## Task 7: Add Diagnostics, CLI, and Usage Documentation

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/diagnostics.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/cli.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v4/README.md`
- Modify: `Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py`

- [ ] **Step 1: Add a failing diagnostics/CLI test.**

```python
def test_smoke_command_writes_summary_with_required_audit_fields(tmp_path):
    exit_code = main(['--source-root', str(source), '--output-root', str(output),
                      '--series', '8', '--smoke-until', '2024-07-30T23:00:00'])
    summary = json.loads((output / 'diagnostics' / 'series_08.json').read_text())
    assert {'raw_fraction', 'temporal_fraction', 'knn_fraction',
            'proxy_history_fraction', 'mean_confidence', 'unresolved_count'} <= set(summary['frames'][0])
    assert exit_code == 0
```

- [ ] **Step 2: Run it to verify failure.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py -k smoke_command`

Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement diagnostics and CLI modes.**

```python
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
parser.add_argument('--series', type=int)
parser.add_argument('--all-series', action='store_true')
parser.add_argument('--smoke-until')
parser.add_argument('--query-workers', type=int, default=21)
parser.add_argument('--series-workers', type=int, default=9)
parser.add_argument('--full', action='store_true')
```

Require exactly one of `--series` and `--all-series`. Require `--full` for a complete series or all-series run; `--smoke-until` is allowed only with one series. Write per-frame CSV plus per-series JSON and a root `manifest.json` containing parameters, completed outputs, software versions, and timestamps. README commands must use the direct user-approved output root and explain that the root must be empty.

- [ ] **Step 4: Run diagnostics/CLI tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py -k smoke_command`

Expected: PASS.

## Task 8: Add V4 Visual and Held-Out Proxy Evaluation Tools

**Files:**
- Create: `Data_Imputation/visualization/paper_figs/fig_temporal_hourly_v4_4panel.py`
- Create: `Data_Imputation/eval_diag/eval_temporal_hourly_v4_proxy.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v4_diagnostics.py`

- [ ] **Step 1: Write failing metric/plot contract tests.**

```python
def test_evaluation_masks_metrics_to_synthetically_withheld_raw_ocean_pixels():
    metrics = evaluate_withheld(reference, prediction, withheld_mask, land_mask)
    assert metrics['pixel_count'] == int(withheld_mask.sum())
    assert {'mae', 'gradient_mae', 'gradient_correlation', 'ssim'} <= set(metrics)

def test_four_panel_plot_requires_raw_temporal_gaussian_and_final_fields(tmp_path):
    output = render_four_panel(frame_data, tmp_path / 'frame.png')
    assert output.exists()
```

- [ ] **Step 2: Run diagnostic tests to verify failure.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_diagnostics.py`

Expected: FAIL because v4 evaluation and visualization modules are absent.

- [ ] **Step 3: Implement metric masking and the four-panel renderer.**

```python
valid = withheld_mask & ~land_mask & np.isfinite(reference) & np.isfinite(prediction)
mae = np.abs(prediction[valid] - reference[valid]).mean()
```

Calculate gradients only where the center and both compared neighbors are valid ocean withheld pixels; use the same valid-pair semantics in gradient MAE/correlation. The plot must use the colorbar range and typography of `fig_temporal_hourly_legacy_4panel.py`, with panels ordered raw JAXA, temporal fill, Gaussian result, and final v4 result.

- [ ] **Step 4: Run diagnostic tests.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_diagnostics.py`

Expected: PASS.

## Task 9: Verify, Smoke-Test, Benchmark, and Gate Full Generation

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v4/README.md`
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v4/diagnostics.py`

- [ ] **Step 1: Run the full V4 unit and integration suite.**

Run: `pytest -q Data_Imputation/tests/test_temporal_hourly_v4_source.py Data_Imputation/tests/test_temporal_hourly_v4_algorithm.py Data_Imputation/tests/test_temporal_hourly_v4_writer.py Data_Imputation/tests/test_temporal_hourly_v4_pipeline.py Data_Imputation/tests/test_temporal_hourly_v4_diagnostics.py`

Expected: PASS.

- [ ] **Step 2: Run a real-data smoke reconstruction through the heavy-cloud period.**

Run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m Data_Imputation.preprocessing.temporal_hourly_v4.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root /data/sst_data/sst_missing_value_imputation/jaxa_processed_data_smoke_v4 \
  --series 8 --smoke-until 2024-07-30T23:00:00 --query-workers 21
```

Expected: every produced H5 has zero unresolved ocean pixels; `diagnostics/series_08.json` records per-frame coverage and confidence; output root is separate from the final approved directory.

- [ ] **Step 3: Render and evaluate the smoke output.**

Run the v4 four-panel renderer for frames 24--26 and the held-out evaluator on the same deterministic mask seed used by the script. Inspect v4 versus legacy/v3 for interface gradient jumps, gradient distribution, MAE, Gradient MAE, gradient correlation, and SSIM. Record the decision and measured frame throughput in the smoke summary.

- [ ] **Step 4: Require explicit approval before full data generation.**

Do not run the `--all-series --full` command until the user has reviewed smoke figures/metrics and explicitly approved full output. On approval, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m Data_Imputation.preprocessing.temporal_hourly_v4.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root /data/sst_data/sst_missing_value_imputation/jaxa_processed_data \
  --all-series --full --series-workers 9 --query-workers 21
```

Expected: 216 H5 files at the legacy-compatible direct `h00`--`h23` layout, a manifest, and per-series diagnostics. The command refuses a non-empty output root.

- [ ] **Step 5: Final scoped checks.**

Run:

```bash
git diff --check -- Data_Imputation/preprocessing/temporal_hourly_v4 Data_Imputation/tests/test_temporal_hourly_v4_*.py Data_Imputation/visualization/paper_figs/fig_temporal_hourly_v4_4panel.py Data_Imputation/eval_diag/eval_temporal_hourly_v4_proxy.py
git status --short -- Data_Imputation/preprocessing/temporal_hourly_v4 Data_Imputation/tests/test_temporal_hourly_v4_*.py Data_Imputation/visualization/paper_figs/fig_temporal_hourly_v4_4panel.py Data_Imputation/eval_diag/eval_temporal_hourly_v4_proxy.py
```

Expected: no whitespace errors; only the scoped v4 files appear in addition to pre-existing user changes.
