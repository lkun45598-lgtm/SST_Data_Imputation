# 因果逐时序列时空 KNN v3 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改旧预处理代码、旧 H5 和旧 checkpoint 的前提下，实现“过去 72 小时同位置时序填充 + 原 Gaussian 平滑 + 时空/2D 渐进式 KNN”的独立 JAXA 预处理版本。

**Architecture:** 新建 `Data_Imputation/preprocessing/temporal_hourly_v3/` 包。`source.py` 负责从 `h00`--`h23` H5 建立真实时间戳索引并只读取 `original_obs_mask` 对应的原始观测；`algorithm.py` 负责时序填充、Gaussian 和渐进式 KNN；`pipeline.py` 负责按 series 处理、写 H5 和输出诊断统计；`cli.py` 提供 smoke/full 命令。所有结果写入新的 `hourly_data_v3_temporal72` 目录。

**Tech Stack:** Python 3、NumPy、h5py、SciPy `ndimage`、SciPy `cKDTree`、pytest、tqdm。

---

### Task 1: 建立真实逐时时间索引和原始观测读取器

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v3/__init__.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v3/source.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v3_source.py`

- [ ] **Step 1: Write the failing tests**

在测试中创建最小 H5 fixture，包含 `sst_data`、`original_obs_mask`、`land_mask` 和 `timestamps`，并验证以下接口：

```python
def test_index_uses_timestamps_not_array_positions(tmp_path):
    write_fixture(tmp_path / "h00" / "jaxa_knn_filled_00.h5", [
        ("2024-01-01T00:00:00", 10.0, True),
        ("2024-01-01T03:00:00", 30.0, True),
    ])
    index = SourceIndex.from_hour_dirs(tmp_path, series_id=0)
    assert index.available_times == (
        datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 3)
    )
    assert index.lookup(datetime(2024, 1, 1, 1)) is None
    assert index.lookup(datetime(2024, 1, 1, 3)).timestamp.hour == 3


def test_only_original_observations_are_returned(tmp_path):
    # sst_data contains a finite synthetic value, but original_obs_mask is 0.
    write_fixture(tmp_path / "h00" / "jaxa_knn_filled_00.h5", [
        ("2024-01-01T00:00:00", 99.0, False),
    ])
    frame = SourceIndex.from_hour_dirs(tmp_path, series_id=0).read(
        datetime(2024, 1, 1, 0)
    )
    assert np.isnan(frame.values[0, 0])
    assert not frame.observed[0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_source.py -q`

Expected: FAIL with `ModuleNotFoundError` because the v3 source package does not exist.

- [ ] **Step 3: Implement the source index**

Implement these concrete types and methods in `source.py`:

```python
@dataclass(frozen=True)
class SourceFrame:
    timestamp: datetime
    values: np.ndarray       # NaN outside raw observations
    observed: np.ndarray     # bool, raw observed ocean pixels only
    land_mask: np.ndarray    # bool

class SourceIndex:
    @classmethod
    def from_hour_dirs(cls, root: Path, series_id: int) -> "SourceIndex": ...
    @property
    def available_times(self) -> tuple[datetime, ...]: ...
    def lookup(self, timestamp: datetime) -> SourceLocator | None: ...
    def read(self, timestamp: datetime) -> SourceFrame: ...
    def history(self, target: datetime, max_hours: int) -> list[tuple[int, SourceFrame]]: ...
```

`from_hour_dirs` must scan only `h00` through `h23`, decode byte timestamps, reject duplicate timestamps, validate one spatial shape, and store H5 path plus frame index rather than loading all arrays. Legacy hourly files can differ by a small number of coastal pixels in `land_mask`; the index must use the conservative union of all source land masks and expose that same mask for every `SourceFrame`. `read` must load `sst_data[frame_index]`, `original_obs_mask[frame_index]`, and set every non-original-observation pixel or union-land pixel to NaN. `history` must enumerate actual timestamps `target - timedelta(hours=delta)` for `delta=1..max_hours`; absent timestamps are skipped and the returned delta is the actual integer hour difference.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_source.py -q`

Expected: PASS, including the gap test showing that a missing 01:00 timestamp is not treated as adjacent to 03:00.

- [ ] **Step 5: Commit**

The environment may not permit writing `.git/index`; if commit is unavailable, record the exact commit command for the user and leave only the new files staged/unstaged without touching unrelated worktree changes.

### Task 2: Implement causal temporal filling and the existing Gaussian stage

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v3/algorithm.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v3_algorithm.py`

- [ ] **Step 1: Write failing tests**

Use 3x3 synthetic grids to test the exact behavior:

```python
def test_temporal_fill_uses_same_location_and_inverse_actual_lag():
    target = raw_frame({(1, 1): 0.0})
    history = {
        1: raw_frame({(0, 0): 10.0}),
        3: raw_frame({(0, 0): 30.0}),
    }
    result = causal_temporal_fill(target, history, max_hours=72)
    expected = (10.0 / 1.0 + 30.0 / 3.0) / (1.0 + 1.0 / 3.0)
    assert result.values[1, 1] == pytest.approx(0.0)
    assert result.values[0, 0] == pytest.approx(expected)


def test_temporal_fill_does_not_use_future_or_gap_as_adjacency():
    target = missing_frame()
    history = {3: raw_frame({(0, 0): 30.0})}
    result = causal_temporal_fill(target, history, max_hours=2)
    assert np.isnan(result.values[0, 0])


def test_gaussian_only_writes_temporal_fill_pixels():
    data = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, np.nan], [np.nan, np.nan, np.nan]])
    raw_mask = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    temporal_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    result = gaussian_temporal_fills(data, raw_mask, temporal_mask, sigma=1.5)
    assert result[0, 0] == 1.0
    assert np.isnan(result[0, 2])
    assert result[1, 1] != 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_algorithm.py -q`

Expected: FAIL because `causal_temporal_fill` and `gaussian_temporal_fills` are not implemented.

- [ ] **Step 3: Implement the temporal and Gaussian functions**

Implement:

```python
@dataclass(frozen=True)
class TemporalFillResult:
    values: np.ndarray
    residual_missing: np.ndarray
    temporal_fill_mask: np.ndarray

def causal_temporal_fill(
    target: SourceFrame,
    history: Sequence[tuple[int, SourceFrame]],
    *,
    max_hours: int = 72,
) -> TemporalFillResult: ...

def gaussian_temporal_fills(
    values: np.ndarray,
    residual_missing: np.ndarray,
    temporal_fill_mask: np.ndarray,
    *,
    sigma: float = 1.5,
) -> np.ndarray: ...
```

Temporal filling must accumulate `value / delta` and `1 / delta` over finite raw historical observations at the same grid cell, only write into target missing ocean pixels, and produce `residual_missing` before Gaussian. Gaussian must mirror the existing algorithm: finite raw-plus-temporal values enter the convolution, invalid pixels are replaced only inside the convolution input by `nanmean`, and the filtered result is written only where `temporal_fill_mask` is true. Raw observations and residual NaNs remain unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_algorithm.py -q`

Expected: PASS, proving that a 3-hour observation is not treated as a 1-hour neighbor and Gaussian cannot modify raw observations.

- [ ] **Step 5: Commit**

Record the commit command `git add Data_Imputation/preprocessing/temporal_hourly_v3/algorithm.py Data_Imputation/tests/test_temporal_hourly_v3_algorithm.py && git commit -m "feat: add causal temporal fill and gaussian stage"`; do not stage unrelated files.

### Task 3: Implement progressive spatiotemporal KNN with current-frame completion

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v3/algorithm.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v3_knn.py`

- [ ] **Step 1: Write failing tests**

The tests must distinguish raw observations from synthetic values and verify current-frame propagation:

```python
def test_knn_prefers_current_spatial_seed_when_no_history_exists():
    frame = missing_center_with_neighbors(298.0, 299.0)
    filled, knn_mask = progressive_spatiotemporal_knn(
        frame.values, frame.observed, frame.land_mask, history=[], k=2
    )
    assert np.isfinite(filled[1, 1])
    assert knn_mask[1, 1]


def test_knn_uses_actual_hour_lag_for_history():
    target = missing_center_frame()
    history = [(1, raw_frame({(1, 0): 300.0})), (24, raw_frame({(1, 2): 280.0}))]
    filled, _ = progressive_spatiotemporal_knn(
        target.values, target.observed, target.land_mask, history=history, k=2
    )
    assert filled[1, 1] > 290.0


def test_knn_never_uses_historical_synthetic_values():
    target = missing_center_frame()
    synthetic_history = SourceFrame(
        timestamp=target.timestamp - timedelta(hours=1),
        values=np.array([[np.nan, 999.0, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
        observed=np.zeros((3, 3), dtype=bool),
        land_mask=np.zeros((3, 3), dtype=bool),
    )
    filled, _ = progressive_spatiotemporal_knn(
        target.values, target.observed, target.land_mask,
        history=[(1, synthetic_history)], k=1
    )
    assert filled[1, 1] != pytest.approx(999.0)


def test_progressive_knn_completes_connected_gap():
    frame = frame_with_large_connected_gap_and_one_seed()
    filled, knn_mask = progressive_spatiotemporal_knn(
        frame.values, frame.observed, frame.land_mask, history=[], k=1
    )
    assert np.all(np.isfinite(filled[~frame.land_mask]))
    assert knn_mask.sum() == int((~frame.observed & ~frame.land_mask).sum())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_knn.py -q`

Expected: FAIL because the progressive spatiotemporal KNN function is not implemented.

- [ ] **Step 3: Implement the progressive KNN**

Implement this interface:

```python
def progressive_spatiotemporal_knn(
    values: np.ndarray,
    observed_mask: np.ndarray,
    land_mask: np.ndarray,
    history: Sequence[tuple[int, SourceFrame]],
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
) -> tuple[np.ndarray, np.ndarray]: ...
```

The function must:

- initialize current seeds from finite `values` on ocean pixels; raw observations receive reliability 1, Stage-1 temporal pixels receive `temporal_reliability`;
- build a fixed history candidate list only from `SourceFrame.observed` pixels and attach actual hour lag;
- sort residual missing pixels by the number of missing neighbors within `density_radius`, ascending;
- query up to `source_k` spatial neighbors per history frame and up to `k` current-frame neighbors;
- combine candidates using `reliability * (distance_space + eps)^(-power_space) * (1 + delta_hours)^(-power_time)`;
- choose the `k` largest combined-weight candidates and use their normalized weighted mean;
- allow newly filled current-frame pixels to participate after each `rebuild_interval` and in the interim local list, so a connected gap completes like the old progressive 2D KNN;
- never append newly filled values to the historical candidate list;
- raise `RuntimeError` with the target timestamp supplied by the caller when a target frame has no valid seed at all, and raise `ValueError` if any ocean NaN remains after the progressive pass.

The implementation must not use the old 3D `time_weight=0.9`, sequence-index time distance, historical filled arrays, or global-mean fallback.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_knn.py -q`

Expected: PASS, including current-frame completion and rejection of a historical synthetic value.

- [ ] **Step 5: Commit**

Record the commit command `git add Data_Imputation/preprocessing/temporal_hourly_v3/algorithm.py Data_Imputation/tests/test_temporal_hourly_v3_knn.py && git commit -m "feat: add causal spatiotemporal progressive knn"`.

### Task 4: Add isolated H5 pipeline and command-line smoke runner

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v3/pipeline.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v3/cli.py`
- Create: `Data_Imputation/tests/test_temporal_hourly_v3_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

The pipeline test must write two tiny source hours and one target hour, run one target frame, and assert the output contains the required datasets and no ocean NaNs:

```python
def test_pipeline_writes_isolated_compatible_h5(tmp_path):
    source_root = make_hourly_fixture(tmp_path / "source")
    output_root = tmp_path / "output"
    run_series(
        source_root=source_root,
        output_root=output_root,
        series_id=0,
        target_hour=12,
        frame_limit=1,
    )
    output_path = output_root / "h12" / "jaxa_knn_filled_00.h5"
    with h5py.File(output_path, "r") as handle:
        assert {"sst_data", "original_obs_mask", "original_missing_mask",
                "temporal_fill_mask", "knn_fill_mask", "unresolved_mask",
                "timestamps"}.issubset(handle.keys())
        sst = handle["sst_data"][:]
        land = handle["land_mask"][:].astype(bool)
        assert not np.any(np.isnan(sst[:, ~land]))
```

- [ ] **Step 2: Run the pipeline test to verify it fails**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_pipeline.py -q`

Expected: FAIL because the isolated pipeline does not exist.

- [ ] **Step 3: Implement H5 writing and CLI**

Implement:

```python
def run_series(
    source_root: Path,
    output_root: Path,
    *,
    series_id: int,
    target_hour: int = 12,
    max_history_hours: int = 72,
    gaussian_sigma: float = 1.5,
    frame_limit: int | None = None,
) -> Path: ...
```

For each target timestamp in the selected target-hour index, read the raw target frame, call `history`, apply temporal fill, Gaussian, and progressive KNN, and save:

- `sst_data`: final filled frame;
- `land_mask`, `latitude`, `longitude`, `timestamps`;
- `original_obs_mask`: raw target observations;
- `temporal_fill_mask`: Stage-1 fills;
- `original_missing_mask`: residual missing after Stage 1 and Gaussian, before KNN;
- `knn_fill_mask`: Stage-2 fills;
- `unresolved_mask`: ocean pixels unresolved after KNN, expected to be all false.

Write attributes `preprocess_version=v3_temporal_hourly`, `target_hour`, `max_history_hours`, `gaussian_sigma`, `knn_k`, `knn_source_k`, `knn_power_space`, `knn_power_time`, and `source_policy=original_obs_mask_only`. Refuse to write into a non-empty output directory. Preserve land NaN values and never overwrite any existing output H5.

The CLI must support:

```bash
python -m Data_Imputation.preprocessing.temporal_hourly_v3.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root Data_Imputation/experiments/hourly_data_v3_temporal72 \
  --series 8 --hour 12 --frame 223 --neighbor-frames 5
```

The smoke mode must process frame 223 plus five frames before and after when available, report coverage of raw/temporal/KNN/unresolved masks, list missing historical timestamps, and write diagnostics under `Data_Imputation/experiments/hourly_data_v3_temporal72/diagnostics/frame223/`. Full mode must require an explicit `--full` flag.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_pipeline.py -q`

Expected: PASS, with output in the temporary isolated directory and no changes to existing experiment files.

- [ ] **Step 5: Commit**

Record the commit command `git add Data_Imputation/preprocessing/temporal_hourly_v3 Data_Imputation/tests/test_temporal_hourly_v3_pipeline.py && git commit -m "feat: add isolated temporal hourly v3 pipeline"`.

### Task 5: Run repository verification and the Series-8 smoke comparison

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v3/README.md`
- Create: `Data_Imputation/experiments/hourly_data_v3_temporal72/diagnostics/frame223/summary.json`
- Create: `Data_Imputation/experiments/hourly_data_v3_temporal72/diagnostics/frame223/overview.png`

- [ ] **Step 1: Run focused tests**

Run: `pytest Data_Imputation/tests/test_temporal_hourly_v3_source.py Data_Imputation/tests/test_temporal_hourly_v3_algorithm.py Data_Imputation/tests/test_temporal_hourly_v3_knn.py Data_Imputation/tests/test_temporal_hourly_v3_pipeline.py -q`

Expected: all focused tests pass.

- [ ] **Step 2: Run smoke preprocessing**

Run:

```bash
python -m Data_Imputation.preprocessing.temporal_hourly_v3.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root Data_Imputation/experiments/hourly_data_v3_temporal72 \
  --series 8 --hour 12 --frame 223 --neighbor-frames 5
```

Expected: the command reports the actual preceding timestamps used, temporal/KNN coverage, no future timestamp use, and a zero unresolved ocean count. It must not modify `Data_Imputation/experiments/hourly_data/h12` or any checkpoint.

- [ ] **Step 3: Inspect smoke artifacts**

Read `summary.json` and inspect `overview.png`. Compare the target frame, old h12 output, v3 output, raw observation boundary, gradient magnitude, and fill masks. Reject the output if raw observations change, if future timestamps appear, if the KNN fill is incomplete, or if new interface jumps are materially worse than the old result.

- [ ] **Step 4: Add README with reproducible commands**

Document source paths, the exact causal rule, Gaussian rule, KNN candidate policy, output masks, smoke command, and the explicit command required for full regeneration. State that model retraining is not part of this preprocessing plan.

- [ ] **Step 5: Run final checks**

Run:

```bash
python -m compileall Data_Imputation/preprocessing/temporal_hourly_v3
pytest Data_Imputation/tests/test_temporal_hourly_v3_*.py -q
git diff --check
```

Expected: compile succeeds, all v3 tests pass, and there are no whitespace errors. Report any pre-existing unrelated test failures separately.

---

## Plan self-review

- **Spec coverage:** The plan covers raw-observation-only source selection, timestamp gaps, strict 72-hour causality, same-location temporal fill, the original Gaussian operation, current-frame progressive completion, non-recursive historical candidates, compatible masks, isolated H5 output, and frame-223 smoke validation.
- **Placeholder scan:** No unresolved placeholder or unspecified implementation step is used. Every task names concrete files, functions, commands, and expected outcomes.
- **Type consistency:** `SourceFrame`, `SourceIndex`, `TemporalFillResult`, `causal_temporal_fill`, `gaussian_temporal_fills`, `progressive_spatiotemporal_knn`, and `run_series` are used consistently across tasks.
- **Scope check:** Only new v3 preprocessing files, focused tests, documentation, and new diagnostics are in scope. Existing preprocessing scripts, H5 files, checkpoints, caches, and training code are explicitly excluded.
