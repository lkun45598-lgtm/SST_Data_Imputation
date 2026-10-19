# Temporal Hourly V6 Parallel Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the interrupted V5 H5 generation in place while preserving V5 numerical results and using persistent cross-tree parallelism across at most 192 CPU threads.

**Architecture:** Add an isolated `temporal_hourly_v6` package. V6 keeps V5's chronological, causal data flow and candidate order, but uses one persistent thread executor per annual-series process to query independent historical KD-trees and evaluate independent rows concurrently. A resumable writer validates the existing H5 schema, identifies only a strict prefix of fully consistent frames, reconstructs the last 72 hours of raw/proxy cache, and overwrites the first incomplete frame before continuing.

**Tech Stack:** Python 3.13, NumPy, SciPy `cKDTree`, h5py, `concurrent.futures`, pytest.

---

### Task 1: Exact parallel interpolation

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v6/__init__.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v6/algorithm.py`
- Test: `Data_Imputation/tests/test_temporal_hourly_v6.py`

- [ ] Add a test that runs the same mixed raw/proxy candidate case through V5 and V6 with four executor threads and requires exact equality for SST, confidence, source codes, masks, and local proxy offsets.
- [ ] Run the test and confirm it fails because V6 does not exist.
- [ ] Implement persistent-executor helpers that submit independent historical-tree queries concurrently, gather results in the original chronological order, and call V4 `_weighted_estimate` independently for each row.
- [ ] Keep every individual KD-tree query single-threaded so the 9 annual processes share at most `9 * 21 = 189` persistent worker threads without nested oversubscription.
- [ ] Run the focused exactness test and require zero mismatches.

### Task 2: Strict-prefix H5 resume

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v6/writer.py`
- Test: `Data_Imputation/tests/test_temporal_hourly_v6.py`

- [ ] Add a test that creates the full output schema, writes a chronological prefix, closes it as interrupted, resumes it, and verifies that only the complete prefix is accepted.
- [ ] Include a deliberately inconsistent partial frame in the test and require V6 to classify it as incomplete and overwrite it.
- [ ] Run the test and confirm it fails because the resumable writer does not exist.
- [ ] Implement schema/timestamp validation and conservative completion checks across `sst_data`, all provenance masks, `source_code`, and `proxy_confidence`.
- [ ] Reject non-prefix completion patterns, expose completed timestamps, and load the last 72 hours as `ProxyFrame` objects for cache restoration.
- [ ] On exceptional exit, flush and close without declaring the incomplete series complete; on normal exit, retain the existing strict no-unwritten/no-ocean-NaN checks.
- [ ] Run the focused resume test.

### Task 3: Resumable 9-series production pipeline

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v6/pipeline.py`
- Create: `Data_Imputation/preprocessing/temporal_hourly_v6/cli.py`
- Test: `Data_Imputation/tests/test_temporal_hourly_v6.py`

- [ ] Add a tiny real-H5 pipeline test comparing uninterrupted V5 output against an interrupted-and-resumed V6 output.
- [ ] Run it first and confirm the resume API is missing.
- [ ] Implement `run_series(..., resume=True)` to restore only the real-time 72-hour cache and continue from the first incomplete timestamp.
- [ ] Implement `run_all_series(..., resume=True)` with 9 processes and 21 persistent threads per process, bounded to 189 threads.
- [ ] Add CLI `--resume` and require an existing non-empty output root in resume mode.
- [ ] Run the focused V6 test file and compile the new package.

### Task 4: Switch the production run safely

**Files:**
- Runtime output only: `/data/sst_data/sst_missing_value_imputation/jaxa_processed_data`
- Runtime log only: `/tmp/jaxa_proxy_v6_20260825.log`

- [ ] Run one real-data equivalence benchmark in `/tmp`, comparing V5 and V6 output fields with zero numerical and provenance mismatches.
- [ ] Confirm the V6 benchmark materially increases CPU utilization or throughput before touching the active run.
- [ ] Send Ctrl-C to the V5 tmux process group so h5py handles flush and close; confirm no V5 process still holds an output file.
- [ ] Audit all nine series for a strict complete prefix and report the recovered frame count before resuming.
- [ ] Launch V6 in a new tmux session with `--all-series --full --resume --series-workers 9 --query-workers 21`.
- [ ] Confirm the V6 processes remain active, the log has no traceback, and output files continue growing.

### Plan Review

- Target is measurable: exact V5 output fields plus higher sustained CPU utilization and successful in-place continuation.
- Modified scope is limited to the new V6 package, one focused test file, and this plan. V4/V5, original H5 inputs, training code, checkpoints, and paper files are excluded.
- V6 imports stable V4/V5 data structures; no existing caller changes.
- Rollback is to stop V6 and retain the flushed partial H5 files; V4/V5 source remains unchanged.
- Verification is deliberately limited to focused unit behavior and one real-data field-for-field benchmark, per the user's request not to add excessive tests.
- Main risks are accepting a partial frame, changing candidate order, and nested thread oversubscription. Conservative frame consistency checks, ordered future collection, and single-threaded tree calls address them.

