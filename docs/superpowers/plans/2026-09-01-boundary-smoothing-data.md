# Boundary Smoothing Data Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an isolated H5 proxy-data version with conservative smoothing only on the first two natural-gap layers adjacent to original observed ocean pixels.

**Architecture:** Add a standalone frame-level postprocessor under the existing temporal-hourly preprocessing package. It copies the source H5 schema, uses normalized local cubic B-spline weights over original observed ocean anchors, writes only natural-gap boundary pixels, records modified layers for audit, and processes series files independently in parallel.

**Tech Stack:** Python 3.10, NumPy, SciPy ndimage, h5py, pytest, ProcessPoolExecutor.

---

### Task 1: Define frame-level boundary smoothing behavior

**Files:**
- Create: `Data_Imputation/preprocessing/temporal_hourly_v6/boundary_smooth.py`
- Test: `Data_Imputation/tests/test_boundary_smooth.py`

- [x] Write tests for layer selection, protected observed/land values, finite output, and no-anchor no-op.
- [x] Run the focused tests and verify they fail because the module does not exist.
- [x] Implement mask-aware cubic-kernel smoothing and conservative layer-weighted fusion.
- [x] Run the focused tests and verify they pass.

### Task 2: Add H5 batch processing and audit metadata

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v6/boundary_smooth.py`
- Test: `Data_Imputation/tests/test_boundary_smooth.py`

- [x] Copy each source H5 without altering its existing datasets or attributes.
- [x] Overwrite only `sst_data` values at selected boundary pixels and add `boundary_smooth_mask` and `boundary_smooth_layer` datasets.
- [x] Add protocol attributes and refuse non-empty output roots.
- [x] Verify a tiny H5 round trip preserves source fields and changes only selected pixels.

### Task 3: Process all H12 series in an isolated remote directory

**Files:**
- Create remotely: `/dev/shm/lz/gradient_boundary_transition_20260901/data/h12_boundary_smooth`
- Create remotely: processing log and manifest under the same experiment root.

- [x] Process all nine latest v5 H12 files with bounded parallel workers, refusing to overwrite existing output.
- [x] Verify frame counts, masks, timestamps, observed/land invariants, finite ocean values, and modification statistics.
- [x] Keep the existing 8-card training task and original H12 data untouched.

### Task 4: Document handoff

- [x] Record the exact output path, kernel, layer rules, and per-series statistics in the output manifest.
- [x] Do not start a new model training run until the processed data are reviewed.
