# Proxy Preprocessing Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an isolated, mask-aware JAXA proxy dataset whose synthetic pixels are conservatively smoothed, and make the training/inference path consume it consistently so the model can learn smoother observed–missing interfaces.

**Architecture:** Extend the existing temporal-hourly preprocessing implementation rather than changing FNO-CBAM. Temporal fills use normalized Gaussian convolution (`G(xm)/G(m)`), then causal KNN fills residual gaps. A separate bounded smoothing pass acts only on finite synthetic ocean pixels, with stronger weights in the first two natural-gap layers and a hard per-pixel change bound. Provenance masks and smoothing metadata are retained in H5, and a new dataset root is used for validation before any training run.

**Tech Stack:** Python, NumPy, SciPy `ndimage`, h5py, pytest, existing temporal-hourly v5/v6 pipeline.

---

### Task 1: Add failing unit tests for normalized and bounded smoothing

**Files:**
- Modify: `Data_Imputation/tests/test_boundary_smooth.py`
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v6/boundary_smooth.py`

- [ ] **Step 1: Add tests for normalized Gaussian support handling**

Add a test that constructs a finite 2-D field with a NaN hole and an asymmetric
valid mask, calls the new normalized smoothing helper, and verifies that a
valid estimate is computed from supported neighbors rather than a global mean.
Add a second test that verifies an all-invalid support mask returns no finite
estimate and does not manufacture values.

- [ ] **Step 2: Add tests for synthetic-only bounded smoothing**

Add a test with observed, natural, KNN, and land pixels. Verify observed and
land values are bitwise unchanged, only finite synthetic pixels can change,
the changed mask is a subset of synthetic pixels, and every absolute change is
at most the configured cap. Verify the first natural layer receives the
configured stronger weight than an interior synthetic pixel.

- [ ] **Step 3: Run the focused tests and verify the intended failures**

Run:

```bash
pytest -q Data_Imputation/tests/test_boundary_smooth.py
```

Expected: the new helper/API assertions fail because the normalized and
bounded smoothing functions are not yet implemented; existing boundary tests
must remain passing.

### Task 2: Implement conservative mask-aware smoothing primitives

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v6/boundary_smooth.py`
- Test: `Data_Imputation/tests/test_boundary_smooth.py`

- [ ] **Step 1: Implement normalized Gaussian estimation**

Add a private helper with signature:

```python
def _normalized_gaussian_estimate(
    values: np.ndarray,
    support_mask: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

It must convolve `values * support_mask` and `support_mask` separately,
divide only where support is positive, return a float32 estimate plus a bool
valid mask, and reject non-positive sigma or mismatched shapes.

- [ ] **Step 2: Implement bounded synthetic smoothing**

Add a frame-level function with explicit parameters:

```python
def smooth_synthetic_frame(
    values: np.ndarray,
    observed_mask: np.ndarray,
    natural_mask: np.ndarray,
    source_code: np.ndarray,
    land_mask: np.ndarray,
    *,
    sigma: float = 0.8,
    first_layer_weight: float = 0.25,
    second_layer_weight: float = 0.12,
    interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

Use original observed finite ocean pixels as anchors; never use synthetic
pixels as anchors for this pass. Identify natural layers with the existing
four-connected `boundary_layers` helper. Apply the selected weight only to
finite synthetic ocean pixels, clip the value delta to `max_change_k`, leave
observed/land/unresolved pixels unchanged, and return `(result, changed,
layer_mask)`.

- [ ] **Step 3: Run focused tests and the full preprocessing test subset**

Run:

```bash
pytest -q Data_Imputation/tests/test_boundary_smooth.py Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py
```

Expected: all tests pass; no existing boundary-smoothing behavior regresses.

### Task 3: Add an isolated H5 processor and audit metadata

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v6/boundary_smooth.py`
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v6/cli.py`
- Test: `Data_Imputation/tests/test_boundary_smooth.py`

- [ ] **Step 1: Extend H5 schema validation**

Require `source_code` when processing the new proxy version, validate its
shape, and verify that `source_code == RAW` agrees with finite original
observations while land pixels remain NaN. Keep all source datasets and
attributes copied unchanged except for synthetic `sst_data` values and the
new audit datasets/attributes.

- [ ] **Step 2: Process each H5 atomically**

Add a processor that refuses an existing destination or stale `.part` file,
copies the source to a sibling `.part`, applies `smooth_synthetic_frame` to
each frame, writes `proxy_smooth_mask` and `proxy_smooth_layer`, records the
kernel, weights, cap, and protocol version, flushes, and publishes with
`os.replace` only after successful completion. Preserve original observations,
land, timestamps, masks, source codes, confidence, and unresolved masks.

- [ ] **Step 3: Add CLI parameters and round-trip test**

Expose `--sigma`, `--first-layer-weight`, `--second-layer-weight`,
`--interior-weight`, and `--max-change-k`; add a tiny H5 test that checks all
invariants and verifies an existing destination is rejected.

- [ ] **Step 4: Run focused H5 tests**

Run:

```bash
pytest -q Data_Imputation/tests/test_boundary_smooth.py
```

Expected: all tests pass, including atomic output and invariant checks.

### Task 4: Integrate normalized Gaussian into the chronological generator

**Files:**
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v4/algorithm.py`
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v5/pipeline.py`
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v6/pipeline.py`
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v5/writer.py`
- Modify: `Data_Imputation/preprocessing/temporal_hourly_v6/writer.py`
- Test: `Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py`

- [ ] **Step 1: Replace global-mean convolution with normalized support**

In `gaussian_temporal_fills`, compute the Gaussian numerator and support
separately. Write estimates only where the target pixel is a temporal fill and
the normalized support is positive. Keep residual NaNs, raw observations,
land, source codes, and confidence semantics unchanged.

- [ ] **Step 2: Apply the bounded synthetic pass after KNN**

In the v5/v6 chronological pipeline, call the new frame-level smoother after
KNN completion, update only the frame values, and write the audit masks. Do
not alter KNN provenance or confidence merely because a value was smoothed.

- [ ] **Step 3: Add metadata and regression tests**

Record the normalized-Gaussian and synthetic-smoothing parameters in output
attributes. Add tests for no future-frame use, finite ocean completion,
unchanged observed/land pixels, and exact source/mask shape compatibility.

- [ ] **Step 4: Run the preprocessing test suite**

Run:

```bash
pytest -q Data_Imputation/tests/test_temporal_hourly_v5_algorithm.py Data_Imputation/tests/test_temporal_hourly_v5_pipeline.py Data_Imputation/tests/test_temporal_hourly_v6_algorithm.py Data_Imputation/tests/test_boundary_smooth.py
```

Expected: all tests pass with no unresolved-ocean regression.

### Task 5: Generate and validate a new proxy dataset without retraining

**Files:**
- Create: `Data_Imputation/experiments/hourly_data_proxy_smooth_v1_20260904/`
- Create: `Data_Imputation/experiments/hourly_data_proxy_smooth_v1_20260904/manifest.json`
- Create: `Data_Imputation/experiments/hourly_data_proxy_smooth_v1_20260904/diagnostics/`

- [ ] **Step 1: Run a fixed Series 8 smoke frame**

Generate the target frame and neighboring fixed frames using the existing
chronological source and the new parameters. Save stage arrays/metrics under
the new diagnostics directory; do not write to `hourly_data`,
`jaxa_processed_data`, or any checkpoint directory.

- [ ] **Step 2: Process all training/validation/test series**

Run the full requested proxy generation with strict chronological order and
bounded workers. Abort on unresolved ocean pixels, schema mismatches, or an
existing destination. Keep the exact source timestamps and series split.

- [ ] **Step 3: Validate numerical invariants and seam metrics**

For fixed Series 8 frames 150, 160, 170, 179, and 190, compare old/new:

- observed and land bitwise equality;
- finite ocean fraction and unresolved count;
- synthetic absolute-change p50/p95/p99/max;
- observed–natural/KNN seam gradient p50/p95/p99/p99.9;
- natural-interior gradient p50/p95/p99/p99.9.

Write the complete comparison to `manifest.json`; stop if the change cap or
finite-value invariants fail.

### Task 6: Wire training and inference to the isolated data root

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/config.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/visualize_realcloud_style.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/realcloud_diagnostics.py`
- Test: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests/`

- [ ] **Step 1: Make the proxy root explicit in run configuration**

Add the new isolated data root as an explicit command-line/configuration
argument rather than replacing the old default. Include the resolved root and
preprocessing protocol in run manifests.

- [ ] **Step 2: Ensure real-cloud inference uses the same composition path**

Load the new proxy H5 files and use the existing deployment composition that
preserves observed/land pixels and predicts natural gaps. Do not silently fall
back to the old data root or apply an extra Gaussian pass at inference.

- [ ] **Step 3: Add configuration and inference smoke checks**

Verify a sample loads the new H5, observed/land pixels remain unchanged, the
proxy smoothing metadata are present, and no artificial ground truth is used
for Series 8 real-cloud diagnostics.

### Task 7: Handoff before retraining

- [ ] Record exact parameters, paths, frame counts, invariant results, and
  seam quantiles in the new manifest.
- [ ] Do not start training in this plan. Retraining is a separate decision
  after the user reviews the new preprocessing diagnostics.
- [ ] Keep all previous proxy roots and checkpoints untouched.
