# JAXA Gradient-Prior Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, deployment-matched JAXA retraining pipeline that uses supervised artificial gaps plus month-conditioned no-reference gradient priors in genuine cloud gaps.

**Architecture:** A new `gradient_prior_v1` package owns dataset construction, monthly statistics, losses, metrics, and a single-GPU trainer. It imports the existing FNO-CBAM model read-only, writes every artifact under a collision-protected experiment directory, and never applies supervised target losses to naturally missing pixels.

**Tech Stack:** Python 3, PyTorch, NumPy, h5py, pytest, RTX 4090/CUDA.

---

## File Map

- Create `Data_Imputation/retraining/gradient_prior_v1/__init__.py`: package marker and protocol version.
- Create `Data_Imputation/retraining/gradient_prior_v1/config.py`: split constants, paths, dataclass configuration, and output collision guard.
- Create `Data_Imputation/retraining/gradient_prior_v1/dataset.py`: deployment-matched JAXA samples.
- Create `Data_Imputation/retraining/gradient_prior_v1/gradient_stats.py`: strict monthly observed-gradient statistics CLI.
- Create `Data_Imputation/retraining/gradient_prior_v1/losses.py`: supervised and no-reference loss masks and reductions.
- Create `Data_Imputation/retraining/gradient_prior_v1/metrics.py`: validation and genuine-gap diagnostics.
- Create `Data_Imputation/retraining/gradient_prior_v1/train.py`: isolated single-GPU pilot/full trainer.
- Create `Data_Imputation/retraining/gradient_prior_v1/README.md`: exact protocol and commands.
- Create tests under `Data_Imputation/retraining/gradient_prior_v1/tests/`.
- Create runtime artifacts only under `Data_Imputation/experiments/gradient_prior_v1/`.

Existing training, dataset, model, cache, figure, and paper files remain unchanged.

### Task 1: Package Configuration And Collision Guard

**Files:**
- Create: `Data_Imputation/retraining/gradient_prior_v1/__init__.py`
- Create: `Data_Imputation/retraining/gradient_prior_v1/config.py`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_config.py`

- [ ] Write failing tests that assert split constants are `(0..6)`, `(7,)`, `(8,)`, split overlap is rejected, and a non-empty output directory is rejected.
- [ ] Run `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_config.py` and confirm import failure.
- [ ] Implement `RunConfig.validate()` and `prepare_new_output_dir()`; the latter may create a missing directory or accept an empty directory but must raise `FileExistsError` for a non-empty directory.
- [ ] Store protocol version, loss weights, mask ratio, seeds, checkpoint, statistics path, and sample limits in the dataclass.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Deployment-Matched Dataset

**Files:**
- Create: `Data_Imputation/retraining/gradient_prior_v1/dataset.py`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_dataset.py`

- [ ] Build a tiny synthetic H5 fixture with observed pixels, natural gaps, land, timestamps, and two time frames.
- [ ] Write failing assertions for:

```python
assert not np.any(sample["natural_mask"] & sample["artificial_mask"])
assert np.all(sample["supervised_mask"] <= sample["original_obs_mask"])
assert np.all(sample["input_sst_seq"][-1][sample["reconstruction_mask"]] == 0.0)
assert np.array_equal(
    sample["mask_seq"][-1].astype(bool),
    sample["natural_mask"] | sample["artificial_mask"],
)
```

- [ ] Implement a deterministic per-`(seed, epoch, index)` real-cloud mask draw, H5 metadata mapping, lazy per-worker H5 handles, 30-frame within-series padding, and normalized target/proxy returns.
- [ ] Return `natural_mask`, `artificial_mask`, `reconstruction_mask`, `supervised_mask`, `original_obs_mask`, `land_mask`, `month`, `series_id`, and local/global indices explicitly.
- [ ] Confirm the target proxy is hidden from both natural and artificial target inputs while the previous 29 causal proxy frames remain unchanged.
- [ ] Re-run the focused tests.

### Task 3: Strict Monthly Gradient Statistics

**Files:**
- Create: `Data_Imputation/retraining/gradient_prior_v1/gradient_stats.py`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_gradient_stats.py`

- [ ] Write failing tests for a hand-computed central gradient, exclusion when one axial neighbor is missing, month aggregation, deterministic histogram quantiles, and rejection of Series 7 or 8.
- [ ] Implement strict central-gradient extraction using center plus four valid observed-ocean neighbors.
- [ ] Accumulate deterministic `0.0001 K/pixel` histograms over `[0, 8]`, sample counts, overflow counts, and P50/P90/P95/P99/P99.5/P99.9 by month.
- [ ] Save canonical JSON containing input identities and SHA-256; implement `load_gradient_stats()` that validates all 12 months, units, training split, and digest.
- [ ] Run focused tests and then generate:

```bash
python Data_Imputation/retraining/gradient_prior_v1/gradient_stats.py \
  --data-dir Data_Imputation/experiments/hourly_data/h12 \
  --series 0 1 2 3 4 5 6 \
  --output Data_Imputation/experiments/gradient_prior_v1/stats/h12_train_monthly.json
```

### Task 4: Mask-Strict Hybrid Loss

**Files:**
- Create: `Data_Imputation/retraining/gradient_prior_v1/losses.py`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_losses.py`

- [ ] Write failing tests showing that changing a natural-gap target value cannot alter MSE, supervised gradient, or supervised boundary losses.
- [ ] Write exact mask-location tests for natural interior edges and observed-natural seam edges.
- [ ] Write tests that high/seam penalties are zero below the monthly threshold and positive above it.
- [ ] Write low-energy tests showing zero penalty when prediction low-pass energy reaches `beta * proxy energy`, positive penalty for an over-flat field, and no penalty from high-frequency noise that disappears under the low-pass definition.
- [ ] Test empty masks for finite differentiable zero and successful `.backward()`.
- [ ] Implement `output_composition`, adjacent-pair mask builders, normalized supervised terms, K/pixel no-reference terms, land-aware box filtering, and `hybrid_loss()` returning named components and valid counts.
- [ ] Run focused tests and inspect all component scales on a synthetic batch.

### Task 5: Validation And Genuine-Gap Diagnostics

**Files:**
- Create: `Data_Imputation/retraining/gradient_prior_v1/metrics.py`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_metrics.py`

- [ ] Write tests for masked MAE/RMSE, strict Gradient MAE, boundary MAE, natural-interior exceedance rate, seam exceedance rate, and local predicted-outlier rate.
- [ ] Implement numerator/count accumulators so epoch metrics are pixel/edge weighted instead of unweighted batch means.
- [ ] Keep SSIM outside checkpoint selection and label all genuine-gap measures as no-reference diagnostics.
- [ ] Run focused tests.

### Task 6: Single-GPU Trainer And Reproducibility

**Files:**
- Create: `Data_Imputation/retraining/gradient_prior_v1/train.py`
- Create: `Data_Imputation/retraining/gradient_prior_v1/README.md`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_train.py`

- [ ] Write failing tests for resolved configuration serialization, training/validation dataset splits, checkpoint metadata, non-finite loss rejection, and output collision handling.
- [ ] Implement CLI arguments for hour, experiment name, epochs, batch size, workers, device, checkpoint, statistics, train/validation sample limits, and all loss weights.
- [ ] Load the existing FNO-CBAM checkpoint with memory mapping, require compatible model keys, initialize a fresh AdamW optimizer, clip gradients, and reject non-finite components.
- [ ] Save `resolved_config.json`, `source_manifest.json`, per-epoch metrics JSONL, each epoch checkpoint, and a best-by-validation-MAE checkpoint in the new output directory.
- [ ] Set the dataset epoch before each training epoch so artificial masks vary deterministically.
- [ ] Document that raw inference has no Gaussian or hard gradient clipping.
- [ ] Run all CPU tests and Python compilation.

### Task 7: One-Batch GPU Smoke Test

**Files:**
- Runtime output: `Data_Imputation/experiments/gradient_prior_v1/h12_smoke_<timestamp>/`

- [ ] Confirm the local GPU identity and free memory.
- [ ] Run one train batch and one validation batch with `batch_size=1`, `workers=0`, and all three no-reference terms enabled.
- [ ] Confirm finite forward loss, backward gradients, optimizer step, peak GPU memory, named component values, and checkpoint write/read.
- [ ] Inspect component scale; change pilot weights only if a prior dominates the supervised objective at initialization, and record any change in the resolved config and design deviation log.

### Task 8: Isolated Five-Epoch H12 Pilot

**Files:**
- Runtime output: a new timestamped directory under `Data_Imputation/experiments/gradient_prior_v1/`

- [ ] Start with a deterministic bounded pilot subset so execution and loss behavior can be checked before committing to a full-data run.
- [ ] Run five epochs on the RTX 4090 without Gaussian post-processing.
- [ ] Compare validation MAE, RMSE, strict Gradient MAE, boundary MAE, natural-interior exceedance, seam exceedance, local-outlier rate, and low-pass energy ratio against the frozen current H12 model on identical Series 7 samples.
- [ ] Render fixed raw-output comparisons only into the new experiment directory; visually distinguish preserved observations from reconstructed pixels.
- [ ] Decide pass/fail against the specification. Do not start the 60-epoch run unless the pilot passes the supervised quality gates and shows no new ringing.

### Task 9: Final Verification

**Files:**
- Verify all new package, test, statistics, and pilot files.

- [ ] Run `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests`.
- [ ] Run `python -m py_compile Data_Imputation/retraining/gradient_prior_v1/*.py`.
- [ ] Run `git diff --check` for new source and documentation.
- [ ] Assert existing tracked training/dataset/model files have no new diff attributable to this implementation.
- [ ] Verify every output path is under `Data_Imputation/experiments/gradient_prior_v1/` and no pre-existing checkpoint/cache/figure timestamp changed.
