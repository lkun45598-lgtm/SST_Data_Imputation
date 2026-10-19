# Boundary Transition Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce discontinuities between FNO reconstruction pixels and preserved proxy pixels while retaining only reliable supervised losses.

**Architecture:** Add a focused artificial-boundary transition loss that supervises boundary values and cross-edge gradients using original observed targets. Keep a weak natural seam jump guard for deployment safety, and apply a guarded first-layer proxy/FNO transition only inside natural cloud gaps during inference visualization.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, existing FNO-CBAM retraining package.

---

### Task 1: Add focused boundary-transition loss tests

**Files:**
- Test: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests/test_losses.py`

- [ ] **Step 1: Add a test for boundary value and gradient coverage**

Create a small observed-ocean field with an artificial mask touching a preserved
pixel. Verify the transition loss is zero for a prediction equal to the target
and positive when the masked boundary value or cross-edge gradient changes.

- [ ] **Step 2: Add a test that natural pixels cannot enter the supervised transition loss**

Mark an adjacent natural pixel as non-observed and alter its target value. The
loss and valid count must remain unchanged, proving the loss only uses
artificial pixels with original observed targets.

- [ ] **Step 3: Run the focused tests and confirm the new test fails before implementation**

Run:
`pytest -q Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests/test_losses.py -k boundary_transition`

Expected: FAIL because the new loss function is not yet defined.

### Task 2: Implement simplified loss composition

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/losses.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/config.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/train.py`
- Test: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests/test_config.py`
- Test: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests/test_train.py`

- [ ] **Step 1: Implement `supervised_boundary_transition_loss`**

Use one-pixel artificial-mask boundary edges with both endpoints in the
original observed ocean. Penalize a robust boundary value error on the
artificial endpoint and an L1 cross-edge gradient error. Return a scalar loss
and count, using the existing `_masked_mean` convention.

- [ ] **Step 2: Reduce `HybridLossWeights` and `hybrid_loss`**

Retain `mse`, `gradient`, `boundary_transition`, and the weak natural
`seam` upper-bound guard. Remove the active contributions and configuration
fields for `high`, `seam_continuity`, `observation_seam_continuity`, `low`, and
`coarse_proxy`; leave their helper functions unused for backward-compatible
diagnostic imports.

- [ ] **Step 3: Update configuration and manifest metadata**

Set defaults to `lambda_grad=0.2`, `lambda_boundary_transition=0.05`, and
`lambda_seam=0.02`. Ensure resolved configuration and checkpoint manifests
record the new names and weights.

- [ ] **Step 4: Update train-side metric accumulation**

Report the new `boundary_transition` component and count. Keep existing MAE,
RMSE, gradient MAE, boundary MAE, and seam exceedance metrics so old and new
runs remain comparable.

- [ ] **Step 5: Run all package unit tests**

Run:
`pytest -q Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests`

Expected: PASS with no non-finite-loss or mask-leakage failures.

### Task 3: Add inference boundary transition composition

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/losses.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/visualize_realcloud_style.py`
- Test: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests/test_losses.py`

- [x] **Step 1: Implement `boundary_transition_composition`**

Identify first-layer natural-cloud pixels adjacent to preserved ocean pixels.
Evaluate the 0.5/0.5 proxy/FNO candidate against every adjacent preserved-ocean
edge and accept it only when no edge jump increases and at least one decreases.
Preserve all original observed and land pixels exactly; do not apply Gaussian
filtering or modify artificial-mask training composition.

- [x] **Step 2: Add composition invariants tests**

Verify observed and land pixels are bitwise unchanged, interior natural pixels
use the FNO output, and the first natural boundary pixel is the weighted blend.

- [x] **Step 3: Wire the composition into real-cloud visualization/inference**

Use the transition only for natural-cloud deployment output. Keep the existing
training/evaluation composition for artificial-mask validation so its pointwise
supervision remains well-defined.

### Task 4: Smoke verification and isolated training launch

**Files:**
- Create: a new experiment directory under
  `Data_Imputation/experiments/gradient_prior_v2_natural_gap/`
- Modify: `Data_Imputation/retraining/gradient_prior_v2_natural_gap/README.md`

- [ ] **Step 1: Run syntax and focused smoke checks**

Run `python -m py_compile Data_Imputation/retraining/gradient_prior_v2_natural_gap/*.py`
and a one-batch CPU/GPU forward-loss smoke test with finite output.

- [ ] **Step 2: Run one fixed real-cloud inference comparison**

Render Series 8 frame 179 with the prior and new composition. Record seam
edge p95/p99 jump and verify observed/land preservation.

- [ ] **Step 3: Start a new training run only after smoke checks pass**

Use a fresh experiment name and output directory. Do not overwrite the
existing `h12_natural_gap_v2_obsseam_20260831_081044` artifacts.

- [ ] **Step 4: Document the active loss and composition protocol**

Update the README with the exact active terms, weights, and inference-only
transition rule.
