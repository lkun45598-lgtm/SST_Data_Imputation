# Extra-mask-aware SSIM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace strict-interior SSIM with a vectorized extra-mask-aware local SSIM, recalculate all nine methods into isolated `ssim_v3` caches, and update only SSIM-related paper content.

**Architecture:** `structural_metrics.py` computes masked local population statistics with 2-D convolution and returns the pair score plus eligible/candidate counts. `eval_all_methods.py` records SSIM coverage while preserving all existing prediction and non-SSIM metric paths. Existing predictions are reevaluated method-by-method into new cache files, merged only after field-mask key validation.

**Tech Stack:** Python, NumPy, SciPy `ndimage.convolve`, pytest, LaTeX.

---

### Task 1: Specify Mask-aware Local SSIM In Tests

**Files:**
- Modify: `Data_Imputation/tests/test_masked_ssim.py`

- [ ] Replace strict-interior expectations with tests for a three-value return `(ssim, n_eligible, n_candidates)`.
- [ ] Add a boundary-window test whose centre contributes with 25 masked pixels even though the full window is not in the extra-mask.
- [ ] Add a threshold test showing a 24-pixel masked window is excluded.
- [ ] Add a test proving values outside the extra-mask do not affect the score.
- [ ] Add a hand-computed population-statistics test using:

```python
mx = x.mean()
my = y.mean()
vx = ((x - mx) ** 2).mean()
vy = ((y - my) ** 2).mean()
cov = ((x - mx) * (y - my)).mean()
expected = ((2 * mx * my + c1) * (2 * cov + c2)) / (
    (mx * mx + my * my + c1) * (vx + vy + c2)
)
```

- [ ] Run `pytest -q Data_Imputation/tests/test_masked_ssim.py` and confirm failure because the production function still implements strict-interior SSIM and returns two values.

### Task 2: Implement Vectorized Extra-mask-aware SSIM

**Files:**
- Modify: `Data_Imputation/baselines/structural_metrics.py`
- Test: `Data_Imputation/tests/test_masked_ssim.py`

- [ ] Replace erosion and `skimage.metrics.structural_similarity` with `scipy.ndimage.convolve` over valid selected pixels.
- [ ] Compute local `count`, `sum_x`, `sum_y`, `sum_x2`, `sum_y2`, and `sum_xy`; derive population mean, variance, covariance, and SSIM only where `count >= min_valid`.
- [ ] Restrict candidate centres to finite valid-reference extra-mask pixels; restrict eligible centres to full in-array windows, finite predicted centres, and the count threshold.
- [ ] Return `(mean_ssim, n_eligible, n_candidates)` and validate `win_size`, `min_valid`, shapes, and finite positive `data_range`.
- [ ] Run `pytest -q Data_Imputation/tests/test_masked_ssim.py` and confirm all focused tests pass.

### Task 3: Record Coverage In Evaluation Outputs

**Files:**
- Modify: `Data_Imputation/baselines/eval_all_methods.py`
- Test: `Data_Imputation/tests/test_eval_level_sampling.py`

- [ ] Add `SSIM_MIN_VALID = 25` and call the metric with `min_valid=SSIM_MIN_VALID`.
- [ ] Store `n_ssim_candidates`, `n_ssim`, and `ssim_coverage` in every record:

```python
ssim_coverage = (
    float(n_ssim / n_ssim_candidates)
    if n_ssim_candidates > 0 else np.nan
)
```

- [ ] Update metric documentation and console summary without changing prediction generation or any non-SSIM metric.
- [ ] Run focused tests and Python compilation.

### Task 4: Recalculate Into Isolated V3 Caches

**Files:**
- Create only under: `Data_Imputation/visualization/paper_figs/cache/ssim_v3/`

- [ ] Run smoke evaluations for FNO and OI and inspect score, candidate count, eligible count, and coverage.
- [ ] Run neural methods, OI, KNN, linear, and cubic in isolated output files.
- [ ] Run DINEOF separately for low, mid, and high levels without changing `k=20`, `max_iter=20`, or `tol=5e-4`.
- [ ] Verify each method has 657 records and identical `(level, idx, n)` keys; verify candidate counts match across methods for each pair.
- [ ] Merge validated fragments into `ssim_v3/all_methods_ssim.npz`, storing records, per-level/overall SSIM, coverage summaries, metric definition, and source file names.

### Task 5: Update Only SSIM Paper Content

**Files:**
- Modify: `_overleaf_review/main.tex`

- [ ] Replace strict-interior wording with the approved extra-mask-aware formula, 25-pixel threshold, population statistics, `P99-P1` range, macro-average, and coverage definition.
- [ ] Replace only the SSIM table values and best-value bolding based on `ssim_v3`.
- [ ] Update only SSIM interpretation sentences; retain every MAE, RMSE, bias, and gradient value unchanged.

### Task 6: Verify End To End

**Files:**
- Verify all modified files and `ssim_v3` caches.

- [ ] Run `pytest -q Data_Imputation/tests`.
- [ ] Run `python -m py_compile` for modified Python files.
- [ ] Reload the merged cache and assert 5913 records, nine methods, 657 records per method, summary reproducibility, and coverage bounds `[0, 1]`.
- [ ] Run `git diff --check`.
- [ ] Run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` in `_overleaf_review` and confirm exit code zero.
