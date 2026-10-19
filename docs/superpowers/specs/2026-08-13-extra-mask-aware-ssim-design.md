# Extra-mask-aware SSIM Design

## Goal

Evaluate local SST structure over the artificial extra-mask, including its
irregular boundaries, without allowing unwithheld observations to inflate SSIM.
This replaces the strict-interior SSIM in the paper; it does not change any
other metric, model, prediction, or training result.

## Selected Approach

Use a mask-aware local SSIM. For every field-mask pair and method, candidate
centres are valid reference-ocean pixels in the artificial extra-mask. A full
`7 x 7` spatial window is placed at each candidate centre. Within that window,
local SSIM statistics use only pixels satisfying all of the following:

- the pixel belongs to the artificial extra-mask;
- the reference pixel is valid ocean and finite;
- the method prediction is finite.

The window is eligible when its centre has a finite prediction, its spatial
extent is inside the array, and at least 25 of its 49 pixels satisfy the mask.
Thus irregular cloud boundaries can contribute, but very sparse windows cannot
produce unstable variance and covariance estimates.

For an eligible centre `p`, let `A_p` be its selected pixels. Compute

```
SSIM_p = ((2 mu_x mu_y + C1) (2 cov_xy + C2))
         / ((mu_x^2 + mu_y^2 + C1) (var_x + var_y + C2))
```

using population variance and covariance over `A_p` (`ddof=0`). The constants
are `C1 = (0.01 L)^2` and `C2 = (0.03 L)^2`. For each field-mask pair, `L` is
the valid-reference-ocean SST `P99 - P1` range and is shared by every method.

## Aggregation And Coverage

The pair-level score is the unweighted mean of all eligible local scores. The
reported method score is the macro-average of defined pair-level scores across
the 657 field-mask pairs. A pair with no eligible centre has undefined SSIM and
is excluded from the macro-average.

Each record must also store:

- `n_ssim_candidates`: extra-mask reference-ocean candidate centres;
- `n_ssim`: eligible centres used in the pair-level SSIM;
- `ssim_coverage = n_ssim / n_ssim_candidates`.

The results report coverage alongside SSIM so that method comparisons cannot
hide low spatial support. Prediction invalidity is method-specific and is
therefore reflected in method-specific coverage.

## Alternatives Considered

1. Whole-image standard SSIM was rejected because unwithheld pixels are often
   copied from the reference and can dominate the score.
2. Strict-interior SSIM (`ssim_v2`) was rejected as the final paper metric
   because it removes irregular mask boundaries and small cloud components.
3. Extra-mask-aware local SSIM was selected because it evaluates irregular
   hole geometry while excluding all unwithheld pixels from local statistics.

## Code And Data Boundaries

Modify only:

- `Data_Imputation/baselines/structural_metrics.py`;
- `Data_Imputation/baselines/eval_all_methods.py`;
- focused SSIM and sampling tests under `Data_Imputation/tests/`;
- the SSIM definition, SSIM values, and SSIM interpretation in
  `_overleaf_review/main.tex`.

Do not modify training, checkpoints, prediction generation, MAE, RMSE, bias,
gradient metrics, baseline settings, or existing cache files. All recalculated
records go under
`Data_Imputation/visualization/paper_figs/cache/ssim_v3/`. The `ssim_v1` and
`ssim_v2` directories remain unchanged.

## Verification

- Unit tests prove that unwithheld pixels never affect local statistics.
- Boundary centres with at least 25 selected pixels contribute.
- Windows below the threshold are excluded.
- Identical selected values yield SSIM 1.
- Hand-computed masked statistics match the implementation.
- Single-level sampling remains identical to the corresponding full run.
- Every method has 657 records with consistent field-mask keys.
- Cache summaries reproduce record-level macro-averages and coverage.
- Python compilation, the full project test suite, `git diff --check`, and the
  LaTeX build all pass before completion is reported.
