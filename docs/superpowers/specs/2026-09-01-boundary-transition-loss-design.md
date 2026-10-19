# Boundary Transition Constraint for Natural JAXA Cloud Gaps

## Objective

Reduce the temperature and gradient discontinuity where the model output in a
natural cloud gap meets the preserved causal proxy field, without treating a
proxy value inside a natural gap as ground truth.

## Diagnosis

The current implementation combines nine loss terms. Only the artificial-mask
MSE, gradient, and boundary terms have pointwise SST truth. The natural-gap
continuity term requires two consecutive observed exterior pixels and therefore
has very limited coverage on irregular real clouds. The natural seam term is
only an upper-bound penalty; it does not require the two sides of a seam to
continue the same local slope. The remaining proxy terms are weak priors and
can transfer temporal/KNN artifacts into the reconstruction.

The deployed output also hard-composes raw FNO predictions inside the
reconstruction mask with the unchanged proxy outside it. A loss change alone
cannot guarantee continuity at this hard interface.

## Training design

Keep the reliable supervised objective and replace the collection of
overlapping seam/proxy terms with one boundary-transition objective:

```text
L = L_MSE(artificial) + 0.2 L_gradient(artificial)
    + 0.05 L_boundary_transition(artificial)
    + 0.02 L_natural_jump_guard
```

`L_boundary_transition` is evaluated on one-pixel artificial-mask boundary
edges that have valid observed-ocean neighbors. It contains:

1. a robust value term for the masked boundary pixel against the true target;
2. a robust cross-edge gradient term against the true target gradient.

Both terms use only artificial pixels whose target is an original observation;
natural pixels, KNN values, and temporal-fill values are never used as
pointwise targets. The existing interior gradient supervision is retained.

`L_natural_jump_guard` is a weak hinge penalty on natural-to-preserved ocean
edges whose absolute jump exceeds the monthly observed-gradient upper bound.
It is a safety guard, not a continuity target. The proxy low-energy,
coarse-proxy, proxy-continuity, and strict two-pixel observation-continuity
terms are disabled for this experiment.

## Inference composition

Keep all original observed pixels unchanged. Inside a natural cloud gap, use
the FNO prediction by default. For a first-layer cloud pixel adjacent to
preserved ocean, evaluate the candidate equal proxy/FNO blend against every
cross-seam edge: it is accepted only if no such jump becomes larger and at
least one becomes smaller. Otherwise retain the raw FNO prediction. This local
guarded transition is applied only inside the natural reconstruction mask; it
is not a global Gaussian filter and does not modify land or observed pixels.

## Scope

Planned implementation files:

- `Data_Imputation/retraining/gradient_prior_v2_natural_gap/losses.py`
- `Data_Imputation/retraining/gradient_prior_v2_natural_gap/config.py`
- `Data_Imputation/retraining/gradient_prior_v2_natural_gap/train.py`
- the deployment-matched inference composition used by the real-cloud
  visualisation

The FNO-CBAM architecture, OSTIA pretraining checkpoint, source data, and old
experiment directories remain unchanged. A new experiment directory is used.

## Verification

Use focused checks only:

- loss-mask tests confirm no natural/proxy pixel enters supervised terms;
- one-GPU smoke run confirms finite losses and unchanged observed/land pixels;
- fixed Series 8 frames compare seam jump quantiles and boundary outlier rates
  before and after the change;
- standard and large artificial-mask validation ensure pointwise MAE does not
  materially regress.

## Acceptance criteria

- observed and land pixels are bitwise preserved;
- no NaN or inf is introduced;
- p95/p99 natural seam jumps and local seam outlier rate decrease on the fixed
  real-cloud diagnostic set;
- artificial-mask validation MAE remains within 2% of the preceding run.
