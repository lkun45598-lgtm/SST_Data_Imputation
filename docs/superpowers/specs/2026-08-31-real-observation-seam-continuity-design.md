# Real-Observation Seam Continuity Loss

## Goal

Reduce visible gradient discontinuities where the FNO output inside a natural
cloud gap meets preserved original JAXA observations, without globally
smoothing the SST field or weakening genuine fronts.

## Scope

The change is isolated to the `gradient_prior_v2_natural_gap` retraining
package. Existing checkpoints, inference outputs, caches, paper files, and
the existing loss terms remain unchanged. The new experiment must use a new
output directory and an explicitly recorded loss weight.

## Boundary definition

For each four-connected edge from a natural-gap pixel `p` to a preserved ocean
pixel `q`, the term is eligible only when `q` and the next outward pixel `r`
are both original JAXA observations in the same frame. Land, natural missing
pixels, artificial masks, temporal fills, and KNN fills are excluded from the
reference mask.

## Loss

The loss matches the FNO-to-observation normal difference with the observed
normal difference immediately outside the boundary:

`(prediction[p] - observation[q]) - (observation[q] - observation[r])`.

The same rule is evaluated in all four grid directions. The residual is
converted to Kelvin and passed through an L1 penalty. The observation values
are detached from the computation graph. An empty eligible set returns a
differentiable zero and count zero.

This is a local C1-style continuation constraint. It does not force the SST
values on both sides to be equal, does not apply TV to the cloud interior, and
does not use proxy values as pseudo-ground truth.

## Integration

Add one component and one non-negative configuration weight to `losses.py`,
`config.py`, and `train.py`. Include the component and eligible count in epoch
metrics and run manifests. Keep the existing seam upper-bound and low-pass
proxy continuity terms; the new term complements them by using only direct
observations.

## Verification

Unit tests will cover: zero loss for a field with matching continuation slope,
positive loss for a slope mismatch, exclusion of proxy/non-observed exterior
pixels, correct four-direction counts, and differentiable zero for no eligible
edges. Existing package tests and Python compilation must remain green.

## Risks

An overly large weight could flatten real fronts or bias the model toward the
observed-side slope. The default weight must therefore be small relative to
the supervised MSE, recorded in the experiment manifest, and evaluated in a
new isolated training run before replacing any model.
