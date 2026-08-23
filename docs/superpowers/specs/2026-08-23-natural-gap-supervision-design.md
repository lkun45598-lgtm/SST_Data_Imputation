# Natural-Gap Supervision Design

## Objective

Improve large real-cloud SST reconstructions without treating the KNN/temporal
proxy as pixel-level ground truth or changing the deployment output rule. The
network continues to predict a complete field, while the delivered field keeps
the input outside the reconstruction mask and uses FNO-CBAM inside it.

The change addresses the observed failure on Series 8 frame 179: the
natural-gap interior is comparatively smooth, but the FNO-to-preserved-input
interface has large discontinuities. In that frame, only 2,067 of 18,889
natural-to-preserved interface edges connect to original observations. The
existing seam losses therefore covered only 10.9 percent of deployed seam
edges.

## Training Data

Each target frame remains split into three disjoint areas:

1. Original observations: the only source of pixel-level SST targets.
2. Artificial masks sampled within original observations: supervised with
   pointwise, gradient, and boundary losses.
3. Natural cloud gaps: no collocated SST target; used only by weak constraints.

The existing training split contains 2,434 frames from Series 0--6. Of ocean
pixels, 35.7 percent are original observations, 34.1 percent are residual
natural gaps, and 30.2 percent are temporally filled proxy inputs. There are
354 frames with at least 70 percent original-observation coverage. These
frames make large supervised masks feasible without inventing targets.

## Proposed Training Objective

### Large-Mask Supervised Curriculum

Keep standard real-cloud-shape masks at 20--30 percent of the eligible
original observations. Add a large-mask stream sampled from frames with at
least 70 percent original-observation coverage. Its masks use the existing
real-cloud footprint generator, target 40--60 percent of the eligible
observations, and never include natural gaps, land, or temporally filled
pixels.

The training sampler mixes the standard and large-mask streams evenly during
fine-tuning. Both streams use the same trusted original SST target and the
existing MSE, gradient, and supervised-boundary losses. No rectangular mask
generator or pseudo target is introduced.

### Confidence-Gated Coarse Proxy Loss

Add a weak loss only in natural-gap cores. For the composed prediction and the
causal proxy, compute the same land-aware 15-by-15 low-pass field. Select a
natural-gap centre only when its low-pass neighbourhood has at least 80 percent
natural-gap support and the proxy's local high-pass residual is below a
precomputed monthly observed-SST threshold. This rejects locations whose proxy
contains a localized interpolation artifact.

The selected loss is a Huber distance between the two low-pass fields:

    L_coarse = mean Huber(LP15(T_composed) - stopgrad(LP15(T_proxy)))

This supplies a small-weight large-scale temperature anchor in unobserved
regions. It does not compare raw proxy pixels, does not supervise proxy
high-frequency texture, and does not apply a filter to the inference output.
The initial loss weight is 0.02 and is subject to held-out validation.

### Deployment-Seam Continuity

Redefine an eligible natural seam from the actual deployment composition:
natural prediction adjacent to any preserved ocean input, excluding land and
artificially reconstructed pixels. This replaces the current requirement that
the preserved side must be an original observation.

Use the low-pass proxy only to form the outward reference difference, and skip
edges whose raw proxy residual is anomalous. The cross-seam condition then
encourages a smooth continuation of the reliable large-scale exterior field
without forcing FNO to reproduce a raw KNN artifact. The gradient upper-bound
constraint applies to the same full set of deployed seam edges.

## Non-Goals

- Do not use proxy values as full-resolution MSE targets in natural gaps.
- Do not overwrite original observations with raw FNO output at inference.
- Do not use Gaussian output smoothing, gradient clipping, or iterative
  optimization during inference.
- Do not modify old checkpoints, figures, or evaluation caches.

## Validation

The new run must be evaluated separately from existing experiments.

1. Series 7 standard and large artificial real-cloud masks: MAE, RMSE,
   Gradient MAE, boundary MAE, and per-mask-level results.
2. Series 8 no-reference diagnostics: natural-interior and seam adjacent-edge
   difference quantiles, seam exceedance rate, and local outlier rate.
3. Identical Series 8 frame 179 visualization and attribution diagnostic,
   compared with the formal and current seam-continuity checkpoints.
4. Unit tests must demonstrate strict mask separation, no pseudo-loss gradient
   outside natural-gap cores, confidence-gate exclusion, and coverage of all
   deployed natural-to-preserved seam edges.

A model is not accepted merely because its no-reference seam metrics improve:
it must not regress the held-out artificial-mask errors or create visibly
blurred large-scale SST structure.
