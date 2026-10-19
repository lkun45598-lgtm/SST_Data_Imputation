# Mask-aware smoothing for the JAXA proxy field

## Objective

Make the proxy field used during FNO-CBAM training spatially continuous at
the interface between generated values and original observations, while
preserving real observations and land pixels exactly. The proxy is a training
input/conditioning field, not a ground-truth label for natural cloud gaps.

## Data-flow design

The chronological causal order remains unchanged:

1. use only earlier raw observations for temporal filling;
2. apply mask-normalized Gaussian smoothing to temporal fills;
3. fill the remaining ocean gaps with causal spatial KNN/proxy candidates;
4. apply a bounded, mask-aware smoothing pass to synthetic pixels only;
5. write provenance masks and confidence values together with the proxy field.

The smoothing passes never read future frames and never modify an original
observation or land pixel.

## Smoothing rule

For a field `x` and valid support mask `m`, the local estimate is computed as

`G(x*m) / G(m)`

with a small Gaussian kernel (sigma 0.8 by default). This normalized form
prevents NaNs or a global mean from becoming artificial boundary values.

The final synthetic field is a bounded blend with that estimate. Pixels in
the first and second natural-gap layers next to observed ocean use larger
weights (0.25 and 0.12 by default); other synthetic pixels use a weak weight
(0.08). The absolute change per pixel is clipped to 0.10 K by default. The
observed-ocean and land masks are copied bitwise from the source.

The pass is conservative: it smooths only finite synthetic values, leaves
unresolved values unresolved, and records the changed mask, layer mask, and
parameters in the output H5 attributes. It does not apply a global Gaussian
filter and does not use synthetic pixels as anchors for their own estimate.

## Training consistency

The new proxy directory is used consistently by dataset loading, training,
validation, and real-cloud inference. Existing four-term training remains:

`MSE + gradient + artificial-boundary transition + natural-continuity guard`.

No additional loss is introduced. The natural-continuity term constrains both
natural-gap seam jumps and anomalously large internal natural-gap gradients,
only where the required observed support exists.

## Validation and isolation

The implementation is written under a new versioned output root; existing H5
files and checkpoints are not overwritten. Validation must confirm:

- observed and land values are bitwise unchanged;
- no new ocean NaNs or infinities are introduced;
- modified pixels are synthetic only and obey the change bound;
- seam and natural-interior gradient quantiles are compared with the current
  proxy on fixed Series 8 frames;
- stage diagnostics identify whether the improvement occurs before or after
  KNN completion.

Only after these checks pass should a new training run be considered.
