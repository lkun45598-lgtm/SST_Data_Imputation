# JAXA Gradient-Prior Retraining Design

## Objective

Retrain the hourly FNO-CBAM model so that it does not introduce implausible
structures inside genuine JAXA cloud gaps. Original satellite observations are
always preserved. The change must not overwrite existing source files, models,
caches, metrics, or figures.

The implementation is successful when it aligns target-frame masking between
training and deployment, retains supervised reconstruction quality on held-out
observations, reduces model-generated gradient outliers in genuine gaps, and
does not replace naturally missing pixels with KNN values treated as truth.

## Scope And Isolation

All new source code lives under:

`Data_Imputation/retraining/gradient_prior_v1/`

All generated artifacts live under:

`Data_Imputation/experiments/gradient_prior_v1/<experiment-name>/`

The existing files below are read-only dependencies and are not edited:

- `Data_Imputation/models/fno_cbam_temporal.py`
- `Data_Imputation/training/train_jaxa_hourly.py`
- `Data_Imputation/inference/jaxa_inference_dataset.py`
- existing checkpoints, evaluation caches, paper figures, and LaTeX files

Every run writes its resolved configuration, source checkpoint identity,
gradient-statistics hash, metrics, and checkpoints. A non-empty output
directory is rejected unless a different experiment name is supplied.

## Data Split

- Training: annual Series 0--6
- Validation and hyperparameter decisions: annual Series 7
- Final test only: annual Series 8

Series 8 must not be read by the training loop, threshold builder, pilot model
selection, or weight tuning.

## Deployment-Matched Input

For each target frame, define disjoint masks over ocean pixels:

- `natural_mask`: the original JAXA missing mask.
- `artificial_mask`: a real-cloud-shaped mask sampled only from pixels that
  were originally observed.
- `reconstruction_mask = natural_mask OR artificial_mask`.
- `supervised_mask = artificial_mask`.

The preceding 29 frames retain the deployed causal KNN proxy and their original
missing masks. In the target frame, both natural and artificial gaps are
replaced by the normalization mean, and the target mask channel is
`reconstruction_mask`. This prevents target-day KNN proxy values from leaking
into genuine gaps.

The target array may contain KNN-filled values for storage compatibility, but
supervised losses are allowed to read it only through `supervised_mask`, whose
pixels are guaranteed to be original finite ocean observations.

Output composition uses the model in `reconstruction_mask`, preserves the
input satellite observations outside it, and excludes land. The model does not
perform observation quality control and does not replace observed outliers.

## Gradient Statistics

Month-conditioned gradient priors are computed only from Series 0--6. A
central gradient sample is valid only when the center and all four axial
neighbors are finite, originally observed ocean pixels. No one-sided
difference is used at land, missing pixels, or array boundaries.

For each month, separately save distributions for strict five-point central
gradient magnitude and strict two-endpoint adjacent-edge difference, including
P50, P90, P95, P99, P99.5, and P99.9 in physical units (K/pixel). The adjacent
edge P99.9 is the soft upper threshold for edge-based `L_high` and `L_seam`;
central-gradient statistics remain available for diagnostics. This prevents a
threshold measured with one finite-difference stencil from being applied to a
different stencil. The statistics file records input file identities, split,
algorithm version, and a SHA-256 digest.

## Losses

The composed prediction is denoted by `T_hat`. All physical-gradient priors are
computed after converting normalized output differences to K/pixel.

The total training objective is:

`L = L_mse + lambda_g L_grad + lambda_b L_boundary + lambda_h L_high + lambda_s L_seam + lambda_l L_low`.

### Supervised Terms

- `L_mse`: masked MSE only on `artificial_mask`.
- `L_grad`: L1 gradient error only on adjacent pairs whose two endpoints are
  original observed ocean pixels and where at least one endpoint is artificial.
- `L_boundary`: L1 gradient error only on artificial-mask transitions whose two
  endpoints are original observed ocean pixels.

Natural cloud pixels never participate in these terms.

### No-Reference Natural-Gap Terms

These are regularizers, not supervised errors and not evidence of cloud-under
truth accuracy.

- `L_high`: on natural-gap interior edges, penalize only
  `relu(g_pred / U_month - 1)^2`, where both edge endpoints are naturally
  missing and `U_month` is the training-set monthly adjacent-edge P99.9.
- `L_seam`: apply the same soft exceedance penalty only to transitions between
  a preserved observation and a natural-gap prediction.
- `L_low`: low-pass prediction and the causal proxy with a fixed odd-window
  average filter, compute local RMS gradient energy, and penalize only when
  prediction energy is below `beta` times proxy energy. Patch centers must have
  sufficient natural-gap support. The proxy supplies a weak energy floor; no
  pixelwise value loss or full gradient-vector match to proxy is permitted.

All reductions divide by the number of valid samples. Empty masks produce a
differentiable zero. Priors use soft hinges; inference performs no gradient
clipping, replacement, or Gaussian post-processing.

The pilot starts with `lambda_g=0.2`, `lambda_b=0.1`, `lambda_h=0.05`,
`lambda_s=0.05`, `lambda_l=0.02`, `beta=0.5`, low-pass window 9, and energy
window 9. The resolved values are stored in the run configuration and may be
changed only using Series 7 diagnostics.

## Training And Selection

The first run is an `h12` five-epoch pilot on the local RTX 4090, initialized
from the same clean OSTIA checkpoint used by the current model. It uses a
single-GPU trainer and a fixed validation-mask seed. It saves every epoch and a
best checkpoint without touching existing artifacts.

Validation reports MAE, RMSE, strict Gradient MAE, and artificial-boundary MAE.
SSIM is diagnostic only and is not a model-selection criterion. Genuine-gap
diagnostics report natural-interior upper-threshold exceedance, seam
exceedance, local-outlier rate, and low-pass prediction/proxy energy ratio.

Pilot acceptance requires:

- validation MAE no more than 3% worse than the frozen baseline;
- validation Gradient MAE not worse than the frozen baseline;
- lower model-generated natural-gap outlier/exceedance rates on fixed frames;
- no new periodic or ringing patterns in raw, non-Gaussian visual comparisons.

Selection is Pareto-based after the quality gates rather than a weighted sum of
unrelated metrics. If the pilot passes, run 60 epochs for `h12`. Series 8 is
then evaluated once. Other hourly models are retrained only after the `h12`
protocol is accepted.

## Tests And Failure Handling

Automated tests must establish that:

- natural and artificial masks are disjoint;
- the supervised mask is a subset of original finite ocean observations;
- all target-frame natural and artificial gaps are hidden from model input;
- natural gaps do not affect supervised losses;
- gradient thresholds cannot be built from validation or test series;
- high, seam, and low masks select only their intended locations;
- empty prior masks return finite differentiable zeros;
- changing a natural-gap target value cannot change the supervised loss;
- a one-batch forward/backward/optimizer smoke test is finite;
- existing non-empty output directories are rejected.

Training aborts on missing monthly thresholds, non-finite loss components,
split overlap, absent checkpoint keys, or an output collision. It never silently
falls back to treating proxy pixels as truth.

## Interpretation Boundary

The method can show that reconstructions on held-out observed pixels are
accurate and that outputs in real gaps obey measured structural priors. It
cannot establish error against unavailable cloud-under SST. Original observed
anomalies remain visible by design and must not be attributed to model output.
