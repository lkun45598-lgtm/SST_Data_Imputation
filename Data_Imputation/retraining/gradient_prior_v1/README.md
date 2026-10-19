# Gradient-Prior V1 Retraining

This directory is an isolated JAXA FNO-CBAM retraining implementation. It does
not modify the existing training or inference modules and does not overwrite
existing checkpoints, caches, figures, or paper files.

## Protocol

- Series 0--6: training and monthly observed-gradient statistics.
- Series 7: fixed-mask validation and hyperparameter decisions.
- Series 8: final test only; the trainer never opens it.
- Historical frames: causal KNN proxy plus original missing-mask channels.
- Target frame: natural gaps and artificial gaps are both hidden.
- Supervised MSE/gradient/boundary terms: artificial observed gaps only.
- Genuine natural gaps: month-conditioned high-gradient, observation-seam,
  one-sided seam-continuity, and low-frequency proxy-energy regularizers only.
  These are no-reference priors,
  not errors against unavailable cloud-under truth.
- Output composition preserves every original observed pixel and all land
  values. There is no Gaussian filtering or hard gradient clipping.

## Generate Monthly Priors

```bash
python -m Data_Imputation.retraining.gradient_prior_v1.gradient_stats \
  --data-dir Data_Imputation/experiments/hourly_data/h12 \
  --series 0 1 2 3 4 5 6 \
  --output Data_Imputation/experiments/gradient_prior_v1/stats/h12_train_monthly_v2.json
```

The statistics writer refuses to replace an existing file. The checked H12
training statistics have SHA-256:

`f9d3346ba243dda50341c8bc4291d4bd4a6c5b952f7906b8ccaa13ddb627a818`

Each month stores two distinct distributions: strict five-point central-gradient
magnitudes for diagnostics and strict observed-ocean adjacent-edge differences
for `L_high` and `L_seam`. The loss never applies a threshold measured with a
different finite-difference stencil.

`L_seam-cont` is evaluated only where a natural-gap pixel borders two preserved
observed-ocean pixels in the outward direction. It matches the cross-seam
finite difference to that one-sided preserved-side difference; it does not use
cloud-under truth, Gaussian filtering, or hard clipping.

## GPU Smoke Run

Run this only when one local RTX 4090 has enough free memory. Standard FP32
AdamW requires about 9.3 GiB for parameters, gradients, and optimizer states,
plus FNO activations and CUDA workspaces.

```bash
python -m Data_Imputation.retraining.gradient_prior_v1.train \
  --experiment-name h12_smoke_YYYYMMDD_HHMMSS \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --train-limit 1 \
  --valid-limit 1 \
  --device cuda:0
```

## Five-Epoch Bounded Pilot

The bounded pilot samples frames evenly across every included annual series;
it is not restricted to the beginning of the time span.

```bash
python -m Data_Imputation.retraining.gradient_prior_v1.train \
  --experiment-name h12_pilot_YYYYMMDD_HHMMSS \
  --epochs 5 \
  --batch-size 1 \
  --num-workers 2 \
  --train-limit 256 \
  --valid-limit 64 \
  --device cuda:0
```

## Four-A100 DDP Training

Expose physical GPUs 2, 3, 6, and 7 to four `torchrun` processes. Each process
uses a rank-local batch size of 1, giving a global batch size of 4. The learning
rate remains `5e-4` and is not scaled by world size.

Run the four-process smoke gate first:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 torchrun --standalone --nproc_per_node=4 \
  -m Data_Imputation.retraining.gradient_prior_v1.train \
  --experiment-name h12_ddp_smoke_YYYYMMDD_HHMMSS \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --train-limit 4 \
  --valid-limit 4 \
  --max-epoch-checkpoints 1
```

After the smoke run completes with finite metrics, launch the formal run:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 torchrun --standalone --nproc_per_node=4 \
  -m Data_Imputation.retraining.gradient_prior_v1.train \
  --experiment-name h12_gradient_prior_v1_ddp_YYYYMMDD_HHMMSS \
  --epochs 60 \
  --batch-size 1 \
  --num-workers 2 \
  --learning-rate 5e-4 \
  --max-epoch-checkpoints 1
```

Training uses a deterministic `DistributedSampler` with equal rank-local step
counts. Validation uses non-padding rank-strided shards, so each Series 7 sample
is evaluated exactly once. Metrics are finalized only after reducing raw sums
and counts across all ranks. Rank 0 alone writes manifests, metrics, and the
unwrapped model checkpoint.

Every run creates a new directory below
`Data_Imputation/experiments/gradient_prior_v1/`. A non-empty experiment
directory is rejected. Each epoch checkpoint includes model and fresh optimizer
state, resolved configuration, exact annual split, monthly-statistics digest,
loss weights, normalization values, and validation metrics. `best_model.pth` is
a hard link to the best epoch checkpoint, so it does not duplicate the large
checkpoint contents.

## Verification

```bash
pytest -q Data_Imputation/retraining/gradient_prior_v1/tests
python -m py_compile Data_Imputation/retraining/gradient_prior_v1/*.py
```
