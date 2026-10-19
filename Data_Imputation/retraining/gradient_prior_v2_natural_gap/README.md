# Natural-Gap V2 Retraining

This directory is an isolated JAXA FNO-CBAM retraining implementation. It does
not modify the existing training or inference modules and does not overwrite
existing checkpoints, caches, figures, or paper files.

## Protocol

- Series 0--6: training and monthly observed-gradient statistics.
- Series 7: fixed-mask validation and hyperparameter decisions.
- Series 8: final test only; the trainer never opens it.
- Historical frames: causal KNN proxy plus original missing-mask channels.
- Target frame: natural gaps and artificial gaps are both hidden.
- The active training objective is
  `L_MSE(artificial) + 0.2 L_gradient(artificial) + 0.05 L_boundary-transition(artificial) + 0.02 L_natural-seam`.
- `L_boundary-transition` uses only one-pixel artificial-mask boundary edges
  whose two endpoints are original observed ocean. It combines a Huber value
  error at the artificial endpoint with an L1 cross-edge gradient error.
- Large artificial masks are real cloud shapes placed only on original observed
  ocean pixels of frames with at least 70% original-observation coverage. They
  use the same true SST targets as standard masks.
- Natural gaps never receive a proxy pixel, proxy-gradient, or pseudo-target
  loss. `L_natural-seam` is only a weak upper-bound guard on excessive
  natural-to-preserved ocean jumps, using the monthly observed-edge P99.9
  threshold.
- Real-cloud deployment preserves observed ocean and land values exactly. At a
  first-layer natural-cloud pixel, the candidate `0.5 * causal proxy + 0.5 *
  FNO` is used only when it does not increase the jump across any adjacent
  preserved-ocean edge and strictly reduces at least one; otherwise the raw FNO
  value is retained. The proxy is the unmasked current causal field rather than
  the mean-filled model input, and deeper cloud layers always use the raw FNO
  output. This local, adaptive transition is inference-only: artificial-mask
  training and validation retain pure FNO reconstruction inside their hidden
  pixels.
- There is no global Gaussian smoothing, clipping, or iterative test-time
  optimization.

## Generate Monthly Priors

```bash
python -m Data_Imputation.retraining.gradient_prior_v2_natural_gap.gradient_stats \
  --data-dir Data_Imputation/experiments/hourly_data/h12 \
  --series 0 1 2 3 4 5 6 \
  --proxy-lowpass-window 15 \
  --proxy-min-observed-support 0.8 \
  --output Data_Imputation/experiments/gradient_prior_v2_natural_gap/stats/h12_train_monthly_v3.json
```

The statistics writer refuses to replace an existing file. The checked H12
training statistics have SHA-256:

`00da7aba19b6873e6815854d3198e44ecc3b32d974dc9562b16f020bc13e700e`

Each month stores strict five-point central-gradient magnitudes and strict
observed-ocean adjacent-edge differences. The active objective reads the
observed-edge threshold for `L_natural-seam`; proxy high-pass fields remain in
the v3 statistics schema for compatibility with archived diagnostics but are
not active training losses. The v3 loader intentionally rejects earlier v2
JSON files.

## GPU Smoke Run

Run this only when one local RTX 4090 has enough free memory. Standard FP32
AdamW requires about 9.3 GiB for parameters, gradients, and optimizer states,
plus FNO activations and CUDA workspaces.

```bash
python -m Data_Imputation.retraining.gradient_prior_v2_natural_gap.train \
  --experiment-name h12_smoke_YYYYMMDD_HHMMSS \
  --gradient-stats Data_Imputation/experiments/gradient_prior_v2_natural_gap/stats/h12_train_monthly_v3.json \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --train-limit 1 \
  --valid-limit 1 \
  --device cuda:0
```

## Bounded Pilot

The bounded pilot samples frames evenly across every included annual series;
it is not restricted to the beginning of the time span.

```bash
python -m Data_Imputation.retraining.gradient_prior_v2_natural_gap.train \
  --experiment-name h12_pilot_YYYYMMDD_HHMMSS \
  --epochs 5 \
  --batch-size 1 \
  --num-workers 2 \
  --train-limit 256 \
  --valid-limit 64 \
  --device cuda:0
```

## Local Multi-GPU Training

After the local single-GPU smoke completes with finite metrics, expose only
free local RTX 4090 GPUs. Each process uses rank-local batch size 1; the
learning rate remains `5e-4` and is not scaled by world size. Start the formal
run from tmux and use an experiment name that does not exist:

```bash
tmux new-session -d -s sst_natural_gap_YYYYMMDD_HHMMSS \
  'cd /data1/user/lz/SST_Data_Imputation && CUDA_VISIBLE_DEVICES=<free_local_gpus> \
  torchrun --standalone --nproc_per_node=<gpu_count> \
  -m Data_Imputation.retraining.gradient_prior_v2_natural_gap.train \
  --experiment-name h12_natural_gap_YYYYMMDD_HHMMSS \
  --gradient-stats Data_Imputation/experiments/gradient_prior_v2_natural_gap/stats/h12_train_monthly_v3.json \
  --epochs 60 --batch-size 1 --num-workers 2 --learning-rate 5e-4 \
  --max-epoch-checkpoints 1 2>&1 | tee \
  Data_Imputation/experiments/gradient_prior_v2_natural_gap/h12_natural_gap_YYYYMMDD_HHMMSS/train.log'
```

Training uses a deterministic `DistributedSampler` with equal rank-local step
counts. Standard and large fixed-mask Series 7 validation each use non-padding
rank-strided shards, so every selected sample is evaluated exactly once. Metrics
are finalized only after reducing raw sums and counts across all ranks. Rank 0
alone writes manifests, metrics, and the unwrapped model checkpoint. The best
checkpoint is selected only by standard-mask MAE.

## Evaluation and Diagnostics

`evaluate.py` writes separately named `standard` and `large` Series 7 metric
groups. Run `realcloud_diagnostics.py` on Series 8 to report exact global
natural-interior and deployment-seam edge-difference quantiles, full seam
exceedance, and local outlier rates for the same local-transition composition
used by real-cloud visualization:

```bash
python -m Data_Imputation.retraining.gradient_prior_v2_natural_gap.realcloud_diagnostics \
  --checkpoint <best_model.pth> \
  --gradient-stats Data_Imputation/experiments/gradient_prior_v2_natural_gap/stats/h12_train_monthly_v3.json \
  --output Data_Imputation/experiments/gradient_prior_v2_natural_gap/comparisons/series8_diagnostics.json
```

Every run creates a new directory below
`Data_Imputation/experiments/gradient_prior_v2_natural_gap/`. A non-empty experiment
directory is rejected. Each epoch checkpoint includes model and fresh optimizer
state, resolved configuration, exact annual split, monthly-statistics digest,
loss weights, normalization values, and validation metrics. `best_model.pth` is
a hard link to the best epoch checkpoint, so it does not duplicate the large
checkpoint contents.

## Verification

```bash
pytest -q Data_Imputation/retraining/gradient_prior_v2_natural_gap/tests
python -m py_compile Data_Imputation/retraining/gradient_prior_v2_natural_gap/*.py
```
