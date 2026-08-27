# JAXA Unified 72-Hour Diffusion Feasibility Design

## Status

Draft for user review. This document defines a memory and execution feasibility
prototype only. It does not authorize full training or replacement of the
existing FNO-CBAM experiments.

## Goal

Determine whether one mask-conditioned diffusion model can jointly reconstruct
72 consecutive hourly JAXA SST fields at the native `451 x 351` grid on one
24 GB RTX 4090 per DDP rank. The prototype must run one real-data forward pass,
one backward pass, and one short reverse-diffusion sample while reporting peak
CUDA memory and tensor shapes.

If the per-rank workload fits on one 4090, the same model will later be eligible
for eight-GPU DistributedDataParallel (DDP) training. DDP is treated as a
throughput mechanism; the design does not assume that eight cards combine into
one 192 GB memory pool.

## Non-Goals

- Do not start a full training run.
- Do not use OSTIA pre-training in this prototype.
- Do not use Gaussian, temporal interpolation, or KNN proxy fields as model
  inputs or targets.
- Do not modify the current FNO-CBAM, proxy-generation, checkpoint, paper-cache,
  or visualization code.
- Do not claim reconstruction accuracy from a memory smoke test.
- Do not copy STIMP's dense spatial graph, hard-coded data paths, chlorophyll
  log transform, or eight-day data assumptions.

## Data Contract

The source is the original hourly JAXA NetCDF archive at
`/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3`.
Each available file contains one Kelvin SST field on a `451 x 351` grid. The
filename timestamp is the authoritative timestamp used to build the timeline.

One sample contains exactly 72 hourly slots from `t-71h` through `t`, inclusive.
Slots are never compressed to the next available observation. If a source file
is absent, its slot contains a zero normalized-value placeholder and an all-zero
observation mask. Missing pixels in an available frame use the same placeholder
and a zero mask. The NetCDF files do not contain a separate land variable, so
the prototype requires an explicit H5 `land_mask` path from the existing JAXA
products. It reads only that static mask from the H5 file and never reads its SST
values. The mask shape must be `451 x 351`, and land is excluded from all losses.

The smoke test normalizes SST using the mean and standard deviation of observed
ocean pixels in its selected 72-slot segment. A full experiment must instead
precompute one fixed normalization from raw observed-ocean pixels in training
Series 0--6 and reuse it unchanged for validation, test, and inference.

The eventual data split remains the approved annual-series split:

- training: Series 0--6;
- validation: Series 7;
- test: Series 8.

No sample may cross a series boundary. The feasibility smoke test reads one
72-slot segment from a training series and does not write a derived dataset.

## Training Mask Semantics

Let `O` denote naturally observed ocean pixels and `A` denote artificially
withheld pixels sampled only from `O`. The conditioning mask is `C = O & ~A`.
Artificial masks are drawn from real JAXA cloud shapes from training-series
timestamps and intersected with `O`; independent random pixels are not the
scientific training protocol.

The clean SST target is known only on `O`. The supervised diffusion loss is
therefore evaluated only on `A`. Naturally missing ocean pixels never receive a
fabricated pixel target. During reverse diffusion, all pixels in `C` are clamped
back to their real JAXA observations after every denoising step. Land remains
excluded. This is the JAXA adaptation of STIMP's masked-observation training
principle.

For the memory smoke test, a deterministic small artificial mask may be used to
guarantee a non-empty loss, but the prototype interface must accept the final
real-cloud-shaped mask contract.

## Prototype Architecture

### Shared Spatial Encoder

Each hourly pair `[normalized SST, observation mask]` is passed through the same
2D convolutional encoder. The encoder downsamples spatial dimensions by a factor
of eight before any temporal attention. The implementation processes fixed-size
time chunks through this shared encoder to bound peak activation memory without
changing frame values or temporal order.

### Time and Diffusion Conditioning

The latent representation receives:

- diffusion-step embedding;
- hour-of-day sine and cosine encoding;
- day-of-year sine and cosine encoding;
- slot-validity information derived from the observation mask.

Actual slots remain one hour apart by construction. Calendar encodings expose
diurnal and seasonal phase without treating a missing file as elapsed zero time.

### Spatiotemporal Denoiser

The denoiser alternates residual spatial convolutions with temporal self-attention
over the 72 latent states at each downsampled spatial location. It uses PyTorch
scaled-dot-product attention so a memory-efficient CUDA kernel can be selected.
No dense `N x N` spatial adjacency matrix is constructed.

For training, the shared encoder creates a clean latent state from the original
observations `O` and a conditioning latent state from the withheld observations
`C`. The clean-state placeholder remains zero at naturally missing pixels, but
those pixels are never supervised. Gaussian diffusion noise is applied to the
clean latent state at one randomly selected diffusion step. The denoiser receives
the noisy latent, conditioning latent, conditioning mask, calendar encodings,
and diffusion-step embedding.

The prototype predicts the clean latent state, matching STIMP's original-data
prediction formulation rather than predicting diffusion noise. Its prediction
is decoded to native-grid SST, and the training MSE is evaluated in pixel space
only on `A`. The spatial decoder is shared across time and reconstructs all 72
full-resolution fields. It processes fixed-size time chunks because temporal
interaction has already occurred in latent space.

### Output Composition

The raw model output has shape `[B, 72, 1, 451, 351]`. Its final composed output
uses original JAXA values wherever the conditioning mask is one and model values
elsewhere. Composition is applied to every reconstructed hour, not only the last
frame.

## Feasibility Configuration

The first CUDA test uses these fixed conservative settings:

- per-rank batch size: 1;
- latent width: 32;
- spatial downsampling: 8;
- temporal-attention blocks: 1;
- attention heads: 4;
- diffusion training step: one randomly sampled step;
- reverse-sampling smoke steps: 4;
- precision: BF16 autocast when supported, otherwise FP16 autocast;
- gradient checkpointing: enabled for the spatiotemporal block.

The smoke configuration is deliberately smaller than a final scientific model.
It validates the data path and memory strategy, not final capacity. Model width,
block count, and diffusion steps remain unchanged during this smoke test.

## Distributed Execution

The new experiment will use `torchrun` and the repository's established NCCL
DDP pattern. Each rank owns one complete model and one real 72-hour sample. With
eight ranks and per-rank batch size one, the global batch size is eight. Only
rank zero writes reports or checkpoints.

The feasibility sequence is:

1. CPU shape and mask-contract test.
2. One-GPU forward-only test.
3. One-GPU forward/backward test with peak-memory reporting.
4. One-GPU four-step reverse-diffusion sampling test.
5. Two-GPU DDP one-step synchronization test.
6. Eight-GPU DDP one-step test only if all eight GPUs are available and the
   one-GPU test leaves an adequate memory margin.

No background `tmux` session is required for these bounded smoke tests.

## Failure Handling

CUDA out-of-memory is handled by recording the failing stage and configuration,
clearing the process, and reducing memory in this order:

1. reduce encoder/decoder time-chunk size;
2. enable or broaden activation checkpointing;
3. reduce latent width from 32 to 24;
4. increase spatial downsampling from 8 to 16.

The 72-slot temporal window and native output resolution are not silently
reduced. Any such scope change requires explicit user approval.

Missing source files are valid all-missing time slots. A corrupt or wrong-shaped
NetCDF file is a data error and must fail with its exact path and timestamp.
Samples with no artificially withheld observed ocean pixels are rejected and
resampled; their loss is never replaced with zero.

## Validation Gates

The prototype passes only if all of the following hold:

1. Input, latent, raw-output, and composed-output shapes match the documented
   contract.
2. The composed output exactly preserves every conditioned JAXA observation.
3. Loss support equals the artificial-mask ocean support and excludes natural
   gaps and land.
4. One forward/backward optimizer step produces finite loss and finite gradients.
5. Four reverse steps produce finite SST values on all ocean pixels.
6. Peak allocated and reserved CUDA memory are recorded; the backward test must
   fit below 22 GB on a 24 GB 4090 to retain operating margin.
7. The two-rank DDP smoke test completes with finite, synchronized parameters.

The eight-rank test is an execution check, not a substitute for the per-rank
22 GB gate.

## Isolation and Artifacts

All implementation files will live under a new directory:
`Data_Imputation/retraining/stimp_jaxa_unified/`.

All smoke-test artifacts will be written beneath a new caller-specified output
directory that refuses to reuse a non-empty path. The report will contain the
resolved configuration, source timestamps, missing-slot count, artificial-mask
support, parameter count, precision, stage runtimes, and CUDA peak-memory values.
No existing checkpoint or cache may be overwritten.

## Decision After the Smoke Test

Passing this design establishes only that native-grid, 72-hour joint diffusion
is computationally feasible. Before full training, a separate approved design
must define final model capacity, diffusion schedule, cloud-mask sampling,
validation metrics, sampling ensemble size, and comparison against the current
FNO-CBAM model. If the prototype cannot meet the 22 GB gate after the documented
memory reductions, the next design will evaluate target-frame-only decoding or
a separately trained autoencoder for latent diffusion.
