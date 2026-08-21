# Gradient-Prior V1 Four-GPU DDP Design

## Goal

Extend the isolated gradient-prior retraining package to run one formal H12
training job on four A100 GPUs without changing the established data split,
loss definitions, validation semantics, or artifact-isolation guarantees.

The launch configuration is four processes, one GPU and one sample per process,
for a global batch size of four. The learning rate remains `5e-4`; it is not
scaled with world size. The intended physical GPUs are 2, 3, 6, and 7, exposed
to the job through `CUDA_VISIBLE_DEVICES` and addressed by local ranks 0--3.

## Launch And Process Ownership

The existing trainer gains native `torchrun` support by detecting `RANK`,
`LOCAL_RANK`, and `WORLD_SIZE`. A world size greater than one initializes an
NCCL process group, binds each process to its local CUDA device, loads the same
OSTIA checkpoint, and wraps the model in `DistributedDataParallel`.

Rank 0 alone creates the collision-protected output directory, writes
manifests and epoch metrics, prints progress, and saves checkpoints. Barriers
ensure other ranks do not read or train before rank 0 has completed shared
filesystem setup. Process-group cleanup runs even when training raises.

Single-GPU execution remains supported and retains its current behavior.

## Dataset Sharding

Training uses `DistributedSampler` with deterministic epoch-dependent
shuffling. It drops at most `world_size - 1` tail samples so every rank executes
the same number of backward passes and DDP cannot hang. The underlying dataset
still receives the epoch number, so artificial masks vary deterministically by
epoch and sample index.

Validation uses a deterministic, non-padding rank-strided sampler. Every
Series 7 validation sample is evaluated exactly once across the four ranks;
none is duplicated to make the length divisible by four. Validation runs on
the unwrapped module with no distributed forward collectives, so unequal final
shard lengths are safe.

Series 0--6 remain training only, Series 7 remains validation only, and Series
8 is never opened by the trainer.

## Metric Reduction

Each rank accumulates raw numerators and valid counts for MAE, RMSE, strict
central Gradient MAE, boundary MAE, and all no-reference diagnostics. Loss
component weighted sums, valid counts, total-loss sums, and batch counts are
also retained.

Before finalizing an epoch, these raw values are summed across ranks with
`all_reduce`. Ratios and square roots are computed only after the reduction.
This preserves the existing pixel/edge-weighted semantics and avoids averaging
four rank-level means.

## Checkpoints And Disk Safety

Checkpoint state is saved from the unwrapped model so files remain compatible
with the current evaluator and single-GPU loader. Metadata records world size,
per-rank batch size, and global batch size in addition to the existing split,
statistics digest, loss weights, and normalization.

Only rank 0 writes artifacts. Before a new epoch checkpoint is serialized, old
epoch path names beyond the retention allowance are unlinked while a best-model
hard link continues to preserve the selected inode. This bounds normal output
storage to the best checkpoint plus the latest checkpoint and avoids a
three-checkpoint transient on the space-constrained A100 filesystem.

The locally verified OSTIA source checkpoint may be removed from the A100 only
after all four workers have loaded it successfully and training has entered the
first epoch. It remains recoverable from the 4090 server by its recorded
SHA-256.

## Verification And Failure Handling

Test-first coverage must establish:

- rank-strided validation shards are disjoint and cover every index exactly;
- distributed raw metric states reduce before finalization;
- checkpoint metadata records DDP batch semantics;
- checkpoint pre-pruning preserves the best hard link;
- single-GPU behavior remains valid.

Local verification consists of the complete isolated pytest suite, Python
compilation, and `git diff --check`. The remote gate is a four-process one-batch
DDP smoke run with finite loss, synchronized completion, one rank-0 checkpoint,
and measured peak memory per rank. Only after that smoke run passes may the
new, collision-protected 60-epoch experiment start.

If any worker fails, `torchrun` terminates the group and the formal run is not
reported as active. Existing unrelated GPU processes are not stopped or
modified. GPU memory and utilization are checked immediately before launch;
the job uses physical GPUs 2, 3, 6, and 7 only while each retains sufficient
free memory.
