# Gradient-Prior V1 Four-GPU DDP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the isolated gradient-prior H12 trainer on four A100 processes while preserving strict split, loss, metric, and checkpoint semantics.

**Architecture:** Add process-group helpers and rank-aware data loaders to `train.py`. Training uses `DistributedSampler`; validation uses disjoint non-padding rank shards. Each rank keeps raw metric numerator/count state and rank 0 reduces and writes artifacts.

**Tech Stack:** PyTorch DDP/NCCL, `torchrun`, pytest, Python 3.

---

### Task 1: Add tested DDP primitives

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/train.py`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_train.py`

- [ ] Add failing tests for rank-strided validation indices, DDP metadata fields, and a raw metric state reducer that sums numerator/count fields.
- [ ] Run the focused tests and confirm they fail because the helpers do not exist.
- [ ] Implement pure helpers `rank_strided_indices(length, rank, world_size)`, `reduce_metric_state(state, reducer)`, and DDP-aware metadata fields.
- [ ] Run focused tests and confirm they pass.

### Task 2: Add rank-aware epoch reduction

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/train.py`
- Test: `Data_Imputation/retraining/gradient_prior_v1/tests/test_train.py`

- [ ] Add failing tests proving a metric accumulator finalized after raw numerator/count summation, not mean-of-means.
- [ ] Implement `MetricState` serialization and `all_reduce` reduction for metric numerators/counts, loss component weighted sums/counts, total-loss sum, and batch count.
- [ ] Keep the existing single-process `run_epoch` output unchanged.
- [ ] Run the focused tests and complete CPU suite.

### Task 3: Integrate torchrun process ownership and loaders

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/train.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/config.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/README.md`

- [ ] Add environment detection for `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`, NCCL initialization/cleanup, rank-0 manifest creation, barriers, and local CUDA binding.
- [ ] Add `DistributedSampler` training with `drop_last=True` and deterministic epoch setting.
- [ ] Add non-padding rank-strided validation subsets so every Series 7 index is evaluated once.
- [ ] Wrap the model in DDP only when world size is greater than one; save the unwrapped module state dict.
- [ ] Record world size, rank-local batch size, and global batch size in resolved config and checkpoint metadata.
- [ ] Document the four-GPU `torchrun` command and rank-0-only artifact behavior.

### Task 4: Verify and launch

**Files:**
- Runtime output: `Data_Imputation/experiments/gradient_prior_v1/h12_formal_ddp_*/`

- [ ] Run all isolated tests, compilation, and `git diff --check` locally.
- [ ] Sync only the isolated package and plan/spec to A100.
- [ ] Restore and hash-check the OSTIA clean checkpoint on A100.
- [ ] Run a four-process one-epoch/one-batch smoke on GPUs 2,3,6,7 and confirm finite completion, one rank-0 checkpoint, and no rank deadlock.
- [ ] Launch the collision-protected 60-epoch formal H12 training with four processes and one sample per GPU.
- [ ] Confirm the first epoch metrics, GPU process placement, output manifest, and checkpoint retention.
