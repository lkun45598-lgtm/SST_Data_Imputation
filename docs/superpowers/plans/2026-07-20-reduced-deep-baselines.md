# Reduced Deep Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a moderately reduced-capacity version of every deep SST baseline without overwriting existing full-capacity artifacts.

**Architecture:** Keep all model inputs, outputs, losses, masks, and evaluation behavior unchanged. Reduce only channel widths and LaMa residual depth, and add an optional training run tag that selects separate output directories and matching pretrained checkpoints.

**Tech Stack:** Python 3, PyTorch, pytest, CUDA, existing `pretrain_deep.py` and `finetune_deep.py` entry points.

---

### Task 1: Add a capacity regression test

**Files:**
- Create: `Data_Imputation/tests/test_reduced_deep_baselines.py`

- [ ] **Step 1: Write the failing test**

  Import `build_model` and assert the first convolution widths, ConvLSTM hidden width, and LaMa block count equal the reduced design. Use a small output size only where construction needs it; do not run a full training loop.

- [ ] **Step 2: Run the test to verify it fails**

  Run `pytest -q Data_Imputation/tests/test_reduced_deep_baselines.py`.
  Expected: failure because the current factory still builds the full-capacity widths.

### Task 2: Apply reduced model capacities

**Files:**
- Modify: `Data_Imputation/baselines/dincae.py`
- Modify: `Data_Imputation/baselines/deep_models.py`

- [ ] **Step 1: Change constructor defaults**

  Set `DINCAE` and the U-Net/partial-convolution defaults to `base=20`, `ConvLSTM` to `hid=28`, and `LaMaFFC` to `base=40, n_blocks=4`.

- [ ] **Step 2: Change factory arguments**

  Make `build_model` pass those same explicit values for all five names so callers get a consistent reduced architecture.

- [ ] **Step 3: Run the regression test**

  Run `pytest -q Data_Imputation/tests/test_reduced_deep_baselines.py`.
  Expected: pass.

- [ ] **Step 4: Run a forward smoke test**

  Instantiate each factory model with `out_size=(32, 32)`, pass tensors shaped `[1, 30, 32, 32]`, and assert the mean output is `[1, 1, 32, 32]` and contains only finite values.

### Task 3: Isolate reduced training artifacts

**Files:**
- Modify: `Data_Imputation/baselines/pretrain_deep.py`
- Modify: `Data_Imputation/baselines/finetune_deep.py`

- [ ] **Step 1: Add an optional `--tag` argument**

  Default it to the empty string so existing commands preserve their paths. For a non-empty tag, append `_{tag}` to the model experiment directory.

- [ ] **Step 2: Make automatic pretraining lookup tag-aware**

  When `--pretrained auto` is used, point to the matching tagged OSTIA directory.

- [ ] **Step 3: Include the tag in `config.json`**

  This records which capacity/run family produced each checkpoint.

### Task 4: Train the reduced family on one GPU per model

**Files:**
- Runtime outputs only: `Data_Imputation/baselines/experiments/*_reduced_*`

- [ ] **Step 1: Launch OSTIA pretraining**

  Run `pretrain_deep.py --model {name} --gpu {gpu} --tag reduced` for `dincae/unet/convlstm/lama/partialconv` on GPUs `0/1/2/3/4` respectively.

- [ ] **Step 2: Verify each process and log**

  Check process status, each `*_reduced_ostia_pretrain/train.log`, and the first epoch output before proceeding.

- [ ] **Step 3: Launch JAXA h12 fine-tuning**

  Run `finetune_deep.py --model {name} --hour 12 --gpu {gpu} --tag reduced` with the same one-model-per-GPU mapping.

- [ ] **Step 4: Verify checkpoints**

  Confirm each tagged directory contains `best_model.pth`, `config.json`, and a non-empty training history, and report any job that cannot start or complete.
