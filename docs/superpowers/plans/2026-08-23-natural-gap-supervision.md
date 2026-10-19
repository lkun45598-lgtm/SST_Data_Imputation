# Natural-Gap Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the isolated FNO-CBAM pipeline with large real-cloud supervised masks, confidence-gated low-frequency natural-gap supervision, and continuity constraints over every seam used by the deployed composed field.

**Architecture:** Keep the deployed composition unchanged: FNO supplies the natural and artificial reconstruction pixels, while land and all non-reconstructed input pixels are copied exactly from the causal input. A balanced training wrapper interleaves standard and large real-cloud artificial masks, both drawn only from original observations. Natural cloud gaps receive no pixel-level target; they receive only a small, confidence-gated 15-by-15 low-pass Huber anchor and full deployment-seam constraints, where the proxy is used as a low-frequency reference rather than copied as texture.

**Tech Stack:** Python 3, PyTorch, NumPy, h5py, scipy.ndimage, pytest, torchrun/DDP.

---

## Plan Review

1. **Goal clarity:** The acceptance decision is observable: the new isolated run must retain or improve Series 7 artificial-mask MAE/RMSE/gradient/boundary metrics for both mask sizes, while reducing Series 8 natural-gap seam discontinuity and outlier diagnostics without inference filtering.
2. **Change boundary:** Modify only `Data_Imputation/retraining/gradient_prior_v1/{config.py,dataset.py,gradient_stats.py,losses.py,metrics.py,train.py,evaluate.py,README.md}` and their tests. Add `Data_Imputation/retraining/gradient_prior_v1/realcloud_diagnostics.py` and its tests. Do not change the FNO-CBAM architecture, `Data_Imputation/inference/real_cloud_mask.py`, historical data H5 files, existing experiment outputs, paper assets, or any old checkpoint.
3. **Dependencies:** `train.py` is the only training caller of `hybrid_loss`; `evaluate.py` builds the same validation dataset and invokes `run_epoch`; visualization keeps its independent real-cloud sampler and only consumes checkpoints. Dataset wrappers must preserve `set_epoch`, and all DDP ranks must build the identical balanced training length before `DistributedSampler` shards it.
4. **Rollback:** This package is isolated and all new runs use `prepare_new_output_dir`; keep the existing 60-epoch checkpoint untouched. Each implementation task is committed independently. Reverting the new commits or selecting the prior checkpoint restores the former behavior with no data rollback.
5. **Verification:** Add mask-separation, high-coverage selection, loss-gradient, proxy-confidence, full-seam coverage, config, DDP-loader, and diagnostics tests. Run the package test suite and `py_compile`, then a one-step local-4090 smoke run before the separate 8x4090 experiment. Compare fixed Series 7 standard/large validation and full Series 8 no-reference diagnostics against the previous checkpoint.
6. **Risks:** The main risks are accidentally treating proxy values as targets, sampling large masks outside trusted observations, an old statistics JSON silently bypassing the confidence gate, and DDP streams losing their 1:1 balance. Prevent them through explicit masks, versioned/validated statistics, stream-index tests, and persisted config/statistics digests. A lower seam rate alone can be achieved by blur, so acceptance also requires artificial-mask errors and the fixed frame-179 visual/attribution diagnostic.

**Plan review passed. Key risk reminder:** proxy values must never enter a full-resolution MSE or supervised gradient target, including through an unmasked low-pass operation.

## File Structure

- Modify `Data_Imputation/retraining/gradient_prior_v1/config.py`: expose large-mask curriculum and proxy-confidence hyperparameters, with strict validation and JSON serialization.
- Modify `Data_Imputation/retraining/gradient_prior_v1/dataset.py`: find high-observation frames, create a deterministic large-mask stream, and provide a balanced training wrapper without changing target construction.
- Modify `Data_Imputation/retraining/gradient_prior_v1/gradient_stats.py`: version the monthly statistics document and add the observed-SST 15-by-15 high-pass threshold used only to reject unreliable proxy locations.
- Modify `Data_Imputation/retraining/gradient_prior_v1/losses.py`: implement the confidence-gated coarse proxy Huber term and replace observation-only seam eligibility with deployment-composition eligibility.
- Modify `Data_Imputation/retraining/gradient_prior_v1/metrics.py`: report full deployment seam exceedance and edge-difference summaries.
- Modify `Data_Imputation/retraining/gradient_prior_v1/train.py`: construct the streams, pass the new monthly threshold into the loss, validate both artificial-mask regimes, and preserve complete run provenance.
- Modify `Data_Imputation/retraining/gradient_prior_v1/evaluate.py`: evaluate standard and large fixed artificial masks separately.
- Create `Data_Imputation/retraining/gradient_prior_v1/realcloud_diagnostics.py`: scan Series 8 with no artificial masks and write no-reference seam/interior diagnostics once.
- Modify `Data_Imputation/retraining/gradient_prior_v1/README.md`: document the precise no-pseudo-target policy, new statistics generation, local-4090 smoke command, and 8x4090 launch command.

### Task 1: Add Curriculum Configuration and Tests

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/config.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/tests/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

Add a valid non-default configuration and invalid-boundary cases:

```python
def test_natural_gap_curriculum_fields_are_serialized(tmp_path):
    config = replace(
        RunConfig.for_test(tmp_path),
        large_mask_ratio_low=0.4,
        large_mask_ratio_high=0.6,
        large_mask_min_observed_fraction=0.7,
        proxy_lowpass_window=15,
        proxy_core_support=0.8,
        lambda_coarse_proxy=0.02,
        proxy_huber_delta_k=0.5,
    )
    config.validate()
    assert config.to_dict()["large_mask_ratio_high"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "changes",
    [
        {"large_mask_ratio_low": 0.0},
        {"large_mask_ratio_high": 1.1},
        {"large_mask_ratio_low": 0.7, "large_mask_ratio_high": 0.6},
        {"large_mask_min_observed_fraction": 1.1},
        {"proxy_lowpass_window": 14},
        {"proxy_core_support": -0.1},
        {"lambda_coarse_proxy": -0.01},
        {"proxy_huber_delta_k": 0.0},
    ],
)
def test_natural_gap_curriculum_rejects_invalid_values(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(RunConfig.for_test(tmp_path), **changes).validate()
```

- [ ] **Step 2: Run the configuration tests and verify failure**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_config.py`

Expected: FAIL because the new `RunConfig` fields do not yet exist.

- [ ] **Step 3: Add validated immutable settings**

Add these `RunConfig` defaults and include all loss settings in `validate()` and `to_dict()` through `asdict`:

```python
large_mask_ratio_low: float = 0.4
large_mask_ratio_high: float = 0.6
large_mask_min_observed_fraction: float = 0.7
proxy_lowpass_window: int = 15
proxy_core_support: float = 0.8
lambda_coarse_proxy: float = 0.02
proxy_huber_delta_k: float = 0.5
```

Validate `0 < large_mask_ratio_low <= large_mask_ratio_high <= 1`, `0 < large_mask_min_observed_fraction <= 1`, an odd `proxy_lowpass_window >= 3`, `0 <= proxy_core_support <= 1`, and non-negative coarse weight with a strictly positive Huber delta. Keep the existing `lowpass_window`, `energy_window`, and `lambda_low` unchanged so the prior experiment remains reproducible.

- [ ] **Step 4: Run the configuration tests and verify success**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_config.py`

Expected: PASS.

- [ ] **Step 5: Commit the isolated configuration change**

```bash
git add Data_Imputation/retraining/gradient_prior_v1/config.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_config.py
git commit -m "feat: configure natural-gap curriculum"
```

### Task 2: Implement High-Observation Large-Mask Stream

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/dataset.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/tests/test_dataset.py`

- [ ] **Step 1: Write failing dataset tests for trusted large masks and balanced selection**

Extend the test H5 fixture to support a clear frame, then test selection and stream alternation:

```python
def test_high_observation_indices_use_only_finite_original_ocean_pixels(tmp_path):
    data_dir = tmp_path / "data"
    _write_series(data_dir, 0)
    indices = high_observation_global_indices(
        data_dir, series_ids=(0,), min_fraction=0.9
    )
    assert indices.tolist() == [0]


def test_large_masks_remain_a_subset_of_original_observations(tmp_path):
    dataset = _large_dataset(tmp_path)
    sample = dataset[0]
    assert sample["mask_stream"] == "large"
    assert np.all(~sample["artificial_mask"] | sample["original_obs_mask"])
    assert not np.any(sample["artificial_mask"] & sample["natural_mask"])


def test_balanced_training_dataset_alternates_streams_and_propagates_epoch(tmp_path):
    standard = _standard_dataset(tmp_path)
    large = _large_dataset(tmp_path)
    mixed = BalancedMaskTrainingDataset(standard, large)
    assert len(mixed) == 2 * max(len(standard), len(large))
    assert mixed[0]["mask_stream"] == "standard"
    assert mixed[1]["mask_stream"] == "large"
    mixed.set_epoch(3)
    assert standard.epoch == large.epoch == 3
```

The fixture's `artificial_mask_provider` should select a fixed valid pixel so this test never relies on random cloud morphology. Add one test asserting a provider attempting a natural/land pixel still raises the existing subset error.

- [ ] **Step 2: Run dataset tests and verify failure**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_dataset.py`

Expected: FAIL because `high_observation_global_indices`, `mask_stream`, and `BalancedMaskTrainingDataset` are missing.

- [ ] **Step 3: Add deterministic high-coverage indexing and a balanced wrapper**

Implement a read-only scanner with the following behavior:

```python
def high_observation_global_indices(
    data_dir: Path | str,
    *,
    series_ids: Iterable[int],
    min_fraction: float,
) -> np.ndarray:
    """Return dataset-global frame indices with trusted ocean coverage >= threshold."""
```

For every frame, define trusted coverage as
`sum(original_obs_mask & ocean & isfinite(sst_data)) / sum(ocean)`.
Accumulate offsets in exactly the `series_ids` order used by `DeploymentMatchedJAXADataset`; do not include Series 7 or 8 when called for training. Raise a clear error if no frame qualifies.

Add optional `active_global_indices: np.ndarray | None` and `mask_stream: str` parameters to `DeploymentMatchedJAXADataset`. Validate indices are sorted, unique, and inside the base frame range; map its external item index through this array before calling the current frame locator. Leave the target, proxy, history, reconstruction-mask, and strict `artificial & ~eligible` checks untouched. Include `"mask_stream": self.mask_stream` in the returned sample.

Add a wrapper:

```python
class BalancedMaskTrainingDataset(Dataset):
    """One standard and one large trusted-mask item per two training indices."""

    def __len__(self) -> int:
        return 2 * max(len(self.standard), len(self.large))

    def __getitem__(self, index: int) -> dict[str, object]:
        stream = self.standard if index % 2 == 0 else self.large
        return stream[(index // 2) % len(stream)]

    def set_epoch(self, epoch: int) -> None:
        self.standard.set_epoch(epoch)
        self.large.set_epoch(epoch)
```

Construct the large dataset with the same cloud bank, `active_global_indices=high_observation_global_indices(config.data_dir, series_ids=config.train_series, min_fraction=config.large_mask_min_observed_fraction)`, and `mask_ratio_range=(config.large_mask_ratio_low, config.large_mask_ratio_high)`. The existing real-cloud generator remains unmodified; it is still intersected with the target frame's original observations and therefore never creates a block or pseudo target.

- [ ] **Step 4: Run dataset tests and verify success**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_dataset.py`

Expected: PASS.

- [ ] **Step 5: Commit the data-stream implementation**

```bash
git add Data_Imputation/retraining/gradient_prior_v1/dataset.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_dataset.py
git commit -m "feat: add balanced large cloud-mask curriculum"
```

### Task 3: Version Monthly Statistics with Proxy Confidence Thresholds

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/gradient_stats.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/tests/test_gradient_stats.py`

- [ ] **Step 1: Write failing numerical/statistics tests**

Add tests for an observed-only, land-aware 15-by-15 residual and the v3 payload requirement:

```python
def test_observed_lowpass_residual_excludes_missing_and_land_values():
    field = np.zeros((17, 17), dtype=np.float32)
    field[8, 8] = 4.0
    observed = np.ones_like(field, dtype=bool)
    observed[8, 9] = False
    land = np.zeros_like(field, dtype=bool)
    residual, valid = strict_observed_lowpass_residual(
        field, observed, land, window=15, min_support=0.8
    )
    assert valid[8, 8]
    assert residual[8, 8] > 0
    assert not valid[8, 9]


def test_stats_loader_requires_monthly_proxy_highpass_threshold(tmp_path):
    payload = _complete_payload()
    write_gradient_stats(payload, tmp_path / "old.json")
    with pytest.raises(ValueError, match="proxy_highpass"):
        load_gradient_stats(tmp_path / "old.json")
```

Update `_complete_payload()` so each month has a finite positive `proxy_highpass["p99_9"]`; assert `load_gradient_stats` round-trips it.

- [ ] **Step 2: Run statistics tests and verify failure**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_gradient_stats.py`

Expected: FAIL because the residual function and `proxy_highpass` document field do not exist.

- [ ] **Step 3: Compute observed-SST proxy-artifact thresholds without using filled values**

Add `strict_observed_lowpass_residual(field, original_observed_mask, land_mask, *, window, min_support)`. It must use only `original_observed_mask & ~land_mask & isfinite(field)` as both numerator and denominator support. Compute the low-pass field as a zero-padded local sum divided by the local valid count, set `valid` only at observed ocean centres with at least `min_support * window**2` valid observed neighbours, and return `abs(field - lowpass)` in **K**. Missing/land values must never influence either sum or count.

Use a second fixed-bin histogram with units K to collect that residual for every original observation in each month. Retain the old central and edge gradient distributions unchanged. The document becomes explicit about distinct dimensions:

```json
{
  "algorithm": "strict-observed-gradient-and-lowpass-residual-v3",
  "gradient_units": "K/pixel",
  "proxy_highpass_units": "K",
  "proxy_lowpass_window": 15,
  "proxy_min_observed_support": 0.8,
  "months": {
    "1": {
      "central": {"p99_9": 0.0},
      "edge": {"p99_9": 0.0},
      "proxy_highpass": {"p99_9": 0.0}
    }
  }
}
```

Update `load_gradient_stats()` to require this schema and a finite positive threshold for every month. This intentionally rejects the current `h12_train_monthly_v2.json`; generate a new `h12_train_monthly_v3.json` once, never overwrite v2.

- [ ] **Step 4: Run statistics tests and generate the isolated v3 input**

Run tests first:

```bash
pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_gradient_stats.py
```

Then generate once after the test suite passes:

```bash
python -m Data_Imputation.retraining.gradient_prior_v1.gradient_stats \
  --data-dir Data_Imputation/experiments/hourly_data/h12 \
  --series 0 1 2 3 4 5 6 \
  --proxy-lowpass-window 15 \
  --proxy-min-observed-support 0.8 \
  --output Data_Imputation/experiments/gradient_prior_v1/stats/h12_train_monthly_v3.json
```

Expected: tests PASS; the writer creates v3 once and refuses any overwrite.

- [ ] **Step 5: Commit statistics code and tests, not generated experiment data**

```bash
git add Data_Imputation/retraining/gradient_prior_v1/gradient_stats.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_gradient_stats.py
git commit -m "feat: add observed proxy-confidence statistics"
```

### Task 4: Add Coarse Proxy Loss and Full Deployment-Seam Constraints

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/losses.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/tests/test_losses.py`

- [ ] **Step 1: Write failing loss and gradient-isolation tests**

Add tests with a small 15-by-15 all-ocean field (the test may use `window=3` for compactness):

```python
def test_coarse_proxy_loss_has_gradient_only_at_confident_natural_cores():
    prediction = torch.zeros((1, 1, 7, 7), requires_grad=True)
    proxy = torch.zeros_like(prediction)
    proxy[:, :, 3, 3] = 0.2
    natural = torch.zeros((1, 7, 7), dtype=torch.bool)
    natural[:, 1:6, 1:6] = True
    land = torch.zeros_like(natural)
    loss, count = natural_coarse_proxy_loss(
        prediction, proxy, natural, land, torch.tensor([1.0]),
        norm_std=1.0, lowpass_window=3, min_natural_support=1.0,
        huber_delta_k=0.5,
    )
    loss.backward()
    assert count == 9
    assert torch.isfinite(prediction.grad).all()
    # The 3x3 low-pass receptive field may propagate gradients around an
    # accepted centre, but no non-natural pixel is ever a loss sample.
    eligible, _ = natural_proxy_core_mask(
        proxy, natural, land, torch.tensor([1.0]),
        norm_std=1.0, window=3, min_natural_support=1.0,
    )
    assert torch.all(eligible <= natural.unsqueeze(1))


def test_coarse_proxy_loss_excludes_anomalous_proxy_core():
    proxy = torch.zeros((1, 1, 7, 7))
    proxy[:, :, 3, 3] = 10.0
    natural = torch.zeros((1, 7, 7), dtype=torch.bool)
    natural[:, 1:6, 1:6] = True
    eligible, _ = natural_proxy_core_mask(
        proxy, natural, torch.zeros_like(natural), torch.tensor([1.0]),
        norm_std=1.0, window=3, min_natural_support=1.0,
    )
    assert not eligible[0, 0, 3, 3]


def test_deployment_seam_masks_cover_proxy_and_observation_preserved_sides():
    natural = _mask(3, 7)
    natural[0, 1, 3] = True
    artificial = _mask(3, 7)
    land = _mask(3, 7)
    seam_y, seam_x = natural_deployment_seam_masks(natural, artificial, land)
    assert seam_y.sum() == 0
    assert seam_x.sum() == 2


def test_full_seam_continuity_uses_lowpass_reference_and_rejects_bad_proxy_edge():
    composed = torch.arange(7, dtype=torch.float32)[None, None, None, :].repeat(1, 1, 5, 1)
    proxy = composed.clone()
    natural = _mask(5, 7)
    natural[0, 2, 1] = True
    artificial = _mask(5, 7)
    land = _mask(5, 7)
    linear_loss, linear_count = natural_deployment_seam_continuity_loss(
        composed, proxy, natural, artificial, land, torch.tensor([1.0]),
        norm_std=1.0, lowpass_window=3,
    )
    bad_proxy = proxy.clone()
    bad_proxy[0, 0, 2, 2] = 12.0
    _, bad_count = natural_deployment_seam_continuity_loss(
        composed, bad_proxy, natural, artificial, land, torch.tensor([1.0]),
        norm_std=1.0, lowpass_window=3,
    )
    assert linear_loss == pytest.approx(0.0, abs=1e-8)
    assert linear_count == 1
    assert bad_count == 0
```

Also update the hybrid-loss component expectation to include `"coarse_proxy"`. Assert that an all-false natural mask gives the coarse term a differentiable zero count and zero loss, and that the accepted-centre mask is always a strict subset of natural ocean pixels. The low-pass receptive field is intentionally allowed to propagate gradients around an accepted centre; it must not create a loss sample outside that gate.

- [ ] **Step 2: Run loss tests and verify failure**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_losses.py`

Expected: FAIL because the coarse loss and deployment seam functions do not exist.

- [ ] **Step 3: Implement strict confidence gating and coarse Huber supervision**

Add a helper that uses the current land-aware box sum for both fields:

```python
def natural_proxy_core_mask(
    target_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_proxy_highpass_upper: torch.Tensor,
    *,
    norm_std: float,
    window: int,
    min_natural_support: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return natural centres and a normalized low-pass proxy with reliable texture."""
```

Let `ocean = ~land`. Compute the 15-by-15 low-pass proxy with `_land_aware_lowpass(target_proxy, ocean, window)`. Use `natural_support = box_sum(natural) / box_sum(ocean)` so coastlines are not incorrectly rejected solely because they contain land. A selected centre must be natural ocean, have support at least `min_natural_support`, have a nonzero low-pass denominator, and have physical residual
`abs((target_proxy - proxy_low) * norm_std) <= monthly_proxy_highpass_upper`.

Implement:

```python
def natural_coarse_proxy_loss(
    composed: torch.Tensor,
    target_proxy: torch.Tensor,
    natural_mask: torch.Tensor,
    land_mask: torch.Tensor,
    monthly_proxy_highpass_upper: torch.Tensor,
    *,
    norm_std: float,
    lowpass_window: int,
    min_natural_support: float,
    huber_delta_k: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ocean = ~_mask(land_mask)
    pred_low, _ = _land_aware_lowpass(composed, ocean, lowpass_window)
    eligible, proxy_low = natural_proxy_core_mask(
        target_proxy, natural_mask, land_mask, monthly_proxy_highpass_upper,
        norm_std=norm_std, window=lowpass_window,
        min_natural_support=min_natural_support,
    )
    delta = huber_delta_k / norm_std
    values = F.huber_loss(pred_low, proxy_low.detach(), delta=delta, reduction="none")
    return _masked_mean(values, eligible)
```

`detach()` is mandatory on the proxy path. This is not raw-proxy MSE: only the low-pass values in reliable natural cores can contribute.

- [ ] **Step 4: Replace observation-only seam masks with deployment composition masks**

Define preserved input exactly as it is after output composition during training:

```python
preserved = ocean & ~(natural | artificial)
```

`natural_deployment_seam_masks(natural, artificial, land)` returns an adjacent edge when one endpoint is natural ocean and the other is `preserved`. It must contain all four directions and no land/artificial endpoint. Use it in the seam soft upper-bound loss and in metrics.

For `natural_deployment_seam_continuity_loss`, require a natural pixel `p`, a preserved seam neighbour `q`, and one additional preserved outward neighbour `r`. Compare the composed cross-seam step with the **low-pass** proxy outward step:

```python
error = (composed[p] - composed[q]) - (proxy_low[q] - proxy_low[r])
```

Use all four orientations. An edge is eligible only if `q` and `r` are proxy-confidence-valid; do not use raw proxy values for the reference difference. Return an absolute physical-unit error and differentiable zero when the set is empty. The upper-bound seam hinge still covers every natural-to-preserved edge, including edges where no outward reference exists.

Update `HybridLossWeights` and `hybrid_loss` so the component keys are exactly:

```python
{"mse", "gradient", "boundary", "high", "seam", "seam_continuity", "low", "coarse_proxy"}
```

Pass `monthly_proxy_highpass_upper`, `proxy_lowpass_window`, `proxy_core_support`, and `proxy_huber_delta_k` explicitly. Remove `original_observed_mask` from natural seam function signatures; retain it only for strict supervised losses.

- [ ] **Step 5: Run loss tests and verify success**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_losses.py`

Expected: PASS, including no gradient outside the selected natural-gap cores and full natural-to-preserved seam coverage.

- [ ] **Step 6: Commit losses and their semantic tests**

```bash
git add Data_Imputation/retraining/gradient_prior_v1/losses.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_losses.py
git commit -m "feat: constrain natural gaps without proxy pseudo targets"
```

### Task 5: Update Metrics and Real-Cloud Diagnostics

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/metrics.py`
- Create: `Data_Imputation/retraining/gradient_prior_v1/realcloud_diagnostics.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/tests/test_metrics.py`
- Create: `Data_Imputation/retraining/gradient_prior_v1/tests/test_realcloud_diagnostics.py`

- [ ] **Step 1: Write failing full-seam metric tests**

Replace the observation-only seam expectation with a preservation-based one and add edge quantile checks:

```python
def test_seam_exceedance_includes_temporal_proxy_preserved_pixels():
    field = torch.zeros((1, 1, 3, 5))
    field[0, 0, 1, 2] = 2.0
    natural = _mask(3, 5)
    natural[0, 1, 2] = True
    artificial = _mask(3, 5)
    land = _mask(3, 5)
    exceed, count = seam_exceedance_terms(
        field, natural, artificial, land, torch.tensor([1.0]), norm_std=1.0
    )
    assert exceed == count == 2


def test_edge_difference_summary_reports_requested_quantiles():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    summary = edge_difference_summary(values, torch.ones(4, dtype=torch.bool))
    assert summary["p50"] == pytest.approx(1.5)
    assert summary["p99_9"] == pytest.approx(2.997)
```

Create a tiny H5 real-cloud fixture and test that the diagnostics writer refuses an existing output and its Series 8 mode returns `natural_interior`, `deployment_seam`, and `local_outlier` summaries.

- [ ] **Step 2: Run metric/diagnostic tests and verify failure**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_metrics.py Data_Imputation/retraining/gradient_prior_v1/tests/test_realcloud_diagnostics.py`

Expected: FAIL because the full seam API and diagnostic module are absent.

- [ ] **Step 3: Report diagnostic quantities matching deployment composition**

Change `seam_exceedance_terms` to call `natural_deployment_seam_masks`. Add an `edge_difference_values` helper shared by diagnostics that returns absolute physical adjacent differences for a supplied `(y_mask, x_mask)`, excluding all invalid edges. Keep the epoch accumulator's reported `seam_exceedance_rate` name, but its denominator now means every full deployment seam edge.

Create `realcloud_diagnostics.py` with a CLI accepting checkpoint, data directory, v3 stats, output path, device, batch size, and optional frame limit. It must build real-cloud samples with **no artificial masks**, run the exact `output_composition`, and accumulate:

```json
{
  "series": [8],
  "artificial_masking": false,
  "natural_interior_adjacent_difference_K_per_pixel": {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "p99_9": 0.0},
  "deployment_seam_adjacent_difference_K_per_pixel": {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "p99_9": 0.0},
  "deployment_seam_exceedance_rate": 0.0,
  "natural_local_outlier_rate": 0.0
}
```

Use an exact finite-value accumulator for quantiles (a list/concatenated NumPy arrays is acceptable for 365 frames of 451x351 data); do not estimate quantiles from per-frame quantiles. Include the checkpoint path, v3 statistics digest, and input file identities. Its writer must refuse to overwrite an output JSON.

Add a module-level `no_artificial_mask(eligible, rng)` in `dataset.py` that returns `np.zeros_like(eligible, dtype=bool)`. The diagnostic dataset uses this named provider rather than a lambda so `num_workers > 0` remains spawn-safe. It never opens Series 7 or any training series when the CLI is invoked with the default Series 8.

- [ ] **Step 4: Run metric and diagnostics tests**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_metrics.py Data_Imputation/retraining/gradient_prior_v1/tests/test_realcloud_diagnostics.py`

Expected: PASS.

- [ ] **Step 5: Commit no-reference diagnostics**

```bash
git add Data_Imputation/retraining/gradient_prior_v1/metrics.py \
  Data_Imputation/retraining/gradient_prior_v1/realcloud_diagnostics.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_metrics.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_realcloud_diagnostics.py
git commit -m "feat: report full real-cloud seam diagnostics"
```

### Task 6: Wire Streams, Statistics, Training, and Fixed Validation

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/train.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/evaluate.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/tests/test_train.py`
- Modify: `Data_Imputation/retraining/gradient_prior_v1/tests/test_evaluate.py`

- [ ] **Step 1: Write failing integration tests**

Add tests asserting:

```python
def test_monthly_proxy_highpass_maps_each_batch_month_to_k_threshold():
    result = monthly_proxy_highpass_for(torch.tensor([1, 12]), _stats(), torch.device("cpu"))
    assert torch.allclose(result, torch.tensor([0.2, 2.4]))


def test_build_datasets_balances_standard_and_large_training_streams(tmp_path, monkeypatch):
    calls = []
    class FakeDataset(_OneSampleDataset):
        def __init__(self, _data_dir, **kwargs):
            self.kwargs = kwargs
            self.epoch = 0
            calls.append(self)
        def set_epoch(self, epoch):
            self.epoch = epoch
    monkeypatch.setattr(train_module, "DeploymentMatchedJAXADataset", FakeDataset)
    monkeypatch.setattr(train_module, "high_observation_global_indices", lambda *_a, **_k: np.array([0]))
    train, validations = build_datasets(RunConfig.for_test(tmp_path), cloud_bank=np.zeros((1, 2, 2), bool))
    assert isinstance(train, BalancedMaskTrainingDataset)
    assert calls[0].kwargs["mask_ratio_range"] == (0.2, 0.3)
    assert calls[1].kwargs["mask_ratio_range"] == (0.4, 0.6)
    assert set(validations) == {"standard", "large"}


def test_checkpoint_metadata_keeps_named_validation_regimes(tmp_path):
    metadata = build_checkpoint_metadata(
        RunConfig.for_test(tmp_path), gradient_stats=_stats(), epoch=1,
        validation_metrics={"standard": {"mae": 0.2}, "large": {"mae": 0.3}},
    )
    assert metadata["validation_metrics"]["standard"]["mae"] == pytest.approx(0.2)
    assert metadata["validation_metrics"]["large"]["mae"] == pytest.approx(0.3)
```

Update `_stats()` to include `proxy_highpass` for each month. Extend evaluation tests to assert one output JSON contains the two separately named fixed validation regimes and preserves the statistics digest.

- [ ] **Step 2: Run integration tests and verify failure**

Run: `pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_train.py Data_Imputation/retraining/gradient_prior_v1/tests/test_evaluate.py`

Expected: FAIL because the data builders, monthly mapper, and validation record format have not been changed.

- [ ] **Step 3: Thread the confidence threshold and curriculum through the trainer**

Add:

```python
def monthly_proxy_highpass_for(months, gradient_stats, device) -> torch.Tensor:
    return torch.tensor(
        [gradient_stats["months"][str(int(month))]["proxy_highpass"]["p99_9"] for month in months],
        device=device,
        dtype=torch.float32,
    )
```

In `_forward_loss`, compute both monthly vectors and pass the new vector/configuration arguments into `hybrid_loss`. Add `coarse_proxy` to component aggregation and checkpoint metadata `loss_weights`; include the statistics algorithm/window/support fields in manifests so a later analysis cannot accidentally interpret v3 as v2.

`build_datasets` must create:

1. standard Series 0--6 dataset with 20--30% masks;
2. high-coverage Series 0--6 dataset with `(large_mask_ratio_low, large_mask_ratio_high)` and high-frame indices;
3. `BalancedMaskTrainingDataset(standard, large)`;
4. fixed Series 7 standard validation dataset;
5. fixed Series 7 high-coverage large validation dataset.

Return two validation loaders under explicit `"standard"` and `"large"` keys. Each rank uses the existing non-padding strided validation split per key. In `train`, run both under `torch.no_grad`, write:

```python
"validation": {"standard": standard_metrics, "large": large_metrics}
```

Select `best_model.pth` only by `validation["standard"]["mae"]`, maintaining comparability with the prior best-checkpoint rule. Do not mix the two regimes into an ambiguous average.

Add CLI options for all Task 1 fields and change default statistics path to `h12_train_monthly_v3.json`. `evaluate.py` follows the same two fixed Series 7 regimes and writes two named metric groups, not a combined score.

- [ ] **Step 4: Run integration tests and the full package suite**

Run:

```bash
pytest -q Data_Imputation/retraining/gradient_prior_v1/tests/test_train.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_evaluate.py
pytest -q Data_Imputation/retraining/gradient_prior_v1/tests
python -m py_compile Data_Imputation/retraining/gradient_prior_v1/*.py
```

Expected: all tests PASS; `py_compile` exits 0.

- [ ] **Step 5: Commit trainer and evaluator wiring**

```bash
git add Data_Imputation/retraining/gradient_prior_v1/train.py \
  Data_Imputation/retraining/gradient_prior_v1/evaluate.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_train.py \
  Data_Imputation/retraining/gradient_prior_v1/tests/test_evaluate.py
git commit -m "feat: train and validate natural-gap curriculum"
```

### Task 7: Document, Smoke-Test, and Run the Isolated Local Experiment

**Files:**
- Modify: `Data_Imputation/retraining/gradient_prior_v1/README.md`

- [ ] **Step 1: Update the protocol documentation**

Replace the natural-gap paragraph with these operational guarantees:

```markdown
- Large artificial masks are real cloud shapes placed only on original observed ocean pixels of frames with at least 70% original-observation coverage; they use the same true SST targets as standard masks.
- Natural gaps never receive raw proxy pixel MSE, proxy gradient MSE, or pseudo target values. Their weak loss is a 15x15 land-aware low-pass Huber term only at natural cores whose proxy high-pass residual is no larger than that month's observed-SST P99.9 threshold.
- Every natural-to-preserved ocean interface in the deployed composition receives the seam upper-bound constraint. One-sided continuity references only reliable low-pass proxy differences; raw anomalous proxy edges are excluded.
- Inference uses the unfiltered model output inside the reconstruction mask and copies the input outside it. There is no Gaussian smoothing, clipping, or iterative test-time optimization.
```

Document `h12_train_monthly_v3.json`, the separate standard/large Series 7 reports, and Series 8 full no-reference diagnostics. Remove all A100-specific launch prose and show only local RTX 4090 commands.

- [ ] **Step 2: Run the single-GPU local smoke gate**

Choose a currently free local 4090 and run an output name that does not exist:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python -m Data_Imputation.retraining.gradient_prior_v1.train \
  --experiment-name h12_natural_gap_smoke_YYYYMMDD_HHMMSS \
  --gradient-stats Data_Imputation/experiments/gradient_prior_v1/stats/h12_train_monthly_v3.json \
  --epochs 1 --batch-size 1 --num-workers 0 --train-limit 4 --valid-limit 4
```

Expected: finite `coarse_proxy`, full seam counts, separately named standard/large validation blocks, and a newly created isolated output directory.

- [ ] **Step 3: Run local 8x4090 training in tmux only after the smoke gate passes**

Start a named detached tmux session; never attach an output path that already exists:

```bash
tmux new-session -d -s sst_natural_gap_YYYYMMDD_HHMMSS \
  'cd /data1/user/lz/SST_Data_Imputation && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 -m Data_Imputation.retraining.gradient_prior_v1.train --experiment-name h12_natural_gap_v1_YYYYMMDD_HHMMSS --gradient-stats Data_Imputation/experiments/gradient_prior_v1/stats/h12_train_monthly_v3.json --epochs 60 --batch-size 1 --num-workers 2 --learning-rate 5e-4 --max-epoch-checkpoints 1 2>&1 | tee Data_Imputation/experiments/gradient_prior_v1/h12_natural_gap_v1_YYYYMMDD_HHMMSS/train.log'
```

Before launch, use `nvidia-smi` to confirm each selected local GPU has sufficient free memory. If a GPU is busy, select another local set; do not use A100 resources. Record the final tmux session name and output directory without deleting any prior run.

- [ ] **Step 4: Compare the selected checkpoint with the preceding seam-continuity checkpoint**

Evaluate both checkpoints into new files:

```bash
python -m Data_Imputation.retraining.gradient_prior_v1.evaluate \
  --checkpoint <new_best_model> --gradient-stats <v3_stats> \
  --output Data_Imputation/experiments/gradient_prior_v1/comparisons/new_series7.json
python -m Data_Imputation.retraining.gradient_prior_v1.realcloud_diagnostics \
  --checkpoint <new_best_model> --gradient-stats <v3_stats> \
  --output Data_Imputation/experiments/gradient_prior_v1/comparisons/new_series8_diagnostics.json
```

Repeat with the current seam-continuity best checkpoint, using a distinct output name. Render the identical Series 8 frame 179 with `visualize_realcloud_style.py` and `diagnose_realcloud_attribution.py` into a new directory. Accept the new checkpoint only if the Series 7 standard and large metrics do not materially regress, Series 8 seam/outlier rates improve, and frame 179 does not show a blurred replacement for the prior artifacts.

- [ ] **Step 5: Commit documentation only**

```bash
git add Data_Imputation/retraining/gradient_prior_v1/README.md
git commit -m "docs: describe natural-gap supervision protocol"
```
