"""Single-node trainer for deployment-matched JAXA gradient priors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from Data_Imputation.models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

from . import PROTOCOL_VERSION
from .config import RunConfig, prepare_new_output_dir
from .dataset import DeploymentMatchedJAXADataset, build_training_cloud_bank
from .gradient_stats import load_gradient_stats
from .losses import HybridLossWeights, hybrid_loss
from .metrics import (
    EpochMetricAccumulator,
    local_predicted_outlier_terms,
    natural_exceedance_terms,
    seam_exceedance_terms,
    supervised_boundary_error_terms,
    supervised_error_terms,
    supervised_gradient_error_terms,
)


_METRIC_STATE_NAMES = (
    "absolute_error",
    "squared_error",
    "gradient_error",
    "boundary_error",
    "natural_exceedance",
    "seam_exceedance",
    "local_outlier",
    "low_energy_ratio",
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def distributed_context_from_env() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid torchrun rank environment")
    if world_size > 1 and not all(
        name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    ):
        raise RuntimeError("distributed training requires the torchrun rank environment")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def _initialize_process_group(
    context: DistributedContext, requested_device: str
) -> torch.device:
    if context.enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA")
        torch.cuda.set_device(context.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", context.local_rank)
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier(device_ids=[context.local_rank])


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def monthly_upper_for(
    months: torch.Tensor,
    gradient_stats: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    month_values = [int(value) for value in months.detach().cpu().reshape(-1).tolist()]
    if any(not 1 <= month <= 12 for month in month_values):
        raise ValueError(f"invalid month values: {month_values}")
    thresholds = [
        float(gradient_stats["months"][str(month)]["edge"]["p99_9"])
        for month in month_values
    ]
    return torch.tensor(thresholds, device=device, dtype=torch.float32)


def require_finite_loss(
    total: torch.Tensor,
    components: dict[str, torch.Tensor],
) -> None:
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite total loss: {total.detach().cpu().item()}")
    for name, value in components.items():
        if not torch.isfinite(value):
            raise FloatingPointError(
                f"non-finite {name} loss: {value.detach().cpu().item()}"
            )


def evenly_spaced_indices(length: int, limit: int | None) -> list[int] | None:
    if limit is None or limit >= length:
        return None
    if length <= 0 or limit <= 0:
        raise ValueError("dataset length and subset limit must be positive")
    if limit == 1:
        return [0]
    return np.linspace(0, length - 1, num=limit, dtype=np.int64).tolist()


def rank_strided_indices(length: int, rank: int, world_size: int) -> list[int]:
    """Shard indices without padding or duplication for exact validation."""
    if length < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid length, rank, or world size")
    return list(range(rank, length, world_size))


def _all_reduce_scalar_mapping(
    values: dict[str, float],
    device: torch.device,
    *,
    all_reduce=None,
) -> dict[str, float]:
    """Sum a scalar mapping in one stable-order collective."""
    if not values:
        return {}
    keys = sorted(values)
    tensor = torch.tensor(
        [float(values[key]) for key in keys], device=device, dtype=torch.float64
    )
    reduce_fn = all_reduce
    if reduce_fn is None and dist.is_available() and dist.is_initialized():
        reduce_fn = dist.all_reduce
    if reduce_fn is not None:
        reduce_fn(tensor)
    reduced = tensor.detach().cpu().tolist()
    return dict(zip(keys, reduced))


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    indices = evenly_spaced_indices(len(dataset), limit)
    return dataset if indices is None else Subset(dataset, indices)


def build_datasets(
    config: RunConfig,
    *,
    cloud_bank: np.ndarray,
) -> tuple[Dataset, Dataset]:
    common = dict(
        window_size=30,
        mean=config.norm_mean,
        std=config.norm_std,
        cloud_bank=cloud_bank,
        mask_ratio_range=(config.mask_ratio_low, config.mask_ratio_high),
    )
    train = DeploymentMatchedJAXADataset(
        config.data_dir,
        series_ids=config.train_series,
        seed=config.train_seed,
        vary_masks_by_epoch=True,
        **common,
    )
    valid = DeploymentMatchedJAXADataset(
        config.data_dir,
        series_ids=config.valid_series,
        seed=config.valid_seed,
        vary_masks_by_epoch=False,
        **common,
    )
    return _limit_dataset(train, config.train_limit), _limit_dataset(valid, config.valid_limit)


def build_data_loaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    config: RunConfig,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    train_sampler = None
    validation_dataset: Dataset = valid_dataset
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.train_seed,
            drop_last=True,
        )
        validation_dataset = Subset(
            valid_dataset,
            rank_strided_indices(len(valid_dataset), rank, world_size),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        # Workers must inherit the new dataset epoch before every pass.
        persistent_workers=False,
    )
    valid_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, valid_loader, train_sampler


def _file_identity(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_checkpoint_metadata(
    config: RunConfig,
    *,
    gradient_stats: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "epoch": int(epoch),
        "train_series": list(config.train_series),
        "valid_series": list(config.valid_series),
        "test_series": list(config.test_series),
        "gradient_stats_sha256": gradient_stats["sha256"],
        "loss_weights": {
            "gradient": config.lambda_grad,
            "boundary": config.lambda_boundary,
            "high": config.lambda_high,
            "seam": config.lambda_seam,
            "seam_continuity": config.lambda_seam_continuity,
            "low": config.lambda_low,
            "low_energy_beta": config.low_energy_beta,
        },
        "validation_metrics": validation_metrics,
        "normalization": {"mean": config.norm_mean, "std": config.norm_std},
        "distributed": {
            "world_size": config.world_size,
            "per_rank_batch_size": config.batch_size,
            "global_batch_size": config.batch_size * config.world_size,
        },
        "resolved_config": config.to_dict(),
    }


def write_run_manifests(
    config: RunConfig,
    *,
    gradient_stats: dict[str, Any],
) -> Path:
    output = prepare_new_output_dir(config.output_dir)
    (output / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    source = {
        "protocol_version": PROTOCOL_VERSION,
        "pretrained_checkpoint": _file_identity(config.pretrained_path),
        "gradient_statistics": _file_identity(config.gradient_stats_path),
        "gradient_stats_sha256": gradient_stats["sha256"],
        "series_split": {
            "train": list(config.train_series),
            "validation": list(config.valid_series),
            "test": list(config.test_series),
        },
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n"
    )
    return output


def _base_dataset(dataset: Dataset) -> DeploymentMatchedJAXADataset:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(base, DeploymentMatchedJAXADataset):
        raise TypeError("unexpected dataset wrapper")
    return base


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def _weights(config: RunConfig) -> HybridLossWeights:
    return HybridLossWeights(
        gradient=config.lambda_grad,
        boundary=config.lambda_boundary,
        high=config.lambda_high,
        seam=config.lambda_seam,
        seam_continuity=config.lambda_seam_continuity,
        low=config.lambda_low,
        low_energy_beta=config.low_energy_beta,
    )


def _forward_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: RunConfig,
    gradient_stats: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    sst_sequence = batch["input_sst_seq"].float()
    mask_sequence = batch["mask_seq"].float()
    target = batch["target_sst"].unsqueeze(1).float()
    target_proxy = batch["target_proxy_sst"].unsqueeze(1).float()
    raw_prediction = model(sst_sequence, mask_sequence)
    upper = monthly_upper_for(batch["month"], gradient_stats, raw_prediction.device)
    return hybrid_loss(
        raw_prediction=raw_prediction,
        last_input=sst_sequence[:, -1:, :, :],
        target=target,
        target_proxy=target_proxy,
        natural_mask=batch["natural_mask"],
        artificial_mask=batch["artificial_mask"],
        original_observed_mask=batch["original_obs_mask"],
        land_mask=batch["land_mask"],
        monthly_upper=upper,
        norm_std=config.norm_std,
        weights=_weights(config),
        lowpass_window=config.lowpass_window,
        energy_window=config.energy_window,
        min_natural_support=config.min_natural_support,
    )


def _add_metrics(
    accumulator: EpochMetricAccumulator,
    composed: torch.Tensor,
    batch: dict[str, Any],
    config: RunConfig,
    gradient_stats: dict[str, Any],
    counts: dict[str, torch.Tensor],
) -> None:
    target = batch["target_sst"].unsqueeze(1).float()
    upper = monthly_upper_for(batch["month"], gradient_stats, composed.device)
    errors = supervised_error_terms(
        composed, target, batch["artificial_mask"], norm_std=config.norm_std
    )
    accumulator.add(
        "absolute_error", numerator=errors["absolute_sum"], count=errors["count"]
    )
    accumulator.add(
        "squared_error", numerator=errors["squared_sum"], count=errors["count"]
    )
    gradient_sum, gradient_count = supervised_gradient_error_terms(
        composed,
        target,
        batch["artificial_mask"],
        batch["original_obs_mask"],
        norm_std=config.norm_std,
    )
    boundary_sum, boundary_count = supervised_boundary_error_terms(
        composed,
        target,
        batch["artificial_mask"],
        batch["original_obs_mask"],
        norm_std=config.norm_std,
    )
    natural_exceed, natural_count = natural_exceedance_terms(
        composed,
        batch["natural_mask"],
        batch["land_mask"],
        upper,
        norm_std=config.norm_std,
    )
    seam_exceed, seam_count = seam_exceedance_terms(
        composed,
        batch["natural_mask"],
        batch["original_obs_mask"],
        batch["artificial_mask"],
        batch["land_mask"],
        upper,
        norm_std=config.norm_std,
    )
    outlier_sum, outlier_count = local_predicted_outlier_terms(
        composed,
        batch["natural_mask"],
        batch["land_mask"],
        upper,
        norm_std=config.norm_std,
    )
    accumulator.add("gradient_error", numerator=gradient_sum, count=gradient_count)
    accumulator.add("boundary_error", numerator=boundary_sum, count=boundary_count)
    accumulator.add(
        "natural_exceedance", numerator=natural_exceed, count=natural_count
    )
    accumulator.add("seam_exceedance", numerator=seam_exceed, count=seam_count)
    accumulator.add("local_outlier", numerator=outlier_sum, count=outlier_count)
    low_count = counts["low"]
    accumulator.add(
        "low_energy_ratio",
        numerator=counts["low_energy_ratio"] * low_count,
        count=low_count,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    config: RunConfig,
    gradient_stats: dict[str, Any],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    metrics = EpochMetricAccumulator()
    metrics.numerators = dict.fromkeys(_METRIC_STATE_NAMES, 0.0)
    metrics.counts = dict.fromkeys(_METRIC_STATE_NAMES, 0.0)
    component_sums = {
        name: 0.0
        for name in (
            "mse",
            "gradient",
            "boundary",
            "high",
            "seam",
            "seam_continuity",
            "low",
        )
    }
    component_counts = dict.fromkeys(component_sums, 0.0)
    total_loss_sum = 0.0
    batch_count = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = _to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            total, components, counts, composed = _forward_loss(
                model, batch, config, gradient_stats
            )
            require_finite_loss(total, components)
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += float(total.detach().cpu())
            batch_count += 1
            for name, value in components.items():
                count = float(counts[name].detach().cpu())
                component_sums[name] += float(value.detach().cpu()) * count
                component_counts[name] += count
            _add_metrics(metrics, composed.detach(), batch, config, gradient_stats, counts)

    metrics.numerators = _all_reduce_scalar_mapping(metrics.numerators, device)
    metrics.counts = _all_reduce_scalar_mapping(metrics.counts, device)
    component_sums = _all_reduce_scalar_mapping(component_sums, device)
    component_counts = _all_reduce_scalar_mapping(component_counts, device)
    totals = _all_reduce_scalar_mapping(
        {"total_loss_sum": total_loss_sum, "batch_count": float(batch_count)},
        device,
    )
    total_loss_sum = totals["total_loss_sum"]
    batch_count = int(totals["batch_count"])
    if batch_count == 0:
        raise RuntimeError("data loader produced no batches")
    result: dict[str, Any] = metrics.finalize()
    result["total_loss"] = total_loss_sum / batch_count
    result["loss_components"] = {
        name: (
            component_sums[name] / component_counts[name]
            if component_counts[name] > 0
            else None
        )
        for name in component_sums
    }
    result["valid_counts"] = component_counts
    result["batches"] = batch_count
    return result


def _load_model(config: RunConfig, device: torch.device) -> torch.nn.Module:
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6,
        use_cbam=True,
    )
    checkpoint = torch.load(
        config.pretrained_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError("pretrained checkpoint has no model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    return model.to(device)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        **metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }


def _update_best_link(epoch_path: Path, best_path: Path) -> None:
    temporary = best_path.with_suffix(best_path.suffix + ".new")
    if temporary.exists():
        temporary.unlink()
    os.link(epoch_path, temporary)
    os.replace(temporary, best_path)


def prune_epoch_checkpoints(output_dir: Path | str, *, keep: int) -> list[Path]:
    """Remove old epoch path names while preserving any best hard-link inode."""
    if keep < 0:
        raise ValueError("keep must be non-negative")
    paths = sorted(Path(output_dir).glob("checkpoint_epoch_*.pth"))
    removed = paths if keep == 0 else paths[:-keep]
    for path in removed:
        path.unlink()
    return removed


def train(config: RunConfig) -> Path:
    context = distributed_context_from_env()
    config = replace(config, world_size=context.world_size)
    config.validate(require_inputs=True)
    device = _initialize_process_group(context, config.device)
    try:
        _seed_everything(config.train_seed + context.rank)
        gradient_stats = load_gradient_stats(config.gradient_stats_path)
        if context.is_main:
            output = write_run_manifests(config, gradient_stats=gradient_stats)
        else:
            output = config.output_dir
        _barrier(context)

        cloud_bank = build_training_cloud_bank(
            config.data_dir,
            config.train_series,
            max_donors=400,
            seed=config.train_seed,
        )
        train_dataset, valid_dataset = build_datasets(config, cloud_bank=cloud_bank)
        train_loader, valid_loader, train_sampler = build_data_loaders(
            train_dataset,
            valid_dataset,
            config,
            device,
            rank=context.rank,
            world_size=context.world_size,
        )
        base_model = _load_model(config, device)
        model: torch.nn.Module = base_model
        if context.enabled:
            model = DDP(
                base_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        best_mae = float("inf")
        metrics_path = output / "metrics.jsonl"
        for epoch in range(config.epochs):
            _base_dataset(train_dataset).set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            started = time.monotonic()
            train_metrics = run_epoch(
                model,
                train_loader,
                config,
                gradient_stats,
                device,
                optimizer=optimizer,
            )
            valid_metrics = run_epoch(
                base_model,
                valid_loader,
                config,
                gradient_stats,
                device,
                optimizer=None,
            )
            scheduler.step()
            peak_gib = 0.0
            if device.type == "cuda":
                peak = torch.tensor(
                    torch.cuda.max_memory_allocated(device) / 1024**3,
                    device=device,
                    dtype=torch.float64,
                )
                if context.enabled:
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                peak_gib = float(peak.cpu())
            record = {
                "epoch": epoch + 1,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "peak_cuda_memory_gib": peak_gib,
                "train": train_metrics,
                "validation": valid_metrics,
            }
            if context.is_main:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

                metadata = build_checkpoint_metadata(
                    config,
                    gradient_stats=gradient_stats,
                    epoch=epoch + 1,
                    validation_metrics=valid_metrics,
                )
                prune_epoch_checkpoints(
                    output, keep=max(config.max_epoch_checkpoints - 1, 0)
                )
                epoch_path = output / f"checkpoint_epoch_{epoch + 1:03d}.pth"
                torch.save(
                    _checkpoint_payload(base_model, optimizer, scheduler, metadata),
                    epoch_path,
                )
                validation_mae = valid_metrics["mae"]
                if validation_mae is not None and validation_mae < best_mae:
                    best_mae = float(validation_mae)
                    _update_best_link(epoch_path, output / "best_model.pth")
                print(json.dumps(record, indent=2, sort_keys=True), flush=True)
            _barrier(context)

        return output
    finally:
        if context.enabled and dist.is_initialized():
            dist.destroy_process_group()


def _default_paths() -> tuple[Path, Path, Path, Path]:
    data_imputation = Path(__file__).resolve().parents[2]
    experiments = data_imputation / "experiments"
    return (
        experiments / "hourly_data" / "h12",
        experiments / "gradient_prior_v1",
        experiments / "ostia_pretrain_clean" / "best_model.pth",
        experiments / "gradient_prior_v1" / "stats" / "h12_train_monthly_v2.json",
    )


def parse_config(argv: list[str] | None = None) -> RunConfig:
    data_default, output_root_default, checkpoint_default, stats_default = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--hour", type=int, default=12)
    parser.add_argument("--data-dir", type=Path, default=data_default)
    parser.add_argument("--output-root", type=Path, default=output_root_default)
    parser.add_argument("--pretrained", type=Path, default=checkpoint_default)
    parser.add_argument("--gradient-stats", type=Path, default=stats_default)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--max-epoch-checkpoints", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lambda-grad", type=float, default=0.2)
    parser.add_argument("--lambda-boundary", type=float, default=0.1)
    parser.add_argument("--lambda-high", type=float, default=0.05)
    parser.add_argument("--lambda-seam", type=float, default=0.05)
    parser.add_argument("--lambda-seam-continuity", type=float, default=0.05)
    parser.add_argument("--lambda-low", type=float, default=0.02)
    parser.add_argument("--low-energy-beta", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not args.experiment_name or Path(args.experiment_name).name != args.experiment_name:
        raise ValueError("experiment-name must be one non-empty path component")
    hour_data = args.data_dir
    if args.data_dir == data_default and args.hour != 12:
        hour_data = data_default.parent / f"h{args.hour:02d}"
    config = RunConfig(
        data_dir=hour_data,
        output_dir=args.output_root / args.experiment_name,
        pretrained_path=args.pretrained,
        gradient_stats_path=args.gradient_stats,
        experiment_name=args.experiment_name,
        hour=args.hour,
        epochs=args.epochs,
        batch_size=args.batch_size,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        lambda_grad=args.lambda_grad,
        lambda_boundary=args.lambda_boundary,
        lambda_high=args.lambda_high,
        lambda_seam=args.lambda_seam,
        lambda_seam_continuity=args.lambda_seam_continuity,
        lambda_low=args.lambda_low,
        low_energy_beta=args.low_energy_beta,
        device=args.device,
        train_limit=args.train_limit,
        valid_limit=args.valid_limit,
        max_epoch_checkpoints=args.max_epoch_checkpoints,
    )
    config.validate()
    return config


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
