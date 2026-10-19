"""Evaluate old or new FNO-CBAM checkpoints under one fixed Series 7 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import PROTOCOL_VERSION
from .config import RunConfig, TRAIN_SERIES, VALID_SERIES
from .dataset import build_training_cloud_bank
from .gradient_stats import load_gradient_stats
from .train import _load_model, build_validation_datasets, run_epoch


def write_evaluation_result(path: Path | str, result: dict[str, object]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation result: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {"protocol_version": PROTOCOL_VERSION, **result}
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    data_dir: Path,
    gradient_stats_path: Path,
    output: Path,
    device_name: str,
    sample_limit: int | None,
    batch_size: int,
    num_workers: int,
    seed: int = 123,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = RunConfig(
        data_dir=data_dir,
        output_dir=output.parent,
        pretrained_path=checkpoint,
        gradient_stats_path=gradient_stats_path,
        experiment_name=output.stem,
        epochs=1,
        batch_size=batch_size,
        num_workers=num_workers,
        valid_seed=seed,
        device=device_name,
        valid_limit=sample_limit,
    )
    config.validate(require_inputs=True)
    stats = load_gradient_stats(gradient_stats_path)
    cloud_bank = build_training_cloud_bank(
        data_dir, TRAIN_SERIES, max_donors=400, seed=config.train_seed
    )
    datasets = build_validation_datasets(config, cloud_bank=cloud_bank)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    model = _load_model(config, device)
    metrics = {
        name: run_epoch(model, loader, config, stats, device, optimizer=None)
        for name, loader in loaders.items()
    }
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "gradient_statistics": str(gradient_stats_path.resolve()),
        "gradient_stats_sha256": stats["sha256"],
        "series": list(VALID_SERIES),
        "seed": seed,
        "sample_limit": sample_limit,
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "metrics": metrics,
    }
    write_evaluation_result(output, result)
    return result


def main() -> None:
    data_imputation = Path(__file__).resolve().parents[2]
    experiments = data_imputation / "experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=experiments / "hourly_data" / "h12"
    )
    parser.add_argument(
        "--gradient-stats",
        type=Path,
        default=experiments
        / "gradient_prior_v2_natural_gap"
        / "stats"
        / "h12_train_monthly_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        gradient_stats_path=args.gradient_stats,
        output=args.output,
        device_name=args.device,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
