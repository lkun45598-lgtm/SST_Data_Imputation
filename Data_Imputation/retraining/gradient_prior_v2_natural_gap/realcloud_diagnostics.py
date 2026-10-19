"""No-reference Series 8 diagnostics for the exact deployed composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import PROTOCOL_VERSION
from .config import RunConfig, TEST_SERIES
from .dataset import DeploymentMatchedJAXADataset, no_artificial_mask
from .gradient_stats import load_gradient_stats
from .losses import (
    boundary_transition_composition,
    natural_deployment_seam_masks,
    natural_edge_masks,
)
from .metrics import (
    edge_difference_values,
    local_predicted_outlier_terms,
    seam_exceedance_terms,
)
from .train import _load_model, evenly_spaced_indices, monthly_upper_for


class ExactValueAccumulator:
    """Accumulate every finite value before computing global quantiles."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def add(self, values: torch.Tensor | np.ndarray) -> None:
        if isinstance(values, torch.Tensor):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)
        finite = np.asarray(array, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            self._chunks.append(finite)

    def summary(self) -> dict[str, int | float | None]:
        if not self._chunks:
            return {"count": 0, "p50": None, "p95": None, "p99": None, "p99_9": None}
        values = np.concatenate(self._chunks)
        return {
            "count": int(values.size),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "p99_9": float(np.quantile(values, 0.999)),
        }


def write_realcloud_diagnostics(path: Path | str, document: dict[str, object]) -> None:
    """Write a single immutable no-reference diagnostics document."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite real-cloud diagnostics: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"protocol_version": PROTOCOL_VERSION, **document}, indent=2, sort_keys=True)
        + "\n"
    )


def _file_identity(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def diagnose_checkpoint(
    *,
    checkpoint: Path,
    data_dir: Path,
    gradient_stats_path: Path,
    output: Path,
    device_name: str,
    frame_limit: int | None,
    batch_size: int,
    num_workers: int,
    seed: int = 123,
) -> dict[str, object]:
    """Run no-reference Series 8 diagnostics without artificial masking."""
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
    )
    config.validate(require_inputs=True)
    stats = load_gradient_stats(gradient_stats_path)
    dataset = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=TEST_SERIES,
        window_size=30,
        mean=config.norm_mean,
        std=config.norm_std,
        seed=seed,
        vary_masks_by_epoch=False,
        artificial_mask_provider=no_artificial_mask,
        mask_stream="realcloud_diagnostics",
    )
    indices = evenly_spaced_indices(len(dataset), frame_limit)
    selected = dataset if indices is None else Subset(dataset, indices)
    loader = DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = _load_model(config, device)
    model.eval()
    natural_interior = ExactValueAccumulator()
    deployment_seam = ExactValueAccumulator()
    seam_exceedance = 0.0
    seam_count = 0.0
    outlier_sum = 0.0
    outlier_count = 0.0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            raw_prediction = model(batch["input_sst_seq"].float(), batch["mask_seq"].float())
            composed = boundary_transition_composition(
                raw_prediction,
                batch["input_sst_seq"][:, -1:, :, :].float(),
                batch["target_proxy_sst"].unsqueeze(1).float(),
                batch["natural_mask"],
                batch["land_mask"],
            )
            natural_y, natural_x = natural_edge_masks(batch["natural_mask"], batch["land_mask"])
            seam_y, seam_x = natural_deployment_seam_masks(
                batch["natural_mask"], batch["artificial_mask"], batch["land_mask"]
            )
            natural_interior.add(
                edge_difference_values(
                    composed, natural_y, natural_x, norm_std=config.norm_std
                )
            )
            deployment_seam.add(
                edge_difference_values(composed, seam_y, seam_x, norm_std=config.norm_std)
            )
            upper = monthly_upper_for(batch["month"], stats, device)
            exceed, seam_edges = seam_exceedance_terms(
                composed,
                batch["natural_mask"],
                batch["artificial_mask"],
                batch["land_mask"],
                upper,
                norm_std=config.norm_std,
            )
            outliers, candidates = local_predicted_outlier_terms(
                composed,
                batch["natural_mask"],
                batch["land_mask"],
                upper,
                norm_std=config.norm_std,
            )
            seam_exceedance += float(exceed.detach().cpu())
            seam_count += float(seam_edges.detach().cpu())
            outlier_sum += float(outliers.detach().cpu())
            outlier_count += float(candidates.detach().cpu())

    files = [data_dir / f"jaxa_knn_filled_{series_id:02d}.h5" for series_id in TEST_SERIES]
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "gradient_statistics": str(gradient_stats_path.resolve()),
        "gradient_stats_sha256": stats["sha256"],
        "data_files": [_file_identity(path) for path in files],
        "series": list(TEST_SERIES),
        "artificial_masking": False,
        "seed": int(seed),
        "frame_limit": frame_limit,
        "frame_count": len(selected),
        "natural_interior_adjacent_difference_K_per_pixel": natural_interior.summary(),
        "deployment_seam_adjacent_difference_K_per_pixel": deployment_seam.summary(),
        "deployment_seam_exceedance_rate": (
            None if seam_count == 0 else seam_exceedance / seam_count
        ),
        "natural_local_outlier_rate": (
            None if outlier_count == 0 else outlier_sum / outlier_count
        ),
    }
    write_realcloud_diagnostics(output, result)
    return result


def main() -> None:
    data_imputation = Path(__file__).resolve().parents[2]
    experiments = data_imputation / "experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=experiments / "hourly_data" / "h12")
    parser.add_argument(
        "--gradient-stats",
        type=Path,
        default=experiments / "gradient_prior_v2_natural_gap" / "stats" / "h12_train_monthly_v3.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    result = diagnose_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        gradient_stats_path=args.gradient_stats,
        output=args.output,
        device_name=args.device,
        frame_limit=args.frame_limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
