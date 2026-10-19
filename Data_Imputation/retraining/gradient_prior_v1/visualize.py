"""Qualitative figures for the trained gradient-prior FNO-CBAM model.

The Series 7 panel uses real-cloud artificial masks and has pointwise proxy
targets on the withheld observed pixels. The Series 8 panel follows the
deployment path and is explicitly no-reference: its KNN-filled field is shown
only as an input proxy, never as ground truth. No Gaussian post-processing or
hard clipping is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import RunConfig, TRAIN_SERIES, VALID_SERIES
from .dataset import DeploymentMatchedJAXADataset, build_training_cloud_bank
from .gradient_stats import load_gradient_stats
from .losses import output_composition
from .train import _load_model


MASK_CMAP = ListedColormap(["#f7f7f7", "#777777", "#e67e22", "#3f3f3f"])
SST_CMAP = plt.get_cmap("RdYlBu_r")
ERROR_CMAP = plt.get_cmap("magma")


def compose_denormalized_prediction(
    raw_prediction: torch.Tensor,
    last_input: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    land_mask: torch.Tensor,
    *,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Compose the normalized prediction, then return physical SST in kelvin."""
    if std <= 0:
        raise ValueError("std must be positive")
    composed = output_composition(
        raw_prediction, last_input, reconstruction_mask, land_mask
    )
    return composed * float(std) + float(mean)


def mask_categories(
    natural_mask: np.ndarray,
    artificial_mask: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Encode observed=0, natural=1, artificial=2, land=3 for plotting."""
    natural = np.asarray(natural_mask, dtype=bool)
    artificial = np.asarray(artificial_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if not (natural.shape == artificial.shape == land.shape):
        raise ValueError("mask shapes must match")
    if np.any(natural & artificial):
        raise ValueError("natural and artificial masks overlap")
    categories = np.zeros(natural.shape, dtype=np.int8)
    categories[natural & ~land] = 1
    categories[artificial & ~land] = 2
    categories[land] = 3
    return categories


def natural_gap_fraction(natural_mask: np.ndarray, land_mask: np.ndarray) -> float:
    """Return natural-gap area divided by the available ocean area."""
    natural = np.asarray(natural_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if natural.shape != land.shape:
        raise ValueError("mask shapes must match")
    ocean = ~land
    return float((natural & ocean).sum() / max(int(ocean.sum()), 1))


def _sample_to_batch(sample: dict[str, object]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in ("input_sst_seq", "mask_seq", "reconstruction_mask", "land_mask"):
        value = sample[key]
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.from_numpy(np.asarray(value))
        batch[key] = tensor.unsqueeze(0)
    return batch


def predict_sample(
    model: torch.nn.Module,
    sample: dict[str, object],
    *,
    device: torch.device,
    mean: float,
    std: float,
) -> dict[str, np.ndarray]:
    batch = _sample_to_batch(sample)
    sst_sequence = batch["input_sst_seq"].float().to(device)
    mask_sequence = batch["mask_seq"].float().to(device)
    reconstruction = batch["reconstruction_mask"].to(device)
    land = batch["land_mask"].to(device)
    with torch.inference_mode():
        raw = model(sst_sequence, mask_sequence)
        composed = compose_denormalized_prediction(
            raw,
            sst_sequence[:, -1:, :, :],
            reconstruction,
            land,
            mean=mean,
            std=std,
        )
    return {
        "prediction": composed[0, 0].cpu().numpy(),
        "input": (sst_sequence[0, -1] * std + mean).cpu().numpy(),
    }


def _load_formal_model(
    checkpoint: Path,
    data_dir: Path,
    stats_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, RunConfig]:
    config = RunConfig(
        data_dir=data_dir,
        output_dir=checkpoint.parent,
        pretrained_path=checkpoint,
        gradient_stats_path=stats_path,
        experiment_name=checkpoint.stem,
        epochs=1,
        device=str(device),
    )
    model = _load_model(config, device)
    model.eval()
    return model, config


def _style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#333333")


def _select_indices(length: int, requested: list[int] | None) -> list[int]:
    if requested:
        if any(index < 0 or index >= length for index in requested):
            raise ValueError(f"requested index outside dataset length {length}")
        return requested
    if length < 3:
        return list(range(length))
    return [length // 8, length // 2, 7 * length // 8]


def _plot_series7(
    model: torch.nn.Module,
    dataset: DeploymentMatchedJAXADataset,
    indices: list[int],
    *,
    config: RunConfig,
    device: torch.device,
    output: Path,
) -> list[dict[str, object]]:
    records = []
    for index in indices:
        sample = dataset[index]
        prediction = predict_sample(
            model,
            sample,
            device=device,
            mean=config.norm_mean,
            std=config.norm_std,
        )
        natural = np.asarray(sample["natural_mask"], dtype=bool)
        artificial = np.asarray(sample["artificial_mask"], dtype=bool)
        land = np.asarray(sample["land_mask"], dtype=bool)
        categories = mask_categories(natural, artificial, land)
        target = np.asarray(sample["target_sst"], dtype=np.float32) * config.norm_std + config.norm_mean
        error = np.where(artificial, np.abs(prediction["prediction"] - target), np.nan)
        records.append({
            "index": index,
            "timestamp": str(sample["timestamp"]),
            "prediction": prediction["prediction"],
            "input": prediction["input"],
            "target": target,
            "error": error,
            "categories": categories,
            "artificial_fraction": float(artificial.sum() / max((~land).sum(), 1)),
        })

    finite_fields = [record[key][np.isfinite(record[key])] for record in records for key in ("prediction", "target")]
    vmin, vmax = np.percentile(np.concatenate(finite_fields), [2, 98])
    error_values = np.concatenate([record["error"][np.isfinite(record["error"])] for record in records])
    error_max = float(np.percentile(error_values, 98)) if error_values.size else 1.0

    fig, axes = plt.subplots(len(records), 5, figsize=(12.5, 2.7 * len(records)), squeeze=False)
    titles = ["Input after masking", "Mask category", "FNO-CBAM reconstruction", "Proxy target", "Absolute error on artificial mask"]
    for row, record in enumerate(records):
        fields = [
            np.where(record["categories"] == 0, record["input"], np.nan),
            record["categories"],
            record["prediction"],
            record["target"],
            record["error"],
        ]
        for col, field in enumerate(fields):
            ax = axes[row, col]
            if col == 1:
                image = ax.imshow(field, origin="lower", cmap=MASK_CMAP, vmin=-0.5, vmax=3.5)
            elif col == 4:
                image = ax.imshow(field, origin="lower", cmap=ERROR_CMAP, vmin=0, vmax=max(error_max, 1e-6))
            else:
                image = ax.imshow(field, origin="lower", cmap=SST_CMAP, vmin=vmin, vmax=vmax)
            _style_axes(ax, titles[col] if row == 0 else "")
            if col == 0:
                ax.set_ylabel(f"frame {record['index']}\n{record['timestamp']}", fontsize=8)
        axes[row, 0].text(
            0.02,
            0.02,
            f"artificial={record['artificial_fraction']:.1%}",
            transform=axes[row, 0].transAxes,
            fontsize=7,
            color="#111111",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 2, "edgecolor": "none"},
        )
    fig.suptitle("Series 7 controlled real-cloud masking: trained FNO-CBAM", fontsize=11)
    fig.text(0.5, 0.01, "Mask: observed / natural gap / artificial real-cloud gap / land. Error is evaluated only on artificial pixels.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [
        {
            "index": record["index"],
            "timestamp": record["timestamp"],
            "artificial_fraction": record["artificial_fraction"],
            "mae_artificial": float(np.nanmean(record["error"])),
            "max_abs_error_artificial": float(np.nanmax(record["error"])),
        }
        for record in records
    ]


def _real_series8_sample(handle: h5py.File, index: int, *, mean: float, std: float) -> dict[str, object]:
    land = np.asarray(handle["land_mask"][:], dtype=bool)
    ocean = ~land
    start = max(0, index - 29)
    sst_sequence = np.asarray(handle["sst_data"][start : index + 1], dtype=np.float32)
    mask_sequence = np.asarray(handle["original_missing_mask"][start : index + 1], dtype=bool)
    padding = 30 - sst_sequence.shape[0]
    if padding:
        sst_sequence = np.concatenate([np.repeat(sst_sequence[:1], padding, axis=0), sst_sequence], axis=0)
        mask_sequence = np.concatenate([np.repeat(mask_sequence[:1], padding, axis=0), mask_sequence], axis=0)
    natural = np.asarray(handle["original_missing_mask"][index], dtype=bool) & ocean
    target_proxy = np.asarray(handle["sst_data"][index], dtype=np.float32)
    target_input = sst_sequence[-1].copy()
    target_input[natural] = mean
    sst_sequence[-1] = target_input
    mask_sequence[-1] = natural
    normalized = np.nan_to_num((sst_sequence - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    timestamp = handle["timestamps"][index]
    if isinstance(timestamp, bytes):
        timestamp = timestamp.decode("utf-8")
    return {
        "input_sst_seq": normalized,
        "mask_seq": mask_sequence,
        "reconstruction_mask": natural,
        "natural_mask": natural,
        "artificial_mask": np.zeros_like(natural),
        "land_mask": land,
        "target_proxy": target_proxy,
        "timestamp": str(timestamp),
        "index": index,
    }


def _plot_series8(
    model: torch.nn.Module,
    data_path: Path,
    indices: list[int],
    *,
    config: RunConfig,
    device: torch.device,
    output: Path,
) -> list[dict[str, object]]:
    records = []
    with h5py.File(data_path, "r") as handle:
        for index in indices:
            sample = _real_series8_sample(handle, index, mean=config.norm_mean, std=config.norm_std)
            prediction = predict_sample(model, sample, device=device, mean=config.norm_mean, std=config.norm_std)
            natural = np.asarray(sample["natural_mask"], dtype=bool)
            land = np.asarray(sample["land_mask"], dtype=bool)
            categories = mask_categories(natural, np.zeros_like(natural), land)
            records.append({
                "index": index,
                "timestamp": sample["timestamp"],
                "input": prediction["input"],
                "prediction": prediction["prediction"],
                "proxy": np.asarray(sample["target_proxy"], dtype=np.float32),
                "categories": categories,
                "natural_fraction": natural_gap_fraction(natural, land),
            })

    finite_fields = [record[key][np.isfinite(record[key])] for record in records for key in ("prediction", "proxy")]
    vmin, vmax = np.percentile(np.concatenate(finite_fields), [2, 98])
    fig, axes = plt.subplots(len(records), 4, figsize=(10.0, 2.7 * len(records)), squeeze=False)
    titles = ["Observed input", "Natural-gap mask", "FNO-CBAM reconstruction", "KNN-filled input proxy"]
    for row, record in enumerate(records):
        fields = [
            np.where(record["categories"] == 0, record["input"], np.nan),
            record["categories"],
            record["prediction"],
            np.where(record["categories"] != 3, record["proxy"], np.nan),
        ]
        for col, field in enumerate(fields):
            ax = axes[row, col]
            if col == 1:
                ax.imshow(field, origin="lower", cmap=MASK_CMAP, vmin=-0.5, vmax=3.5)
            else:
                ax.imshow(field, origin="lower", cmap=SST_CMAP, vmin=vmin, vmax=vmax)
            _style_axes(ax, titles[col] if row == 0 else "")
            if col == 0:
                ax.set_ylabel(f"frame {record['index']}\n{record['timestamp']}", fontsize=8)
        axes[row, 0].text(
            0.02,
            0.02,
            f"natural gap={record['natural_fraction']:.1%}",
            transform=axes[row, 0].transAxes,
            fontsize=7,
            color="#111111",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 2, "edgecolor": "none"},
        )
    fig.suptitle("Series 8 real JAXA gaps: deployment-path reconstruction", fontsize=11)
    fig.text(0.5, 0.01, "No pointwise truth is available under natural clouds; the KNN-filled panel is an input proxy, not ground truth. No Gaussian post-processing.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [
        {
            "index": record["index"],
            "timestamp": record["timestamp"],
            "natural_fraction": record["natural_fraction"],
        }
        for record in records
    ]


def generate_figures(
    *,
    checkpoint: Path,
    data_dir: Path,
    stats_path: Path,
    output_dir: Path,
    device_name: str,
    series7_indices: list[int] | None = None,
    series8_count: int = 3,
) -> dict[str, object]:
    if series8_count <= 0:
        raise ValueError("series8_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    load_gradient_stats(stats_path)
    model, config = _load_formal_model(checkpoint, data_dir, stats_path, device)

    cloud_bank = build_training_cloud_bank(data_dir, TRAIN_SERIES, max_donors=400, seed=config.train_seed)
    series7 = DeploymentMatchedJAXADataset(
        data_dir,
        series_ids=VALID_SERIES,
        window_size=30,
        mean=config.norm_mean,
        std=config.norm_std,
        seed=config.valid_seed,
        vary_masks_by_epoch=False,
        cloud_bank=cloud_bank,
        mask_ratio_range=(config.mask_ratio_low, config.mask_ratio_high),
    )
    indices7 = _select_indices(len(series7), series7_indices)
    series7_records = _plot_series7(
        model,
        series7,
        indices7,
        config=config,
        device=device,
        output=output_dir / "series7_controlled_reconstruction.png",
    )

    series8_path = data_dir / "jaxa_knn_filled_08.h5"
    with h5py.File(series8_path, "r") as handle:
        land = np.asarray(handle["land_mask"][:], dtype=bool)
        ocean = ~land
        observed = np.asarray(handle["original_obs_mask"][:], dtype=bool)
    cloud_fraction = ((~observed) & ocean).reshape(observed.shape[0], -1).sum(axis=1) / max(int(ocean.sum()), 1)
    candidates = [int(i) for i in range(29, len(cloud_fraction)) if 0.30 <= cloud_fraction[i] <= 0.80]
    if len(candidates) < series8_count:
        candidates = [int(i) for i in range(29, len(cloud_fraction)) if cloud_fraction[i] > 0.20]
    if len(candidates) < series8_count:
        raise RuntimeError("not enough real-gap Series 8 frames for visualization")
    positions = np.linspace(0, len(candidates) - 1, num=series8_count, dtype=int)
    indices8 = [candidates[int(position)] for position in positions]
    series8_records = _plot_series8(
        model,
        series8_path,
        indices8,
        config=config,
        device=device,
        output=output_dir / "series8_real_gap_reconstruction.png",
    )
    manifest = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": None,
        "data_dir": str(data_dir.resolve()),
        "no_gaussian_postprocessing": True,
        "series7": series7_records,
        "series8": series8_records,
        "series7_figure": "series7_controlled_reconstruction.png",
        "series8_figure": "series8_real_gap_reconstruction.png",
    }
    (output_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    data_imputation = Path(__file__).resolve().parents[2]
    experiments = data_imputation / "experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=experiments / "hourly_data" / "h12")
    parser.add_argument("--gradient-stats", type=Path, default=experiments / "gradient_prior_v1" / "stats" / "h12_train_monthly_v2.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--series7-indices", type=int, nargs="+")
    parser.add_argument("--series8-count", type=int, default=3)
    args = parser.parse_args()
    manifest = generate_figures(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        stats_path=args.gradient_stats,
        output_dir=args.output_dir,
        device_name=args.device,
        series7_indices=args.series7_indices,
        series8_count=args.series8_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
