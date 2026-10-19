"""Attribute real-cloud high-frequency structure to FNO output or proxy input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from scipy.ndimage import uniform_filter

from .visualize import _sample_to_batch, compose_denormalized_prediction
from .visualize_realcloud_style import (
    GRAD_CMAP,
    SST_CMAP,
    _default_paths,
    _draw_field,
    _make_sample,
    _load_formal_model,
    apply_paper_style,
    setup_geo_ax,
)


PROVENANCE_CMAP = ListedColormap(["#f8f1e3", "#d4956b", "#7fa7c4", "#d8c7a3"])


def _region_core(region: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    valid = np.asarray(region, dtype=bool) & np.asarray(ocean, dtype=bool)
    core = np.zeros_like(valid)
    core[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    return core


def _regional_highpass(
    field: np.ndarray,
    region: np.ndarray,
    ocean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(region, dtype=bool) & np.asarray(ocean, dtype=bool)
    values = np.where(valid, np.asarray(field, dtype=np.float32), 0.0)
    weights = valid.astype(np.float32)
    smooth_sum = uniform_filter(values, size=3, mode="constant", cval=0.0)
    smooth_weight = uniform_filter(weights, size=3, mode="constant", cval=0.0)
    smooth = smooth_sum / np.maximum(smooth_weight, 1e-6)
    highpass = np.abs(np.asarray(field, dtype=np.float32) - smooth)
    return highpass, _region_core(valid, ocean)


def _regional_gradient(field: np.ndarray, region: np.ndarray, ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float32)
    grad_y = np.zeros_like(values)
    grad_x = np.zeros_like(values)
    grad_y[1:-1, :] = 0.5 * (values[2:, :] - values[:-2, :])
    grad_x[:, 1:-1] = 0.5 * (values[:, 2:] - values[:, :-2])
    return np.sqrt(grad_y**2 + grad_x**2), _region_core(region, ocean)


def _regional_laplacian(field: np.ndarray, region: np.ndarray, ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float32)
    laplacian = np.zeros_like(values)
    laplacian[1:-1, 1:-1] = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4.0 * values[1:-1, 1:-1]
    )
    return np.abs(laplacian), _region_core(region, ocean)


def _quantiles(values: np.ndarray, valid: np.ndarray) -> dict[str, float | None]:
    selected = np.asarray(values, dtype=np.float32)[np.asarray(valid, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"p50": None, "p95": None, "p99": None}
    p50, p95, p99 = np.percentile(selected, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}


def _region_stats(field: np.ndarray, region: np.ndarray, ocean: np.ndarray) -> dict[str, object]:
    highpass, hp_valid = _regional_highpass(field, region, ocean)
    gradient, grad_valid = _regional_gradient(field, region, ocean)
    laplacian, lap_valid = _regional_laplacian(field, region, ocean)
    return {
        "pixel_count": int(np.asarray(region, dtype=bool).sum()),
        "highpass_abs_K": _quantiles(highpass, hp_valid),
        "gradient_K_per_pixel": _quantiles(gradient, grad_valid),
        "laplacian_abs_K_per_pixel2": _quantiles(laplacian, lap_valid),
    }


def _run_raw_prediction(
    model: torch.nn.Module,
    sample: dict[str, object],
    *,
    device: torch.device,
    mean: float,
    std: float,
) -> dict[str, np.ndarray]:
    batch = _sample_to_batch(sample)
    sequence = batch["input_sst_seq"].float().to(device)
    masks = batch["mask_seq"].float().to(device)
    reconstruction = batch["reconstruction_mask"].to(device)
    land = batch["land_mask"].to(device)
    with torch.inference_mode():
        raw_normalized = model(sequence, masks)
        composed = compose_denormalized_prediction(
            raw_normalized,
            sequence[:, -1:, :, :],
            reconstruction,
            land,
            mean=mean,
            std=std,
        )
    return {
        "raw": (raw_normalized[0, 0].cpu().numpy() * std + mean).astype(np.float32),
        "composed": composed[0, 0].cpu().numpy().astype(np.float32),
        "input": (sequence[0, -1].cpu().numpy() * std + mean).astype(np.float32),
    }


def _save_diagnostic_figure(
    *,
    output: Path,
    fields: dict[str, np.ndarray],
    ocean: np.ndarray,
    natural: np.ndarray,
    preserved: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    timestamp: str,
    frame: int,
    highpass_max: float,
) -> None:
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    fig, axes = plt.subplots(2, 4, figsize=(11.3, 5.9), layout="constrained")
    sst_values = np.concatenate(
        [fields[key][np.isfinite(fields[key])] - 273.15 for key in ("observed", "raw_natural", "composed")]
    )
    sst_min, sst_max = np.percentile(sst_values, [2, 98])
    top = [
        ("Observed input", fields["observed"] - 273.15),
        ("Raw FNO in\nnatural gap", fields["raw_natural"] - 273.15),
        ("Composed FNO\nfield", fields["composed"] - 273.15),
    ]
    for col, (title, field) in enumerate(top):
        _draw_field(
            axes[0, col],
            field,
            extent=extent,
            ocean=ocean,
            cmap=SST_CMAP,
            vmin=float(sst_min),
            vmax=float(sst_max),
            lon=lon,
            lat=lat,
            show_x=False,
            show_y=col == 0,
        )
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    provenance = np.full(ocean.shape, 3, dtype=np.int8)
    provenance[preserved & ocean] = 2
    provenance[natural & ocean] = 1
    provenance[fields["observed_mask"] & ocean] = 0
    axes[0, 3].imshow(
        provenance,
        extent=extent,
        origin="lower",
        cmap=PROVENANCE_CMAP,
        vmin=-0.5,
        vmax=3.5,
        aspect="equal",
        interpolation="nearest",
    )
    setup_geo_ax(axes[0, 3], lon, lat, draw_xlabel=False, draw_ylabel=False, step=2)
    axes[0, 3].set_title("Provenance mask", fontsize=10, fontweight="bold")
    axes[0, 3].legend(
        handles=[
            mpl.patches.Patch(color="#f8f1e3", label="observed"),
            mpl.patches.Patch(color="#d4956b", label="natural gap / FNO"),
            mpl.patches.Patch(color="#7fa7c4", label="preserved input"),
            mpl.patches.Patch(color="#d8c7a3", label="land"),
        ],
        loc="lower left",
        fontsize=7,
        framealpha=0.9,
    )

    bottom = [
        ("Raw FNO high-pass\n(natural gap)", fields["raw_natural_highpass"]),
        ("Composed high-pass\n(natural gap)", fields["composed_natural_highpass"]),
        ("Composed high-pass\n(preserved input)", fields["composed_preserved_highpass"]),
        ("Gradient magnitude\n(composed field)", fields["composed_gradient"]),
    ]
    bottom_images = []
    for col, (title, field) in enumerate(bottom):
        image = _draw_field(
            axes[1, col],
            field,
            extent=extent,
            ocean=ocean,
            cmap=GRAD_CMAP,
            vmin=0.0,
            vmax=highpass_max if col < 3 else float(np.nanpercentile(field, 98)),
            lon=lon,
            lat=lat,
            show_x=True,
            show_y=col == 0,
        )
        axes[1, col].set_title(title, fontsize=9, fontweight="bold")
        bottom_images.append(image)

    cbar_top = fig.colorbar(axes[0, 2].images[-1], ax=axes[0, :3].tolist(), location="right", shrink=0.82, pad=0.008, aspect=18)
    cbar_top.set_label(r"SST ($^\circ$C)", fontsize=10)
    cbar_bottom = fig.colorbar(bottom_images[0], ax=axes[1, :3].tolist(), location="right", shrink=0.82, pad=0.008, aspect=18)
    cbar_bottom.set_label("3x3 high-pass (K)", fontsize=10)
    axes[0, 0].text(-0.48, 0.5, "SST attribution", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    axes[1, 0].text(-0.48, 0.5, "High-frequency diagnostic", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle(
        f"Real-cloud attribution diagnostic: Series 8 frame {frame} ({timestamp})",
        fontsize=12.5,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render(
    *,
    checkpoint: Path,
    data_path: Path,
    stats_path: Path,
    output: Path,
    frame: int,
    device_name: str,
) -> dict[str, object]:
    apply_paper_style()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model, config = _load_formal_model(checkpoint, data_path.parent, stats_path, device)
    with h5py.File(data_path, "r") as handle:
        sample = _make_sample(handle, frame, mean=config.norm_mean, std=config.norm_std)
        lat = np.asarray(handle["latitude"][:], dtype=np.float32)
        lon = np.asarray(handle["longitude"][:], dtype=np.float32)

    fields = _run_raw_prediction(
        model,
        sample,
        device=device,
        mean=config.norm_mean,
        std=config.norm_std,
    )
    ocean = ~np.asarray(sample["land_mask"], dtype=bool)
    natural = np.asarray(sample["natural_mask"], dtype=bool) & ocean
    observed_mask = np.asarray(sample["observed_mask"], dtype=bool) & ocean
    preserved = ocean & ~natural
    proxy = np.asarray(sample["target_proxy"], dtype=np.float32)
    fields["observed"] = np.where(observed_mask, proxy, np.nan)
    fields["observed_mask"] = observed_mask
    fields["raw_natural"] = np.where(natural, fields["raw"], np.nan)
    raw_hp, raw_hp_valid = _regional_highpass(fields["raw"], natural, ocean)
    composed_nat_hp, composed_nat_valid = _regional_highpass(fields["composed"], natural, ocean)
    composed_pres_hp, composed_pres_valid = _regional_highpass(fields["composed"], preserved, ocean)
    composed_grad, composed_grad_valid = _regional_gradient(fields["composed"], ocean, ocean)
    fields["raw_natural_highpass"] = np.where(raw_hp_valid, raw_hp, np.nan)
    fields["composed_natural_highpass"] = np.where(composed_nat_valid, composed_nat_hp, np.nan)
    fields["composed_preserved_highpass"] = np.where(composed_pres_valid, composed_pres_hp, np.nan)
    fields["composed_gradient"] = np.where(composed_grad_valid, composed_grad, np.nan)

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        ocean = ocean[::-1, :]
        natural = natural[::-1, :]
        preserved = preserved[::-1, :]
        observed_mask = observed_mask[::-1, :]
        for key, value in list(fields.items()):
            if isinstance(value, np.ndarray) and value.ndim == 2:
                fields[key] = value[::-1, :]

    hp_values = np.concatenate(
        [fields[key][np.isfinite(fields[key])] for key in ("raw_natural_highpass", "composed_preserved_highpass")]
    )
    hp_max = float(np.percentile(hp_values, 98))
    metrics = {
        "raw_natural": _region_stats(fields["raw"], natural, ocean),
        "composed_natural": _region_stats(fields["composed"], natural, ocean),
        "composed_preserved": _region_stats(fields["composed"], preserved, ocean),
        "input_preserved": _region_stats(fields["input"], preserved, ocean),
    }
    natural_core = _region_core(natural, ocean)
    preserved_core = _region_core(preserved, ocean)
    metrics["composition_checks"] = {
        "max_abs_composed_minus_raw_in_natural_K": float(np.nanmax(np.abs(fields["composed"][natural] - fields["raw"][natural]))),
        "max_abs_composed_minus_input_in_preserved_K": float(np.nanmax(np.abs(fields["composed"][preserved] - fields["input"][preserved]))),
        "natural_core_count": int(natural_core.sum()),
        "preserved_core_count": int(preserved_core.sum()),
    }
    manifest = {
        "checkpoint": str(checkpoint.resolve()),
        "data_path": str(data_path.resolve()),
        "frame": int(frame),
        "timestamp": str(sample["timestamp"]),
        "no_gaussian_postprocessing": True,
        "natural_fraction": float(natural.mean()),
        "observed_fraction": float(observed_mask[ocean].mean()),
        "preserved_fraction": float(preserved.mean()),
        "metrics": metrics,
        "output_png": str(output.resolve()),
        "output_pdf": str(output.with_suffix(".pdf").resolve()),
    }
    _save_diagnostic_figure(
        output=output,
        fields=fields,
        ocean=ocean,
        natural=natural,
        preserved=preserved,
        lon=lon,
        lat=lat,
        timestamp=str(sample["timestamp"]),
        frame=frame,
        highpass_max=hp_max,
    )
    output.with_name(output.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    default_checkpoint, default_data, default_stats, default_output = _default_paths()
    default_output = default_output.with_name("series8_frame179_attribution_diagnostic.png")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--data-path", type=Path, default=default_data)
    parser.add_argument("--stats-path", type=Path, default=default_stats)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--frame", type=int, default=179)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(render(checkpoint=args.checkpoint, data_path=args.data_path, stats_path=args.stats_path, output=args.output, frame=args.frame, device_name=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
