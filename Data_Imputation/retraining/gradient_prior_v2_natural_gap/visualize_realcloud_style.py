"""Render a formal FNO-CBAM real-cloud result in the paper figure style.

The layout follows ``fig_realcloud_gradient``: the first row shows filled SST
and the second row shows gradient magnitude.  Series 8 has no collocated
ground truth, so the KNN field is labelled as a proxy and the observation
column is not assigned a gradient reference inside the cloud gap.

The formal checkpoint is used without Gaussian post-processing or clipping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from Data_Imputation.visualization.paper_figs.style import (
        CMAPS,
        LAND_COLOR,
        MISSING_OVERLAY,
        apply_paper_style,
        setup_geo_ax,
    )
except ImportError:
    # The A100 inference environment does not need the optional cmocean
    # dependency used by the broader paper-figure collection.
    CMAPS = {"sst": plt.get_cmap("RdYlBu_r")}
    LAND_COLOR = "#d8c7a3"
    MISSING_OVERLAY = "#b8b8b8"

    def apply_paper_style() -> None:
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                "font.size": 9,
                "axes.linewidth": 0.8,
                "axes.edgecolor": "#222222",
                "xtick.direction": "out",
                "ytick.direction": "out",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def setup_geo_ax(
        ax: plt.Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        *,
        draw_xlabel: bool = True,
        draw_ylabel: bool = True,
        step: int = 2,
    ) -> None:
        xticks = np.arange(np.ceil(lon.min()), np.floor(lon.max()) + 1, step)
        yticks = np.arange(np.ceil(lat.min()), np.floor(lat.max()) + 1, step)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{int(x)}$^\\circ$E" for x in xticks])
        ax.set_yticklabels([f"{int(y)}$^\\circ$N" for y in yticks])
        if draw_xlabel:
            ax.set_xlabel("Longitude")
        if draw_ylabel:
            ax.set_ylabel("Latitude")
        ax.tick_params(direction="out", length=3, width=0.7, color="#333333")

from .config import RunConfig
from .losses import boundary_ramp_composition, boundary_transition_composition
from .train import _load_model


SST_CMAP = CMAPS["sst"]
GRAD_CMAP = mpl.colormaps["magma"]


def _default_paths() -> tuple[Path, Path, Path, Path]:
    package_root = Path(__file__).resolve().parents[2]
    experiments = package_root / "experiments"
    checkpoint = (
        experiments
        / "gradient_prior_v2_natural_gap"
        / "h12_gradient_prior_v2_natural_gap_ddp_20260821_115832"
        / "best_model.pth"
    )
    data_path = experiments / "hourly_data" / "h12" / "jaxa_knn_filled_08.h5"
    stats_path = experiments / "gradient_prior_v2_natural_gap" / "stats" / "h12_train_monthly_v2.json"
    output = (
        experiments
        / "gradient_prior_v2_natural_gap"
        / "inference_20260822_030000"
        / "figures"
        / "series8_realcloud_gradient_style.png"
    )
    return checkpoint, data_path, stats_path, output


def _decode_timestamp(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _make_sample(
    handle: h5py.File,
    index: int,
    *,
    mean: float,
    std: float,
) -> dict[str, object]:
    frame_count = int(handle["sst_data"].shape[0])
    if not 0 <= index < frame_count:
        raise IndexError(f"frame {index} outside [0, {frame_count})")

    land = np.asarray(handle["land_mask"][:], dtype=bool)
    ocean = ~land
    start = max(0, index - 29)
    sst_sequence = np.asarray(handle["sst_data"][start : index + 1], dtype=np.float32)
    mask_sequence = np.asarray(
        handle["original_missing_mask"][start : index + 1], dtype=bool
    )
    padding = 30 - sst_sequence.shape[0]
    if padding:
        sst_sequence = np.concatenate(
            [np.repeat(sst_sequence[:1], padding, axis=0), sst_sequence], axis=0
        )
        mask_sequence = np.concatenate(
            [np.repeat(mask_sequence[:1], padding, axis=0), mask_sequence], axis=0
        )

    # The formal retraining protocol reconstructs the residual KNN-missing
    # region; temporal-fill pixels remain part of the causal input proxy.
    natural = np.asarray(handle["original_missing_mask"][index], dtype=bool) & ocean
    observed = np.asarray(handle["original_obs_mask"][index], dtype=bool) & ocean
    target_proxy = np.asarray(handle["sst_data"][index], dtype=np.float32)

    target_input = sst_sequence[-1].copy()
    target_input[natural] = mean
    sst_sequence[-1] = target_input
    mask_sequence[-1] = natural
    normalized = np.nan_to_num(
        (sst_sequence - mean) / std, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    return {
        "input_sst_seq": normalized,
        "mask_seq": mask_sequence,
        "reconstruction_mask": natural,
        "land_mask": land,
        "natural_mask": natural,
        "observed_mask": observed,
        "target_proxy": target_proxy,
        "timestamp": _decode_timestamp(handle["timestamps"][index]),
        "index": index,
    }


def _predict_realcloud_sample(
    model: torch.nn.Module,
    sample: dict[str, object],
    *,
    device: torch.device,
    mean: float,
    std: float,
    boundary_ramp: bool = False,
    boundary_ramp_layers: int = 3,
    boundary_ramp_max_gradient_k: float = 0.6,
) -> np.ndarray:
    """Run the deployment-only boundary transition for a real-cloud frame."""
    sst_sequence = torch.from_numpy(
        np.asarray(sample["input_sst_seq"], dtype=np.float32)
    ).unsqueeze(0).to(device)
    mask_sequence = torch.from_numpy(
        np.asarray(sample["mask_seq"], dtype=np.float32)
    ).unsqueeze(0).to(device)
    natural = torch.from_numpy(
        np.asarray(sample["natural_mask"], dtype=bool)
    ).unsqueeze(0).to(device)
    land = torch.from_numpy(
        np.asarray(sample["land_mask"], dtype=bool)
    ).unsqueeze(0).to(device)
    causal_proxy = torch.from_numpy(
        np.nan_to_num(
            (np.asarray(sample["target_proxy"], dtype=np.float32) - float(mean))
            / float(std),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        raw_prediction = model(sst_sequence, mask_sequence)
        if boundary_ramp:
            composed = boundary_ramp_composition(
                raw_prediction,
                sst_sequence[:, -1:, :, :],
                causal_proxy,
                natural,
                land,
                max_layers=boundary_ramp_layers,
                max_seam_gradient=float(boundary_ramp_max_gradient_k) / float(std),
            )
        else:
            composed = boundary_transition_composition(
                raw_prediction,
                sst_sequence[:, -1:, :, :],
                causal_proxy,
                natural,
                land,
            )
    return (composed[0, 0] * float(std) + float(mean)).cpu().numpy()


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


def _gradient_magnitude(field: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    filled = np.where(ocean, field, np.nan)
    with np.errstate(invalid="ignore"):
        grad_y, grad_x = np.gradient(filled)
        magnitude = np.sqrt(grad_y**2 + grad_x**2)
    return np.where(ocean, magnitude, np.nan)


def _draw_field(
    ax: plt.Axes,
    field: np.ndarray,
    *,
    extent: list[float],
    ocean: np.ndarray,
    cmap: mpl.colors.Colormap,
    vmin: float,
    vmax: float,
    lon: np.ndarray,
    lat: np.ndarray,
    show_x: bool,
    show_y: bool,
) -> mpl.image.AxesImage:
    ax.imshow(
        np.where(~ocean, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=mpl.colors.ListedColormap([LAND_COLOR]),
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        zorder=1,
    )
    image = ax.imshow(
        field,
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        zorder=2,
    )
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    return image


def render(
    *,
    checkpoint: Path,
    data_path: Path,
    stats_path: Path,
    output: Path,
    frame: int,
    device_name: str,
    boundary_ramp: bool = False,
    boundary_ramp_layers: int = 3,
    boundary_ramp_max_gradient_k: float = 0.6,
) -> dict[str, object]:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    apply_paper_style()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    data_dir = data_path.parent
    model, config = _load_formal_model(checkpoint, data_dir, stats_path, device)
    with h5py.File(data_path, "r") as handle:
        sample = _make_sample(
            handle, frame, mean=config.norm_mean, std=config.norm_std
        )
        latitude = np.asarray(handle["latitude"][:], dtype=np.float32)
        longitude = np.asarray(handle["longitude"][:], dtype=np.float32)

    model_field = _predict_realcloud_sample(
        model,
        sample,
        device=device,
        mean=config.norm_mean,
        std=config.norm_std,
        boundary_ramp=boundary_ramp,
        boundary_ramp_layers=boundary_ramp_layers,
        boundary_ramp_max_gradient_k=boundary_ramp_max_gradient_k,
    )
    ocean = ~np.asarray(sample["land_mask"], dtype=bool)
    observed = np.asarray(sample["observed_mask"], dtype=bool)
    natural = np.asarray(sample["natural_mask"], dtype=bool)
    target_proxy = np.asarray(sample["target_proxy"], dtype=np.float32)
    model_field = np.asarray(model_field, dtype=np.float32)
    observed_field = np.where(observed, target_proxy, np.nan)
    model_field = np.where(ocean, model_field, np.nan)
    proxy_field = np.where(ocean, target_proxy, np.nan)

    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        ocean = ocean[::-1, :]
        observed = observed[::-1, :]
        natural = natural[::-1, :]
        observed_field = observed_field[::-1, :]
        model_field = model_field[::-1, :]
        proxy_field = proxy_field[::-1, :]
    extent = [float(longitude.min()), float(longitude.max()), float(latitude.min()), float(latitude.max())]

    valid_sst = np.concatenate(
        [field[np.isfinite(field)] for field in (observed_field, model_field, proxy_field)]
    )
    sst_min, sst_max = np.percentile(valid_sst - 273.15, [2, 98])
    model_c = model_field - 273.15
    proxy_c = proxy_field - 273.15
    observed_c = observed_field - 273.15
    gradients = {
        "fno": _gradient_magnitude(model_c, ocean),
        "proxy": _gradient_magnitude(proxy_c, ocean),
    }
    valid_grad = np.concatenate(
        [value[np.isfinite(value)] for value in gradients.values()]
    )
    grad_max = float(np.percentile(valid_grad, 98))

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.7), layout="constrained")
    titles = ["JAXA obs\n(cloud gaps)", "FNO-CBAM\n(Ours)", "KNN / IDW\n(proxy)"]
    top_fields = [observed_c, model_c, proxy_c]
    top_images = []
    for col, field in enumerate(top_fields):
        image = _draw_field(
            axes[0, col],
            field,
            extent=extent,
            ocean=ocean,
            cmap=SST_CMAP,
            vmin=float(sst_min),
            vmax=float(sst_max),
            lon=longitude,
            lat=latitude,
            show_x=False,
            show_y=col == 0,
        )
        top_images.append(image)
        axes[0, col].set_title(titles[col], fontsize=10.5, fontweight="bold")

    # Match the reference figure: only the observation column gets a grey
    # cloud overlay, so the filled fields remain visually uninterrupted.
    cloud_overlay = (~observed) & ocean
    axes[0, 0].imshow(
        np.where(cloud_overlay, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        zorder=3,
    )

    # There is no pointwise reference gradient in natural cloud gaps.
    axes[1, 0].imshow(
        np.where(ocean, 0.92, np.nan),
        extent=extent,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="equal",
        zorder=1,
    )
    axes[1, 0].imshow(
        np.where(~ocean, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=mpl.colors.ListedColormap([LAND_COLOR]),
        vmin=0,
        vmax=1,
        aspect="equal",
        zorder=2,
    )
    axes[1, 0].text(
        0.5,
        0.5,
        "(no reference\nin gap)",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="0.45",
        style="italic",
    )
    setup_geo_ax(axes[1, 0], longitude, latitude, draw_xlabel=True, draw_ylabel=True, step=2)

    gradient_images = []
    for col, key in enumerate(("fno", "proxy"), start=1):
        image = _draw_field(
            axes[1, col],
            gradients[key],
            extent=extent,
            ocean=ocean,
            cmap=GRAD_CMAP,
            vmin=0.0,
            vmax=grad_max,
            lon=longitude,
            lat=latitude,
            show_x=True,
            show_y=False,
        )
        gradient_images.append(image)

    cbar_sst = fig.colorbar(
        top_images[-1],
        ax=axes[0, :].tolist(),
        location="right",
        shrink=0.82,
        pad=0.008,
        aspect=18,
    )
    cbar_sst.set_label(r"SST ($^\circ$C)", fontsize=10)
    cbar_grad = fig.colorbar(
        gradient_images[-1],
        ax=axes[1, :].tolist(),
        location="right",
        shrink=0.82,
        pad=0.008,
        aspect=18,
    )
    cbar_grad.set_label(r"$|\nabla \mathrm{SST}|$ (K/pixel)", fontsize=10)
    axes[0, 0].text(
        -0.55,
        0.5,
        "Filled SST",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.55,
        0.5,
        "Gradient magnitude",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    observed_fraction = float(observed[ocean].mean())
    natural_fraction = float(natural[ocean].mean())
    timestamp = str(sample["timestamp"])
    fig.suptitle(
        f"Reconstruction of a real Himawari-8 cloud gap "
        f"(no ground truth; observed {observed_fraction * 100:.0f}%). "
        "Grey overlay = non-observed ocean pixels.",
        fontsize=12.5,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    manifest = {
        "checkpoint": str(checkpoint.resolve()),
        "data_path": str(data_path.resolve()),
        "frame": int(frame),
        "timestamp": timestamp,
        "observed_fraction": observed_fraction,
        "natural_residual_fraction": natural_fraction,
        "no_gaussian_postprocessing": True,
        "boundary_ramp_enabled": bool(boundary_ramp),
        "boundary_ramp_layers": int(boundary_ramp_layers),
        "boundary_ramp_max_gradient_k": float(boundary_ramp_max_gradient_k),
        "output_png": str(output.resolve()),
        "output_pdf": str(output.with_suffix(".pdf").resolve()),
    }
    output.with_name(output.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    default_checkpoint, default_data, default_stats, default_output = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--data-path", type=Path, default=default_data)
    parser.add_argument("--stats-path", type=Path, default=default_stats)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--frame", type=int, default=179)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--boundary-ramp", action="store_true")
    parser.add_argument("--boundary-ramp-layers", type=int, default=3)
    parser.add_argument("--boundary-ramp-max-gradient-k", type=float, default=0.6)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                checkpoint=args.checkpoint,
                data_path=args.data_path,
                stats_path=args.stats_path,
                output=args.output,
                frame=args.frame,
                device_name=args.device,
                boundary_ramp=args.boundary_ramp,
                boundary_ramp_layers=args.boundary_ramp_layers,
                boundary_ramp_max_gradient_k=args.boundary_ramp_max_gradient_k,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
