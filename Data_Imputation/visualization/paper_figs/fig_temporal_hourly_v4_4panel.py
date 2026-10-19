#!/usr/bin/env python3
"""Four-stage diagnostic rendering for the v4 causal hourly proxy."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from Data_Imputation.preprocessing.temporal_hourly_v4.algorithm import (
    causal_temporal_fill,
    gaussian_temporal_fills,
)
from Data_Imputation.preprocessing.temporal_hourly_v4.source import SourceIndex, _decode_timestamp

from .style import CMAPS, LAND_COLOR, MISSING_OVERLAY, apply_paper_style, setup_geo_ax


_REQUIRED_FIELDS = (
    "raw",
    "temporal",
    "gaussian",
    "final",
    "land_mask",
    "latitude",
    "longitude",
)


def _field(data: Mapping[str, object], name: str, shape: tuple[int, int]) -> np.ndarray:
    if name not in data:
        raise ValueError(f"frame_data is missing required field: {name}")
    field = np.asarray(data[name])
    if field.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {field.shape}")
    return field


def _draw_panel(
    ax: plt.Axes,
    field_kelvin: np.ndarray,
    *,
    land: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    show_ylabel: bool,
) -> mpl.image.AxesImage:
    extent = [float(longitude.min()), float(longitude.max()), float(latitude.min()), float(latitude.max())]
    ax.imshow(
        np.where(land, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=mpl.colors.ListedColormap([LAND_COLOR]),
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        zorder=1,
    )
    missing = ~land & ~np.isfinite(field_kelvin)
    if np.any(missing):
        ax.imshow(
            np.where(missing, 1.0, np.nan),
            extent=extent,
            origin="lower",
            cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
            zorder=2,
        )
    image = ax.imshow(
        np.where(land, np.nan, field_kelvin - 273.15),
        extent=extent,
        origin="lower",
        cmap=CMAPS["sst"],
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        zorder=3,
    )
    setup_geo_ax(ax, longitude, latitude, draw_xlabel=True, draw_ylabel=show_ylabel, step=2)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    return image


def render_four_panel(
    frame_data: Mapping[str, object], output_path: Path, *, vmin: float | None = None, vmax: float | None = None
) -> Path:
    """Render raw, temporal, Gaussian, and final v4 SST using one Celsius scale."""

    missing_fields = set(_REQUIRED_FIELDS).difference(frame_data)
    if missing_fields:
        raise ValueError(f"frame_data is missing fields: {sorted(missing_fields)}")
    land = np.asarray(frame_data["land_mask"], dtype=bool)
    if land.ndim != 2:
        raise ValueError("land_mask must be a 2D array")
    raw = _field(frame_data, "raw", land.shape).astype(np.float32)
    temporal = _field(frame_data, "temporal", land.shape).astype(np.float32)
    gaussian = _field(frame_data, "gaussian", land.shape).astype(np.float32)
    final = _field(frame_data, "final", land.shape).astype(np.float32)
    latitude = np.asarray(frame_data["latitude"], dtype=np.float64)
    longitude = np.asarray(frame_data["longitude"], dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1 or land.shape != (len(latitude), len(longitude)):
        raise ValueError("latitude/longitude must describe the field grid")
    raw_celsius = raw[~land & np.isfinite(raw)] - 273.15
    if raw_celsius.size == 0:
        raise ValueError("raw panel contains no finite ocean observations for colour scaling")
    inferred_vmin, inferred_vmax = np.percentile(raw_celsius, [2.0, 98.0]).astype(float)
    if inferred_vmax <= inferred_vmin:
        inferred_vmin -= 0.5
        inferred_vmax += 0.5
    vmin = inferred_vmin if vmin is None else float(vmin)
    vmax = inferred_vmax if vmax is None else float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("vmin and vmax must be finite with vmax > vmin")

    apply_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.75), constrained_layout=True)
    titles = (
        "Raw JAXA observations",
        "Temporal weighted fill",
        r"Gaussian-smoothed fill ($\sigma=1.5$)",
        "V4 confidence-controlled proxy",
    )
    image = None
    for column, (field_kelvin, title) in enumerate(zip((raw, temporal, gaussian, final), titles)):
        image = _draw_panel(
            axes[column],
            field_kelvin,
            land=land,
            longitude=longitude,
            latitude=latitude,
            vmin=vmin,
            vmax=vmax,
            title=title,
            show_ylabel=(column == 0),
        )
    assert image is not None
    colourbar = fig.colorbar(image, ax=axes.tolist(), location="right", shrink=0.88, pad=0.012, aspect=20)
    colourbar.set_label("SST (°C)", fontsize=10)
    colourbar.ax.tick_params(labelsize=8)
    colourbar.outline.set_linewidth(0.5)
    timestamp = frame_data.get("timestamp")
    if timestamp is not None:
        fig.suptitle(f"V4 causal proxy | {timestamp}", fontsize=12, fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def load_v4_panels(
    source_root: Path,
    output_root: Path,
    *,
    series_id: int,
    hour: int,
    frame_index: int,
    max_history_hours: int = 72,
    gaussian_sigma: float = 1.5,
) -> dict[str, object]:
    """Load a final v4 frame and reconstruct its immutable intermediate panels."""

    source_index = SourceIndex.from_hour_dirs(Path(source_root), int(series_id))
    output_path = Path(output_root) / f"h{int(hour):02d}" / f"jaxa_knn_filled_{int(series_id):02d}.h5"
    with h5py.File(output_path, "r") as handle:
        timestamps = tuple(_decode_timestamp(value) for value in handle["timestamps"][:])
        if not 0 <= int(frame_index) < len(timestamps):
            raise IndexError(f"frame_index {frame_index} is outside the output H5")
        timestamp = timestamps[int(frame_index)]
        final = np.asarray(handle["sst_data"][int(frame_index)], dtype=np.float32)
    raw = source_index.read(timestamp)
    temporal = causal_temporal_fill(
        raw,
        source_index.raw_history(timestamp, int(max_history_hours)),
        max_hours=int(max_history_hours),
    )
    gaussian = gaussian_temporal_fills(
        temporal.values,
        temporal.source_code,
        temporal.temporal_fill_mask,
        temporal.residual_missing,
        sigma=float(gaussian_sigma),
    )
    if source_index.latitude is None or source_index.longitude is None:
        raise ValueError("Source H5 must include latitude and longitude for plotting")
    return {
        "raw": raw.values,
        "temporal": temporal.values,
        "gaussian": gaussian,
        "final": final,
        "land_mask": raw.land_mask,
        "latitude": source_index.latitude,
        "longitude": source_index.longitude,
        "timestamp": timestamp.isoformat(),
    }
