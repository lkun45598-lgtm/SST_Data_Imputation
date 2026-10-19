#!/usr/bin/env python3
"""Render four-stage panels for the legacy hourly JAXA proxy pipeline.

The legacy H5 files preserve the final causal 3D-KNN proxy and the Stage-1
audit masks, but not the Stage-1 and Stage-2 fields themselves.  This script
reconstructs those two intermediate fields exactly from immutable original
observations using the legacy implementation:

1. same-UTC observations at one, two, and three previous days;
2. inverse-day-distance weighting (1 / delta_days);
3. Gaussian smoothing with sigma=1.5, written only to Stage-1 fill pixels;
4. the saved legacy causal 3D-KNN proxy read directly from the H5 file.

Before rendering, each reconstructed Stage-1 fill/residual mask is checked
against the corresponding audit masks stored in the legacy H5.  The output is
therefore a visualisation of the actual legacy pipeline, not a newly-run KNN.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import ndimage


DI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DI / "visualization/paper_figs"))

from style import CMAPS, LAND_COLOR, MISSING_OVERLAY, apply_paper_style, setup_geo_ax  # noqa: E402


SERIES_ID = 8
TARGET_HOUR = 12
LEGACY_LOOKBACK_DAYS = 3
GAUSSIAN_SIGMA = 1.5
LEGACY_ROOT = DI / "experiments" / "hourly_data"
DEFAULT_OUT = DI / "visualization" / "paper_figs" / "output" / "temporal_hourly_legacy_first100"


def _decode_timestamp(value: object) -> datetime:
    if isinstance(value, np.bytes_):
        value = value.tobytes()
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")
    return datetime.fromisoformat(value.strip())


def legacy_temporal_weighted_fill(
    raw_values: np.ndarray,
    raw_observed: np.ndarray,
    history: tuple[tuple[int, np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct legacy Stage 1 from original same-UTC observations.

    ``history`` consists of ``(delta_days, values, observed_mask)`` entries.
    Only true original observations can enter the weighted average, regardless
    of the filled values stored in the legacy H5 file.
    """

    values = np.where(raw_observed & np.isfinite(raw_values), raw_values, np.nan).astype(
        np.float32, copy=True
    )
    numerator = np.zeros(values.shape, dtype=np.float64)
    denominator = np.zeros(values.shape, dtype=np.float64)

    for delta_days, history_values, history_observed in history:
        if delta_days < 1 or delta_days > LEGACY_LOOKBACK_DAYS:
            raise ValueError(f"Legacy history day is outside [1, 3]: {delta_days}")
        valid = history_observed & np.isfinite(history_values)
        weight = 1.0 / float(delta_days)
        numerator[valid] += weight * history_values[valid]
        denominator[valid] += weight

    fill_mask = (~np.isfinite(values)) & (denominator > 0.0)
    values[fill_mask] = (numerator[fill_mask] / denominator[fill_mask]).astype(np.float32)
    residual_missing = ~np.isfinite(values)
    return values, fill_mask, residual_missing


def legacy_gaussian_fill(
    stage1_values: np.ndarray,
    residual_missing: np.ndarray,
    temporal_fill_mask: np.ndarray,
    *,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Reconstruct legacy Stage 2 using the original write-back semantics."""

    valid_mask = np.isfinite(stage1_values) & ~residual_missing
    if not np.any(valid_mask):
        return stage1_values.copy()
    filtered_input = stage1_values.astype(np.float32, copy=True)
    filtered_input[~valid_mask] = np.nanmean(stage1_values[valid_mask])
    filtered = ndimage.gaussian_filter(filtered_input, sigma=sigma)
    write = valid_mask & temporal_fill_mask
    return np.where(write, filtered, stage1_values).astype(np.float32)


def _raw_frame(values: np.ndarray, observed: np.ndarray, land: np.ndarray) -> np.ndarray:
    return np.where(observed & ~land & np.isfinite(values), values, np.nan).astype(np.float32)


def read_legacy_frame_batch(dataset: h5py.Dataset, indices: tuple[int, ...] | np.ndarray) -> np.ndarray:
    """Read frames while preserving the leading time dimension for one frame."""

    selector = np.asarray(indices, dtype=np.int64)
    if selector.ndim != 1 or selector.size == 0:
        raise ValueError("Frame indices must be a non-empty one-dimensional sequence")
    return np.asarray(dataset[selector])


def _draw_panel(
    ax: plt.Axes,
    field_c: np.ndarray,
    *,
    land: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    show_ylabel: bool,
):
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
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
    missing = (~land) & ~np.isfinite(field_c)
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
        np.where(land, np.nan, field_c),
        extent=extent,
        origin="lower",
        cmap=CMAPS["sst"],
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        zorder=3,
    )
    setup_geo_ax(ax, lon, lat, draw_xlabel=True, draw_ylabel=show_ylabel, step=2)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    return image


def _render_page(
    *,
    fields_k: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    land: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    vmin: float,
    vmax: float,
    frame_index: int,
    timestamp: datetime,
    out_png: Path,
    pdf: PdfPages,
) -> None:
    fields_c = tuple(field.astype(np.float32) - 273.15 for field in fields_k)
    titles = (
        "Raw JAXA observations",
        "Temporal weighted fill",
        r"Gaussian-smoothed fill ($\sigma=1.5$)",
        "Legacy causal 3D KNN proxy",
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.75), constrained_layout=True)
    image = None
    for column, (ax, field, title) in enumerate(zip(axes, fields_c, titles)):
        image = _draw_panel(
            ax,
            field,
            land=land,
            lon=lon,
            lat=lat,
            vmin=vmin,
            vmax=vmax,
            title=title,
            show_ylabel=(column == 0),
        )
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.tolist(), location="right", shrink=0.88, pad=0.012, aspect=20)
    colorbar.set_label("SST (°C)", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_linewidth(0.5)
    fig.suptitle(
        f"Legacy proxy | Series {SERIES_ID}, {timestamp.isoformat()} UTC | "
        f"target-hour frame {frame_index} | first-100-day sequence",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--series", type=int, default=SERIES_ID)
    parser.add_argument("--hour", type=int, default=TARGET_HOUR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start < 0 or args.count < 1:
        raise SystemExit("--start must be >= 0 and --count must be positive")
    if not 0 <= args.hour <= 23:
        raise SystemExit("--hour must be between 0 and 23")

    apply_paper_style()
    h5_path = LEGACY_ROOT / f"h{args.hour:02d}" / f"jaxa_knn_filled_{args.series:02d}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Legacy H5 was not found: {h5_path}")

    with h5py.File(h5_path, "r") as handle:
        required = {
            "sst_data", "original_obs_mask", "temporal_fill_mask", "original_missing_mask",
            "land_mask", "latitude", "longitude", "timestamps",
        }
        missing = required.difference(handle.keys())
        if missing:
            raise ValueError(f"Legacy H5 is missing required fields: {sorted(missing)}")
        timestamps = tuple(_decode_timestamp(value) for value in handle["timestamps"][:])
        stop = min(len(timestamps), args.start + args.count)
        if args.start >= len(timestamps):
            raise SystemExit(f"--start {args.start} is outside the {len(timestamps)}-frame series")
        selected_indices = tuple(range(args.start, stop))
        selected_times = tuple(timestamps[index] for index in selected_indices)
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
        if len(timestamp_to_index) != len(timestamps):
            raise ValueError("Legacy H5 has duplicate timestamps")

        required_indices = set(selected_indices)
        for timestamp in selected_times:
            for delta_days in range(1, LEGACY_LOOKBACK_DAYS + 1):
                history_index = timestamp_to_index.get(timestamp - timedelta(days=delta_days))
                if history_index is not None:
                    required_indices.add(history_index)
        read_indices = np.asarray(sorted(required_indices), dtype=np.int64)
        source_values = handle["sst_data"][read_indices].astype(np.float32)
        source_observed = handle["original_obs_mask"][read_indices].astype(bool)
        original_fill_masks = read_legacy_frame_batch(
            handle["temporal_fill_mask"], selected_indices
        ).astype(bool)
        original_residual_masks = read_legacy_frame_batch(
            handle["original_missing_mask"], selected_indices
        ).astype(bool)
        final_values = read_legacy_frame_batch(handle["sst_data"], selected_indices).astype(np.float32)
        land = handle["land_mask"][:].astype(bool)
        lat = handle["latitude"][:].astype(np.float32)
        lon = handle["longitude"][:].astype(np.float32)

    values_by_index = {index: source_values[offset] for offset, index in enumerate(read_indices)}
    observed_by_index = {index: source_observed[offset] for offset, index in enumerate(read_indices)}
    raw_fields: list[np.ndarray] = []
    temporal_fields: list[np.ndarray] = []
    gaussian_fields: list[np.ndarray] = []
    for local_index, (frame_index, timestamp) in enumerate(zip(selected_indices, selected_times)):
        raw = _raw_frame(values_by_index[frame_index], observed_by_index[frame_index], land)
        history = []
        for delta_days in range(1, LEGACY_LOOKBACK_DAYS + 1):
            history_index = timestamp_to_index.get(timestamp - timedelta(days=delta_days))
            if history_index is not None:
                history.append((
                    delta_days,
                    _raw_frame(values_by_index[history_index], observed_by_index[history_index], land),
                    observed_by_index[history_index] & ~land,
                ))
        temporal, fill_mask, residual_missing = legacy_temporal_weighted_fill(raw, ~land & np.isfinite(raw), tuple(history))

        # The legacy H5 records these exact Stage-1 masks; refuse to render if
        # a timestamp or observation-mask interpretation has drifted.
        if not np.array_equal(fill_mask, original_fill_masks[local_index]):
            raise RuntimeError(f"Stage-1 fill-mask mismatch at {timestamp.isoformat()}")
        if not np.array_equal(residual_missing, original_residual_masks[local_index]):
            raise RuntimeError(f"Stage-1 residual-mask mismatch at {timestamp.isoformat()}")
        raw_fields.append(raw)
        temporal_fields.append(temporal)
        gaussian_fields.append(legacy_gaussian_fill(temporal, residual_missing, fill_mask))

    raw_stack = np.stack(raw_fields)
    finite_raw_c = raw_stack[np.isfinite(raw_stack)] - 273.15
    if finite_raw_c.size == 0:
        raise RuntimeError("No finite raw observations in selected interval")
    vmin, vmax = np.percentile(finite_raw_c, [2, 98]).astype(float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"temporal_hourly_legacy_{args.start:03d}_{stop - 1:03d}.pdf"
    with PdfPages(pdf_path) as pdf:
        for local_index, (frame_index, timestamp) in enumerate(zip(selected_indices, selected_times)):
            out_png = args.out_dir / f"frame{frame_index:03d}_{timestamp.strftime('%Y%m%dT%H%M')}.png"
            _render_page(
                fields_k=(
                    raw_fields[local_index],
                    temporal_fields[local_index],
                    gaussian_fields[local_index],
                    final_values[local_index],
                ),
                land=land,
                lon=lon,
                lat=lat,
                vmin=float(vmin),
                vmax=float(vmax),
                frame_index=frame_index,
                timestamp=timestamp,
                out_png=out_png,
                pdf=pdf,
            )

    print(f"frames_rendered={len(selected_indices)}")
    print(f"timestamp_first={selected_times[0].isoformat()}")
    print(f"timestamp_last={selected_times[-1].isoformat()}")
    print(f"sst_colorbar_celsius=[{vmin:.3f}, {vmax:.3f}]")
    print(f"output_dir={args.out_dir}")
    print(f"multipage_pdf={pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
