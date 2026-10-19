#!/usr/bin/env python3
"""Four-stage visualization for the first days of temporal-hourly v3.

Each page/image contains, from left to right:

1. raw JAXA observations;
2. causal inverse-lag temporal fill;
3. the Gaussian-smoothed temporal fill;
4. the final v3 spatiotemporal KNN result.

The first three fields are reconstructed from the immutable source H5 files;
the final field is read from the already validated full v3 output.  A single
SST scale is computed over all raw observations in the selected interval so
that colors remain comparable across days.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

DI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DI))
sys.path.insert(0, str(DI / "visualization/paper_figs"))

from preprocessing.temporal_hourly_v3.algorithm import (  # noqa: E402
    causal_temporal_fill,
    gaussian_temporal_fills,
)
from preprocessing.temporal_hourly_v3.source import SourceIndex  # noqa: E402
from style import CMAPS, LAND_COLOR, MISSING_OVERLAY, apply_paper_style, setup_geo_ax  # noqa: E402


SERIES_ID = 8
TARGET_HOUR = 12
MAX_HISTORY_HOURS = 72
GAUSSIAN_SIGMA = 1.5
SOURCE_ROOT = DI / "experiments" / "hourly_data"
V3_ROOT = DI / "experiments" / "hourly_data_v3_temporal72_full_s08_h12"
DEFAULT_OUT = DI / "visualization" / "paper_figs" / "output" / "temporal_hourly_v3_first100"


def _stage_worker(args: tuple[Path, int, datetime, int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    """Compute only Stage 1 and Stage 2 for one timestamp in a child process."""

    source_root, series_id, timestamp, max_history_hours, sigma = args
    index = SourceIndex.from_hour_dirs(source_root, series_id=series_id)
    target = index.read(timestamp)
    history = index.history(timestamp, max_history_hours)
    temporal = causal_temporal_fill(target, history, max_hours=max_history_hours)
    gaussian = gaussian_temporal_fills(
        temporal.values,
        temporal.residual_missing,
        temporal.temporal_fill_mask,
        sigma=sigma,
    )
    return timestamp.isoformat(), temporal.values, gaussian


def _decode_timestamps(values: np.ndarray) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, np.bytes_):
            value = value.tobytes()
        if isinstance(value, bytes):
            value = value.decode("ascii")
        result.append(str(value))
    return result


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
    im = ax.imshow(
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
    return im


def _render_page(
    *,
    fields_k: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    land: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    vmin: float,
    vmax: float,
    frame_index: int,
    timestamp: str,
    out_png: Path,
    pdf: PdfPages,
) -> None:
    fields_c = tuple(field.astype(np.float32) - 273.15 for field in fields_k)
    titles = (
        "Raw JAXA observations",
        "Temporal weighted fill",
        r"Gaussian-smoothed fill ($\sigma=1.5$)",
        "Spatiotemporal KNN fill",
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.75), constrained_layout=True)
    im = None
    for col, (ax, field, title) in enumerate(zip(axes, fields_c, titles)):
        im = _draw_panel(
            ax,
            field,
            land=land,
            lon=lon,
            lat=lat,
            vmin=vmin,
            vmax=vmax,
            title=title,
            show_ylabel=(col == 0),
        )
    assert im is not None
    cbar = fig.colorbar(im, ax=axes.tolist(), location="right", shrink=0.88, pad=0.012, aspect=20)
    cbar.set_label("SST (°C)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)
    fig.suptitle(
        f"Series {SERIES_ID}, {timestamp} UTC  |  target-hour frame {frame_index}  |  first-100-day sequence",
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
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--series", type=int, default=SERIES_ID)
    parser.add_argument("--hour", type=int, default=TARGET_HOUR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start < 0 or args.count < 1 or args.workers < 1:
        raise SystemExit("--start must be >= 0; --count and --workers must be positive")
    if not 0 <= args.hour <= 23:
        raise SystemExit("--hour must be between 0 and 23")

    apply_paper_style()
    source_index = SourceIndex.from_hour_dirs(SOURCE_ROOT, series_id=args.series)
    target_times = tuple(t for t in source_index.available_times if t.hour == args.hour)
    stop = min(len(target_times), args.start + args.count)
    if args.start >= len(target_times):
        raise SystemExit(f"--start {args.start} is outside the {len(target_times)}-frame target-hour series")
    selected_times = target_times[args.start:stop]

    final_path = V3_ROOT / f"h{args.hour:02d}" / f"jaxa_knn_filled_{args.series:02d}.h5"
    if not final_path.exists():
        raise FileNotFoundError(f"Validated v3 output not found: {final_path}")
    with h5py.File(final_path, "r") as handle:
        final_timestamps = _decode_timestamps(handle["timestamps"][:])
        final_fields = handle["sst_data"][args.start:stop].astype(np.float32)
        output_land = handle["land_mask"][:].astype(bool)
        lat = handle["latitude"][:].astype(np.float32)
        lon = handle["longitude"][:].astype(np.float32)
    selected_iso = [t.isoformat() for t in selected_times]
    if final_timestamps[args.start:stop] != selected_iso:
        raise RuntimeError("Source and v3 timestamps are not aligned; refusing to render")
    if not np.array_equal(output_land, source_index.land_mask):
        raise RuntimeError("Source and v3 land masks differ; refusing to render")

    worker_args = [
        (SOURCE_ROOT, int(args.series), timestamp, MAX_HISTORY_HOURS, GAUSSIAN_SIGMA)
        for timestamp in selected_times
    ]
    workers = min(int(args.workers), len(worker_args))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            stage_results = list(executor.map(_stage_worker, worker_args))
    else:
        stage_results = [_stage_worker(item) for item in worker_args]
    stage_by_timestamp = {timestamp: (temporal, gaussian) for timestamp, temporal, gaussian in stage_results}

    raw_values = []
    for timestamp in selected_times:
        raw_values.append(source_index.read(timestamp).values)
    raw_values_arr = np.stack(raw_values)
    finite_raw_c = raw_values_arr[np.isfinite(raw_values_arr)] - 273.15
    if finite_raw_c.size == 0:
        raise RuntimeError("No finite raw observations in selected interval")
    vmin, vmax = np.percentile(finite_raw_c, [2, 98]).astype(float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"temporal_hourly_v3_{args.start:03d}_{stop - 1:03d}.pdf"
    with PdfPages(pdf_path) as pdf:
        for local_index, timestamp in enumerate(selected_times):
            timestamp_iso = timestamp.isoformat()
            temporal, gaussian = stage_by_timestamp[timestamp_iso]
            fields = (raw_values_arr[local_index], temporal, gaussian, final_fields[local_index])
            out_png = args.out_dir / f"frame{args.start + local_index:03d}_{timestamp.strftime('%Y%m%dT%H%M')}.png"
            _render_page(
                fields_k=fields,
                land=output_land,
                lon=lon,
                lat=lat,
                vmin=float(vmin),
                vmax=float(vmax),
                frame_index=args.start + local_index,
                timestamp=timestamp_iso,
                out_png=out_png,
                pdf=pdf,
            )

    print(f"frames_rendered={len(selected_times)}")
    print(f"timestamp_first={selected_iso[0]}")
    print(f"timestamp_last={selected_iso[-1]}")
    print(f"sst_colorbar_celsius=[{vmin:.3f}, {vmax:.3f}]")
    print(f"output_dir={args.out_dir}")
    print(f"multipage_pdf={pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
