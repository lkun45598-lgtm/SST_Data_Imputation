"""Command line entry point for smoke and full v3 preprocessing."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np

from .pipeline import run_series
from .source import SourceIndex


def _write_smoke_diagnostics(
    *,
    source_root: Path,
    output_root: Path,
    series_id: int,
    target_hour: int,
    frame: int,
    neighbor_frames: int,
    output_path: Path,
) -> Path:
    index = SourceIndex.from_hour_dirs(source_root, series_id=series_id)
    target_times = [
        timestamp for timestamp in index.available_times if timestamp.hour == target_hour
    ]
    selected = [
        position
        for position in range(max(0, frame - neighbor_frames), min(len(target_times), frame + neighbor_frames + 1))
    ]
    with h5py.File(output_path, "r") as handle:
        sst = handle["sst_data"][:]
        land = handle["land_mask"][:].astype(bool)
        ocean_count = int((~land).sum())
        per_frame = []
        for row, position in enumerate(selected):
            record = {
                "frame_index": int(position),
                "timestamp": target_times[position].isoformat(),
                "raw_observed_fraction": float(handle["original_obs_mask"][row][~land].mean()),
                "temporal_fill_fraction": float(handle["temporal_fill_mask"][row][~land].mean()),
                "knn_fill_fraction": float(handle["knn_fill_mask"][row][~land].mean()),
                "unresolved_count": int(handle["unresolved_mask"][row][~land].sum()),
                "finite_ocean_count": int(np.isfinite(sst[row][~land]).sum()),
            }
            per_frame.append(record)

    missing_history = {}
    for position in selected:
        timestamp = target_times[position]
        available = {delta for delta, _ in index.history(timestamp, 72)}
        missing_history[timestamp.isoformat()] = [
            int(delta) for delta in range(1, 73) if delta not in available
        ]

    diagnostics = {
        "preprocess_version": "v3_temporal_hourly",
        "series_id": int(series_id),
        "target_hour": int(target_hour),
        "requested_frame": int(frame),
        "neighbor_frames": int(neighbor_frames),
        "selected_frame_indices": selected,
        "ocean_pixel_count": ocean_count,
        "output_path": str(output_path),
        "frames": per_frame,
        "missing_history_hours": missing_history,
        "causality": "all history timestamps are strictly earlier than target",
    }
    diagnostics_dir = output_root / "diagnostics" / f"frame{frame}"
    diagnostics_dir.mkdir(parents=True, exist_ok=False)
    overview_path = diagnostics_dir / "overview.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [record["frame_index"] for record in per_frame]
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.plot(x, [record["raw_observed_fraction"] for record in per_frame], marker="o", label="raw observation")
        ax.plot(x, [record["temporal_fill_fraction"] for record in per_frame], marker="o", label="temporal fill")
        ax.plot(x, [record["knn_fill_fraction"] for record in per_frame], marker="o", label="KNN fill")
        ax.set_xlabel("Target frame index")
        ax.set_ylabel("Ocean-pixel fraction")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(overview_path, dpi=160)
        plt.close(fig)
        diagnostics["overview_path"] = str(overview_path)
    except ImportError:
        diagnostics["overview_path"] = None
    summary_path = diagnostics_dir / "summary.json"
    summary_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal hourly SST preprocessing v3")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--series", type=int, required=True)
    parser.add_argument("--hour", type=int, required=True)
    parser.add_argument("--frame", type=int, default=223)
    parser.add_argument("--neighbor-frames", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="regenerate the complete target-hour series")
    parser.add_argument("--max-history-hours", type=int, default=72)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.neighbor_frames < 0:
        raise SystemExit("--neighbor-frames must be non-negative")
    if not args.full and args.frame < 0:
        raise SystemExit("--frame must be non-negative in smoke mode")

    selected_indices = None
    if not args.full:
        index = SourceIndex.from_hour_dirs(args.source_root, series_id=args.series)
        target_times = [timestamp for timestamp in index.available_times if timestamp.hour == args.hour]
        if args.frame >= len(target_times):
            raise SystemExit(
                f"--frame {args.frame} is outside target-hour series with {len(target_times)} frames"
            )
        selected_indices = list(
            range(max(0, args.frame - args.neighbor_frames),
                  min(len(target_times), args.frame + args.neighbor_frames + 1))
        )

    output_path = run_series(
        args.source_root,
        args.output_root,
        series_id=args.series,
        target_hour=args.hour,
        max_history_hours=args.max_history_hours,
        gaussian_sigma=args.gaussian_sigma,
        selected_indices=selected_indices,
        workers=args.workers,
    )
    print(f"wrote: {output_path}")
    if not args.full:
        summary = _write_smoke_diagnostics(
            source_root=args.source_root,
            output_root=args.output_root,
            series_id=args.series,
            target_hour=args.hour,
            frame=args.frame,
            neighbor_frames=args.neighbor_frames,
            output_path=output_path,
        )
        print(f"diagnostics: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
