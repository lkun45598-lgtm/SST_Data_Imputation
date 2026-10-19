"""Command-line entry point for parallel, resumable V6 preprocessing."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from Data_Imputation.preprocessing.temporal_hourly_v4.diagnostics import (
    write_manifest,
    write_series_diagnostics,
)

from .pipeline import run_all_series, run_series


def _parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.strip().removesuffix("Z"))
    return timestamp.replace(tzinfo=None) if timestamp.tzinfo is not None else timestamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel causal hourly JAXA SST preprocessing (V6)")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--series", type=int)
    mode.add_argument("--all-series", action="store_true")
    parser.add_argument("--smoke-until", type=_parse_timestamp)
    parser.add_argument("--query-workers", type=int, default=21)
    parser.add_argument("--series-workers", type=int, default=9)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smooth-synthetic", action="store_true")
    parser.add_argument("--smoothing-sigma", type=float, default=0.8)
    parser.add_argument("--first-layer-weight", type=float, default=0.25)
    parser.add_argument("--second-layer-weight", type=float, default=0.12)
    parser.add_argument("--interior-weight", type=float, default=0.08)
    parser.add_argument("--max-change-k", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.smoke_until is not None and args.all_series:
        parser.error("--smoke-until is available only with --series")
    if args.smoke_until is not None and (args.full or args.resume):
        parser.error("--smoke-until cannot be combined with --full or --resume")
    if args.smoke_until is None and not args.full:
        parser.error("complete-series processing requires explicit --full")

    parameters = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "query_workers": int(args.query_workers),
        "series_workers": int(args.series_workers),
        "resume": bool(args.resume),
        "full": bool(args.full),
        "smooth_synthetic": bool(args.smooth_synthetic),
        "smoothing_sigma": float(args.smoothing_sigma),
        "first_layer_weight": float(args.first_layer_weight),
        "second_layer_weight": float(args.second_layer_weight),
        "interior_weight": float(args.interior_weight),
        "max_change_k": float(args.max_change_k),
    }
    if args.series is not None:
        summary = run_series(
            args.source_root,
            args.output_root,
            series_id=int(args.series),
            query_workers=int(args.query_workers),
            stop_after=args.smoke_until,
            resume=bool(args.resume),
            smooth_synthetic=bool(args.smooth_synthetic),
            smoothing_sigma=float(args.smoothing_sigma),
            smoothing_first_layer_weight=float(args.first_layer_weight),
            smoothing_second_layer_weight=float(args.second_layer_weight),
            smoothing_interior_weight=float(args.interior_weight),
            max_change_k=float(args.max_change_k),
        )
        write_series_diagnostics(args.output_root, summary)
        write_manifest(args.output_root, [summary], parameters=parameters)
    else:
        summaries = run_all_series(
            args.source_root,
            args.output_root,
            series_workers=int(args.series_workers),
            query_workers=int(args.query_workers),
            resume=bool(args.resume),
            smooth_synthetic=bool(args.smooth_synthetic),
            smoothing_sigma=float(args.smoothing_sigma),
            smoothing_first_layer_weight=float(args.first_layer_weight),
            smoothing_second_layer_weight=float(args.second_layer_weight),
            smoothing_interior_weight=float(args.interior_weight),
            max_change_k=float(args.max_change_k),
        )
        for summary in summaries:
            write_series_diagnostics(args.output_root, summary)
        write_manifest(args.output_root, summaries, parameters=parameters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
