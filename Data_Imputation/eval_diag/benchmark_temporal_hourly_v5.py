"""Benchmark V5 against V4 without writing any data below the final output root."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import h5py
import numpy as np

from Data_Imputation.preprocessing.temporal_hourly_v4.algorithm import KNNResult
from Data_Imputation.preprocessing.temporal_hourly_v4.pipeline import run_series as run_v4
from Data_Imputation.preprocessing.temporal_hourly_v5.pipeline import run_series as run_v5


@dataclass(frozen=True)
class ComparisonReport:
    """Exact provenance and bounded-numeric comparison results."""

    max_sst_abs_error: float
    max_confidence_abs_error: float
    mismatched_source_pixels: int
    mismatched_knn_fill_pixels: int
    mismatched_mask_pixels: int
    left_unresolved_ocean_pixels: int
    right_unresolved_ocean_pixels: int
    passes: bool


@dataclass(frozen=True)
class BenchmarkReport:
    """Timing and output-equivalence evidence for a V4/V5 smoke segment."""

    frame_count: int
    v4_seconds: float
    v5_seconds: float
    speedup: float
    comparison: ComparisonReport


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Return infinity for differing NaN patterns and max finite absolute error otherwise."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    matching_nan = np.isnan(left) == np.isnan(right)
    if not np.all(matching_nan):
        return float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    return 0.0 if not np.any(finite) else float(np.max(np.abs(left[finite] - right[finite])))


def _report(
    *,
    sst_error: float,
    confidence_error: float,
    source_mismatch: int,
    knn_mismatch: int,
    mask_mismatch: int,
    left_unresolved: int,
    right_unresolved: int,
    atol: float,
) -> ComparisonReport:
    passes = (
        source_mismatch == 0
        and knn_mismatch == 0
        and mask_mismatch == 0
        and left_unresolved == 0
        and right_unresolved == 0
        and sst_error <= float(atol)
        and confidence_error <= float(atol)
    )
    return ComparisonReport(
        max_sst_abs_error=sst_error,
        max_confidence_abs_error=confidence_error,
        mismatched_source_pixels=source_mismatch,
        mismatched_knn_fill_pixels=knn_mismatch,
        mismatched_mask_pixels=mask_mismatch,
        left_unresolved_ocean_pixels=left_unresolved,
        right_unresolved_ocean_pixels=right_unresolved,
        passes=passes,
    )


def compare_completed_frames(
    left: KNNResult,
    right: KNNResult,
    *,
    atol: float = 1e-6,
) -> ComparisonReport:
    """Compare in-memory completed frames before any lossy H5 confidence conversion."""

    source_mismatch = int(
        np.asarray(left.source_code).shape != np.asarray(right.source_code).shape
        or np.count_nonzero(np.asarray(left.source_code) != np.asarray(right.source_code))
    )
    knn_mismatch = int(
        np.asarray(left.knn_fill_mask).shape != np.asarray(right.knn_fill_mask).shape
        or np.count_nonzero(np.asarray(left.knn_fill_mask) != np.asarray(right.knn_fill_mask))
    )
    return _report(
        sst_error=_maximum_difference(left.values, right.values),
        confidence_error=_maximum_difference(left.confidence, right.confidence),
        source_mismatch=source_mismatch,
        knn_mismatch=knn_mismatch,
        mask_mismatch=0,
        left_unresolved=int(np.count_nonzero(~np.isfinite(left.values))),
        right_unresolved=int(np.count_nonzero(~np.isfinite(right.values))),
        atol=atol,
    )


def _output_files(root: Path) -> dict[Path, Path]:
    root = Path(root)
    return {
        path.relative_to(root): path
        for path in root.glob("h??/jaxa_knn_filled_*.h5")
    }


def compare_output_roots(left_root: Path, right_root: Path, *, atol: float = 1e-6) -> ComparisonReport:
    """Compare every produced V4/V5 H5 field required for causal provenance."""

    left_files = _output_files(left_root)
    right_files = _output_files(right_root)
    file_mismatch = len(set(left_files).symmetric_difference(right_files))
    max_sst = 0.0
    max_confidence = 0.0
    source_mismatch = 0
    knn_mismatch = 0
    mask_mismatch = file_mismatch
    left_unresolved = 0
    right_unresolved = 0
    mask_fields = (
        "original_obs_mask",
        "temporal_fill_mask",
        "original_missing_mask",
        "unresolved_mask",
        "timestamps",
    )
    for relative_path in sorted(set(left_files).intersection(right_files)):
        with h5py.File(left_files[relative_path], "r") as left, h5py.File(
            right_files[relative_path], "r"
        ) as right:
            if set(left) != set(right):
                mask_mismatch += len(set(left).symmetric_difference(right))
                continue
            max_sst = max(max_sst, _maximum_difference(left["sst_data"][:], right["sst_data"][:]))
            max_confidence = max(
                max_confidence,
                _maximum_difference(left["proxy_confidence"][:], right["proxy_confidence"][:]),
            )
            source_mismatch += int(np.count_nonzero(left["source_code"][:] != right["source_code"][:]))
            knn_mismatch += int(np.count_nonzero(left["knn_fill_mask"][:] != right["knn_fill_mask"][:]))
            for field in mask_fields:
                mask_mismatch += int(np.count_nonzero(left[field][:] != right[field][:]))
            land = np.asarray(left["land_mask"][:], dtype=bool)
            left_unresolved += int(np.asarray(left["unresolved_mask"][:])[:, ~land].sum())
            right_unresolved += int(np.asarray(right["unresolved_mask"][:])[:, ~land].sum())
    return _report(
        sst_error=max_sst,
        confidence_error=max_confidence,
        source_mismatch=source_mismatch,
        knn_mismatch=knn_mismatch,
        mask_mismatch=mask_mismatch,
        left_unresolved=left_unresolved,
        right_unresolved=right_unresolved,
        atol=atol,
    )


def benchmark_series(
    source_root: Path,
    *,
    series_id: int,
    until: datetime,
    query_workers: int,
    work_root: Path,
) -> BenchmarkReport:
    """Run V4 and V5 into an empty temporary root and compare all written H5 fields."""

    work_root = Path(work_root)
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"Refusing non-empty benchmark root: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    v4_root = work_root / "v4"
    v5_root = work_root / "v5"
    start = perf_counter()
    v4_summary = run_v4(
        source_root,
        v4_root,
        series_id=series_id,
        query_workers=query_workers,
        stop_after=until,
    )
    v4_seconds = perf_counter() - start
    start = perf_counter()
    v5_summary = run_v5(
        source_root,
        v5_root,
        series_id=series_id,
        query_workers=query_workers,
        stop_after=until,
    )
    v5_seconds = perf_counter() - start
    if v5_summary.processed_timestamps != v4_summary.processed_timestamps:
        raise RuntimeError("V4 and V5 benchmark runs processed different timestamps")
    comparison = compare_output_roots(v4_root, v5_root)
    return BenchmarkReport(
        frame_count=len(v4_summary.processed_timestamps),
        v4_seconds=v4_seconds,
        v5_seconds=v5_seconds,
        speedup=float("inf") if v5_seconds == 0.0 else v4_seconds / v5_seconds,
        comparison=comparison,
    )


def _parse_timestamp(value: str) -> datetime:
    text = value.strip().removesuffix("Z")
    timestamp = datetime.fromisoformat(text)
    return timestamp.replace(tzinfo=None) if timestamp.tzinfo is not None else timestamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V4/V5 causal hourly proxy benchmark")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--series", type=int, required=True)
    parser.add_argument("--until", type=_parse_timestamp, required=True)
    parser.add_argument("--query-workers", type=int, default=21)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = benchmark_series(
        args.source_root,
        series_id=int(args.series),
        until=args.until,
        query_workers=int(args.query_workers),
        work_root=args.work_root,
    )
    payload = asdict(report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.comparison.passes and report.v5_seconds < report.v4_seconds else 1


if __name__ == "__main__":
    raise SystemExit(main())
