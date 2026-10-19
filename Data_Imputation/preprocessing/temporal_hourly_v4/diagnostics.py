"""Structured auditable diagnostics for v4 causal proxy generation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import scipy

from .pipeline import SeriesSummary


def _frame_records(summary: SeriesSummary) -> list[dict[str, object]]:
    return [
        {
            "timestamp": frame.timestamp.isoformat(),
            "raw_fraction": frame.raw_fraction,
            "temporal_fraction": frame.temporal_fraction,
            "knn_fraction": frame.knn_fraction,
            "proxy_history_fraction": frame.proxy_history_fraction,
            "mean_confidence": frame.mean_confidence,
            "unresolved_count": frame.unresolved_count,
        }
        for frame in summary.frames
    ]


def write_series_diagnostics(output_root: Path, summary: SeriesSummary) -> tuple[Path, Path]:
    """Write one series' per-frame CSV and JSON summary below its output root."""

    diagnostics_root = Path(output_root) / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    records = _frame_records(summary)
    csv_path = diagnostics_root / f"series_{summary.series_id:02d}.csv"
    json_path = diagnostics_root / f"series_{summary.series_id:02d}.json"
    fields = (
        "timestamp",
        "raw_fraction",
        "temporal_fraction",
        "knn_fraction",
        "proxy_history_fraction",
        "mean_confidence",
        "unresolved_count",
    )
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "series_id": summary.series_id,
        "processed_timestamps": [timestamp.isoformat() for timestamp in summary.processed_timestamps],
        "proxy_history_rejected_low_confidence": summary.proxy_history_rejected_low_confidence,
        "frames": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return csv_path, json_path


def write_manifest(
    output_root: Path,
    summaries: Iterable[SeriesSummary],
    *,
    parameters: Mapping[str, object],
) -> Path:
    """Write a root manifest only after the requested command succeeds."""

    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "parameters": dict(parameters),
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
        "series": [
            {
                "series_id": summary.series_id,
                "frame_count": len(summary.processed_timestamps),
                "proxy_history_rejected_low_confidence": summary.proxy_history_rejected_low_confidence,
            }
            for summary in summaries
        ],
    }
    path = Path(output_root) / "diagnostics" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path
