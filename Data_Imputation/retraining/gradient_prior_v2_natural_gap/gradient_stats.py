"""Strict month-conditioned observed-SST gradient statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from . import PROTOCOL_VERSION
from .config import TRAIN_SERIES


QUANTILES = {
    "p50": 0.50,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
    "p99_5": 0.995,
    "p99_9": 0.999,
}


STATS_SCHEMA_VERSION = "monthly-gradient-proxy-v3"


def _zero_padded_box_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Return zero-padded square-window sums without materializing windows."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a 2-D array")
    radius = window // 2
    padded = np.pad(array, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(
        axis=0, dtype=np.float64
    ).cumsum(axis=1, dtype=np.float64)
    return (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    ).astype(np.float32)


def strict_observed_lowpass_residual(
    field: np.ndarray,
    original_observed_mask: np.ndarray,
    land_mask: np.ndarray,
    *,
    window: int,
    min_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed-only low-pass residuals and their trusted centres in K."""
    if not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support must be within [0, 1]")
    values = np.asarray(field, dtype=np.float32)
    observed = np.asarray(original_observed_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if values.ndim != 2 or observed.shape != values.shape or land.shape != values.shape:
        raise ValueError("field, observation mask, and land mask must be equal-shape 2-D arrays")

    valid_observed_ocean = observed & ~land & np.isfinite(values)
    support = _zero_padded_box_sum(valid_observed_ocean.astype(np.float32), window)
    total = _zero_padded_box_sum(
        np.where(valid_observed_ocean, values, 0.0), window
    )
    lowpass = np.divide(
        total,
        support,
        out=np.full(values.shape, np.nan, dtype=np.float32),
        where=support > 0,
    )
    valid = valid_observed_ocean & (support >= min_support * window**2)
    residual = np.where(valid, np.abs(values - lowpass), np.nan).astype(np.float32)
    return residual, valid


def strict_central_gradient(
    field: np.ndarray,
    original_observed_mask: np.ndarray,
    land_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute central K/pixel gradients only for complete observed-ocean stencils."""
    values = np.asarray(field, dtype=np.float32)
    observed = np.asarray(original_observed_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if values.ndim != 2 or observed.shape != values.shape or land.shape != values.shape:
        raise ValueError("field, observation mask, and land mask must be equal-shape 2-D arrays")

    finite_observed_ocean = observed & ~land & np.isfinite(values)
    valid = np.zeros(values.shape, dtype=bool)
    magnitude = np.full(values.shape, np.nan, dtype=np.float32)
    if min(values.shape) < 3:
        return magnitude, valid

    valid_center = (
        finite_observed_ocean[1:-1, 1:-1]
        & finite_observed_ocean[:-2, 1:-1]
        & finite_observed_ocean[2:, 1:-1]
        & finite_observed_ocean[1:-1, :-2]
        & finite_observed_ocean[1:-1, 2:]
    )
    gy = 0.5 * (values[2:, 1:-1] - values[:-2, 1:-1])
    gx = 0.5 * (values[1:-1, 2:] - values[1:-1, :-2])
    valid[1:-1, 1:-1] = valid_center
    magnitude[1:-1, 1:-1] = np.where(valid_center, np.hypot(gx, gy), np.nan)
    return magnitude, valid


def strict_adjacent_edge_magnitude(
    field: np.ndarray,
    original_observed_mask: np.ndarray,
    land_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute adjacent differences for complete observed-ocean pairs."""
    values = np.asarray(field, dtype=np.float32)
    observed = np.asarray(original_observed_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if values.ndim != 2 or observed.shape != values.shape or land.shape != values.shape:
        raise ValueError("field, observation mask, and land mask must be equal-shape 2-D arrays")
    valid_pixels = observed & ~land & np.isfinite(values)
    edge_y = np.abs(values[1:, :] - values[:-1, :])
    edge_x = np.abs(values[:, 1:] - values[:, :-1])
    valid_y = valid_pixels[1:, :] & valid_pixels[:-1, :]
    valid_x = valid_pixels[:, 1:] & valid_pixels[:, :-1]
    return (
        np.concatenate([edge_y[valid_y], edge_x[valid_x]]),
        np.concatenate([valid_y.reshape(-1), valid_x.reshape(-1)]),
    )


class GradientHistogram:
    """Fixed-bin accumulator whose summaries are independent of frame order."""

    def __init__(self, *, bin_width: float = 0.0001, max_gradient: float = 8.0):
        if bin_width <= 0 or max_gradient <= bin_width:
            raise ValueError("bin_width and max_gradient are invalid")
        bins_float = max_gradient / bin_width
        if not np.isclose(bins_float, round(bins_float), rtol=0, atol=1e-8):
            raise ValueError("max_gradient must be an integer multiple of bin_width")
        self.bin_width = float(bin_width)
        self.max_gradient = float(max_gradient)
        self.counts = np.zeros(int(round(bins_float)), dtype=np.int64)
        self.total_count = 0
        self.overflow_count = 0

    @property
    def in_range_count(self) -> int:
        return int(self.counts.sum())

    def update(self, values: np.ndarray) -> None:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        flattened = flattened[np.isfinite(flattened) & (flattened >= 0)]
        self.total_count += int(flattened.size)
        in_range = flattened[flattened <= self.max_gradient]
        self.overflow_count += int(flattened.size - in_range.size)
        if not in_range.size:
            return
        indices = np.floor(in_range / self.bin_width).astype(np.int64)
        indices = np.minimum(indices, self.counts.size - 1)
        self.counts += np.bincount(indices, minlength=self.counts.size)

    def _quantile(self, probability: float) -> float | None:
        count = self.in_range_count
        if count == 0:
            return None
        rank = max(1, int(np.ceil(probability * count)))
        index = int(np.searchsorted(np.cumsum(self.counts), rank, side="left"))
        return float((index + 0.5) * self.bin_width)

    def summary(self) -> dict[str, int | float | None]:
        result: dict[str, int | float | None] = {
            "count": int(self.total_count),
            "in_range_count": self.in_range_count,
            "overflow_count": int(self.overflow_count),
        }
        result.update({name: self._quantile(value) for name, value in QUANTILES.items()})
        return result


def _validate_training_series(series_ids: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in series_ids)
    if tuple(sorted(normalized)) != TRAIN_SERIES:
        raise ValueError("monthly gradient thresholds must use exactly JAXA Series 0-6")
    return normalized


def _input_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def compute_monthly_gradient_stats(
    data_dir: Path | str,
    *,
    series_ids: Iterable[int] = TRAIN_SERIES,
    bin_width: float = 0.0001,
    max_gradient: float = 8.0,
    proxy_lowpass_window: int = 15,
    proxy_min_observed_support: float = 0.8,
) -> dict[str, object]:
    """Scan all training observations and return deterministic monthly summaries."""
    if proxy_lowpass_window < 3 or proxy_lowpass_window % 2 == 0:
        raise ValueError("proxy_lowpass_window must be an odd integer >= 3")
    if not 0.0 <= proxy_min_observed_support <= 1.0:
        raise ValueError("proxy_min_observed_support must be within [0, 1]")
    normalized_series = _validate_training_series(series_ids)
    data_root = Path(data_dir)
    histograms = {
        month: {
            "central": GradientHistogram(bin_width=bin_width, max_gradient=max_gradient),
            "edge": GradientHistogram(bin_width=bin_width, max_gradient=max_gradient),
            "proxy_highpass": GradientHistogram(
                bin_width=bin_width,
                max_gradient=max_gradient,
            ),
        }
        for month in range(1, 13)
    }
    identities = []

    for series_id in normalized_series:
        path = data_root / f"jaxa_knn_filled_{series_id:02d}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        identities.append(_input_identity(path))
        with h5py.File(path, "r") as handle:
            land = np.asarray(handle["land_mask"][:], dtype=bool)
            frame_count = int(handle["sst_data"].shape[0])
            if handle["original_obs_mask"].shape[0] != frame_count:
                raise ValueError(f"observation-mask frame mismatch in {path}")
            timestamps = handle["timestamps"][:]
            if len(timestamps) != frame_count:
                raise ValueError(f"timestamp frame mismatch in {path}")
            for frame_index in range(frame_count):
                raw_timestamp = timestamps[frame_index]
                timestamp = (
                    raw_timestamp.decode("utf-8")
                    if isinstance(raw_timestamp, bytes)
                    else str(raw_timestamp)
                )
                month = int(timestamp[5:7])
                field = np.asarray(handle["sst_data"][frame_index], dtype=np.float32)
                observed = np.asarray(
                    handle["original_obs_mask"][frame_index], dtype=bool
                )
                magnitude, valid = strict_central_gradient(field, observed, land)
                histograms[month]["central"].update(magnitude[valid])
                edge_values, _ = strict_adjacent_edge_magnitude(field, observed, land)
                histograms[month]["edge"].update(edge_values)
                residual, residual_valid = strict_observed_lowpass_residual(
                    field,
                    observed,
                    land,
                    window=proxy_lowpass_window,
                    min_support=proxy_min_observed_support,
                )
                histograms[month]["proxy_highpass"].update(residual[residual_valid])

    return {
        "protocol_version": PROTOCOL_VERSION,
        "statistics_schema": STATS_SCHEMA_VERSION,
        "algorithm": "strict-observed-gradient-and-lowpass-residual-v3",
        "gradient_units": "K/pixel",
        "proxy_highpass_units": "K",
        "proxy_lowpass_window": int(proxy_lowpass_window),
        "proxy_min_observed_support": float(proxy_min_observed_support),
        "series_ids": list(normalized_series),
        "bin_width": float(bin_width),
        "max_gradient": float(max_gradient),
        "inputs": identities,
        "months": {
            str(month): {
                kind: histograms[month][kind].summary()
                for kind in ("central", "edge", "proxy_highpass")
            }
            for month in range(1, 13)
        },
    }


def _canonical_payload(document: dict[str, object]) -> tuple[dict[str, object], bytes]:
    payload = json.loads(json.dumps(document))
    payload.pop("sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, encoded


def write_gradient_stats(document: dict[str, object], path: Path | str) -> str:
    """Write canonical statistics once and return their content digest."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite gradient statistics: {destination}")
    payload, encoded = _canonical_payload(document)
    digest = hashlib.sha256(encoded).hexdigest()
    output = dict(payload)
    output["sha256"] = digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return digest


def load_gradient_stats(path: Path | str) -> dict[str, object]:
    source = Path(path)
    document = json.loads(source.read_text())
    saved_digest = document.get("sha256")
    payload, encoded = _canonical_payload(document)
    actual_digest = hashlib.sha256(encoded).hexdigest()
    if not isinstance(saved_digest, str) or saved_digest != actual_digest:
        raise ValueError(f"gradient-statistics digest mismatch: {source}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported gradient-statistics protocol version")
    if payload.get("statistics_schema") != STATS_SCHEMA_VERSION:
        raise ValueError("unsupported gradient-statistics schema")
    if payload.get("algorithm") != "strict-observed-gradient-and-lowpass-residual-v3":
        raise ValueError("unsupported gradient-statistics algorithm")
    if payload.get("gradient_units") != "K/pixel":
        raise ValueError("gradient statistics must use K/pixel")
    if payload.get("proxy_highpass_units") != "K":
        raise ValueError("proxy high-pass statistics must use K")
    window = payload.get("proxy_lowpass_window")
    if not isinstance(window, int) or window < 3 or window % 2 == 0:
        raise ValueError("proxy low-pass window must be an odd integer >= 3")
    support = payload.get("proxy_min_observed_support")
    if not isinstance(support, (int, float)) or not 0.0 <= float(support) <= 1.0:
        raise ValueError("proxy minimum observed support must be within [0, 1]")
    if tuple(payload.get("series_ids", [])) != TRAIN_SERIES:
        raise ValueError("gradient statistics must come from Series 0-6")
    months = payload.get("months")
    if not isinstance(months, dict) or set(months) != {str(value) for value in range(1, 13)}:
        raise ValueError("gradient statistics must contain all 12 months")
    for month, summary in months.items():
        if not isinstance(summary, dict) or set(summary) != {
            "central",
            "edge",
            "proxy_highpass",
        }:
            raise ValueError(
                f"month {month} must contain central, edge, and proxy_highpass statistics"
            )
        for kind in ("central", "edge", "proxy_highpass"):
            distribution = summary[kind]
            if not isinstance(distribution, dict) or distribution.get("p99_9") is None:
                raise ValueError(f"month {month} {kind} has no P99.9 threshold")
            if not math.isfinite(float(distribution["p99_9"])) or float(distribution["p99_9"]) <= 0:
                raise ValueError(f"month {month} {kind} has a non-positive P99.9 threshold")
    document["sha256"] = saved_digest
    return document


def format_month_summary(month: int, summary: dict[str, object]) -> str:
    central = summary["central"]
    edge = summary["edge"]
    proxy_highpass = summary["proxy_highpass"]
    return (
        f"month={month:02d} central_count={central['count']} "
        f"edge_count={edge['count']} "
        f"proxy_highpass_count={proxy_highpass['count']} "
        f"central_p99.9={central['p99_9']:.4f} "
        f"edge_p99.9={edge['p99_9']:.4f} K/pixel "
        f"proxy_highpass_p99.9={proxy_highpass['p99_9']:.4f} K"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--series", type=int, nargs="+", default=list(TRAIN_SERIES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.0001)
    parser.add_argument("--max-gradient", type=float, default=8.0)
    parser.add_argument("--proxy-lowpass-window", type=int, default=15)
    parser.add_argument("--proxy-min-observed-support", type=float, default=0.8)
    args = parser.parse_args()

    stats = compute_monthly_gradient_stats(
        args.data_dir,
        series_ids=args.series,
        bin_width=args.bin_width,
        max_gradient=args.max_gradient,
        proxy_lowpass_window=args.proxy_lowpass_window,
        proxy_min_observed_support=args.proxy_min_observed_support,
    )
    digest = write_gradient_stats(stats, args.output)
    print(f"Wrote {args.output} (sha256={digest})")
    for month in range(1, 13):
        summary = stats["months"][str(month)]
        print(format_month_summary(month, summary))


if __name__ == "__main__":
    main()
