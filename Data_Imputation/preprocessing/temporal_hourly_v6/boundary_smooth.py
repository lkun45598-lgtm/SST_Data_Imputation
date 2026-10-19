"""Conservative local smoothing for natural-cloud proxy boundaries.

The processor is deliberately separate from the causal V6 generator.  It
copies an existing proxy H5 file, keeps original observations and land pixels
bitwise unchanged, and modifies only the first two natural-gap layers next to
original observed ocean.  The local estimate is a normalized positive cubic
B-spline kernel over observed anchors, so missing values, synthetic proxy
values, and land pixels cannot contribute to the estimate.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import h5py
import numpy as np
from scipy import ndimage


VERSION = "mask_aware_cubic_boundary_v1"
PROXY_SMOOTH_VERSION = "mask_aware_normalized_gaussian_v1"
_KERNEL_1D = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
_KERNEL_2D = np.outer(_KERNEL_1D, _KERNEL_1D)
_KERNEL_2D /= float(_KERNEL_2D.sum())


def _normalized_gaussian_estimate(
    values: np.ndarray,
    support_mask: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate values with a Gaussian normalized by finite local support.

    Unlike a convolution over a NaN-filled array, this computes ``G(x*m) /
    G(m)``.  Consequently, unsupported pixels do not contribute a fabricated
    global-mean value to the estimate.  The returned validity mask identifies
    pixels with positive Gaussian support.
    """
    values = np.asarray(values, dtype=np.float32)
    support = np.asarray(support_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("values must be a 2D frame")
    if support.shape != values.shape:
        raise ValueError("values and support_mask must have equal 2D shapes")
    if not np.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma must be finite and positive")

    support &= np.isfinite(values)
    numerator = ndimage.gaussian_filter(
        np.where(support, values, 0.0).astype(np.float64),
        sigma=float(sigma),
        mode="nearest",
    )
    denominator = ndimage.gaussian_filter(
        support.astype(np.float64),
        sigma=float(sigma),
        mode="nearest",
    )
    valid = denominator > np.finfo(np.float64).eps
    estimate = np.full(values.shape, np.nan, dtype=np.float32)
    estimate[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return estimate, valid


def _has_four_neighbor(mask: np.ndarray) -> np.ndarray:
    """Return positions having at least one four-connected true neighbor."""
    mask = np.asarray(mask, dtype=bool)
    adjacent = np.zeros_like(mask)
    adjacent[1:, :] |= mask[:-1, :]
    adjacent[:-1, :] |= mask[1:, :]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    return adjacent


def boundary_layers(
    observed_mask: np.ndarray,
    natural_mask: np.ndarray,
    land_mask: np.ndarray,
    *,
    max_layers: int = 2,
) -> np.ndarray:
    """Label natural-gap pixels at four-neighbor distance one or two.

    ``0`` denotes all other pixels, ``1`` the first natural layer next to an
    original observed ocean pixel, and ``2`` the next inward natural layer.
    Natural pixels are always excluded from the observed anchors.
    """
    observed = np.asarray(observed_mask, dtype=bool)
    natural = np.asarray(natural_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if observed.shape != natural.shape or observed.shape != land.shape:
        raise ValueError("observed, natural, and land masks must have equal 2D shapes")
    if int(max_layers) < 1:
        raise ValueError("max_layers must be positive")

    observed_ocean = observed & ~land
    natural_ocean = natural & ~observed_ocean & ~land
    layers = np.zeros(natural.shape, dtype=np.uint8)
    frontier = observed_ocean
    for depth in range(1, int(max_layers) + 1):
        selected = natural_ocean & _has_four_neighbor(frontier) & (layers == 0)
        layers[selected] = np.uint8(depth)
        frontier = selected
        if not np.any(frontier):
            break
    return layers


def _mask_aware_cubic_estimate(
    values: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate each pixel from finite anchors with a normalized cubic kernel."""
    values = np.asarray(values, dtype=np.float32)
    anchors = np.asarray(anchor_mask, dtype=bool) & np.isfinite(values)
    numerator = ndimage.convolve(
        np.where(anchors, values, 0.0).astype(np.float64),
        _KERNEL_2D,
        mode="nearest",
    )
    support = ndimage.convolve(
        anchors.astype(np.float64),
        _KERNEL_2D,
        mode="nearest",
    )
    estimate = np.full(values.shape, np.nan, dtype=np.float32)
    valid = support > np.finfo(np.float64).eps
    estimate[valid] = (numerator[valid] / support[valid]).astype(np.float32)
    return estimate, valid


def smooth_boundary_frame(
    values: np.ndarray,
    observed_mask: np.ndarray,
    natural_mask: np.ndarray,
    land_mask: np.ndarray,
    *,
    first_layer_weight: float = 0.75,
    second_layer_weight: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth only the natural-gap boundary layers of one proxy frame.

    The local estimate uses original observed ocean anchors only.  For layer 1
    and layer 2, respectively, it is blended with the existing proxy value by
    ``first_layer_weight`` and ``second_layer_weight``.  All other pixels are
    returned unchanged.  The returned ``changed`` mask records eligible
    boundary pixels where a finite estimate was written; ``layers`` records
    their geometric layer regardless of whether the numeric value changed.
    """
    values = np.asarray(values, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    natural = np.asarray(natural_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("values must be a 2D frame")
    if values.shape != observed.shape or values.shape != natural.shape or values.shape != land.shape:
        raise ValueError("values and masks must have equal 2D shapes")
    if not 0.0 <= float(first_layer_weight) <= 1.0:
        raise ValueError("first_layer_weight must lie in [0, 1]")
    if not 0.0 <= float(second_layer_weight) <= 1.0:
        raise ValueError("second_layer_weight must lie in [0, 1]")

    layers = boundary_layers(observed, natural, land, max_layers=2)
    anchors = observed & ~land & np.isfinite(values)
    estimate, estimate_valid = _mask_aware_cubic_estimate(values, anchors)
    eligible = (layers > 0) & estimate_valid & ~land
    result = values.copy()
    changed = np.zeros(values.shape, dtype=bool)
    if np.any(eligible):
        weight = np.where(
            layers == 1,
            float(first_layer_weight),
            float(second_layer_weight),
        ).astype(np.float32)
        old = result[eligible]
        local = estimate[eligible]
        old_finite = np.isfinite(old)
        blended = weight[eligible] * local + (1.0 - weight[eligible]) * old
        # A malformed/partially filled proxy value must not erase a valid
        # local estimate.  This branch is normally unused for the H12 files,
        # whose ocean pixels are already finite, but keeps the postprocessor
        # safe for other proxy versions.
        result[eligible] = np.where(old_finite, blended, local).astype(np.float32)
        changed[eligible] = np.isfinite(result[eligible])
    return result, changed, layers


def smooth_synthetic_frame(
    values: np.ndarray,
    observed_mask: np.ndarray,
    natural_mask: np.ndarray,
    source_code: np.ndarray,
    land_mask: np.ndarray,
    *,
    sigma: float = 0.8,
    first_layer_weight: float = 0.25,
    second_layer_weight: float = 0.12,
    interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservatively smooth finite synthetic ocean pixels.

    Original observed ocean pixels and land pixels are protected bitwise.  The
    local estimate uses observed finite ocean pixels only; temporal/KNN values
    are never used as anchors for their own smoothing.  Natural-gap layer 1
    and layer 2 receive stronger blends than deeper synthetic pixels, while a
    hard value-change cap prevents an isolated proxy artefact from being
    replaced by a large correction.
    """
    values = np.asarray(values, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    natural = np.asarray(natural_mask, dtype=bool)
    codes = np.asarray(source_code, dtype=np.uint8)
    land = np.asarray(land_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("values must be a 2D frame")
    if not (values.shape == observed.shape == natural.shape == codes.shape == land.shape):
        raise ValueError("values and masks must have equal 2D shapes")
    weights = (first_layer_weight, second_layer_weight, interior_weight)
    if any(not np.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0 for weight in weights):
        raise ValueError("smoothing weights must be finite and lie in [0, 1]")
    if not np.isfinite(float(max_change_k)) or float(max_change_k) < 0.0:
        raise ValueError("max_change_k must be finite and non-negative")

    observed_ocean = observed & ~land & np.isfinite(values)
    layers = boundary_layers(observed, natural, land, max_layers=2)
    estimate, estimate_valid = _normalized_gaussian_estimate(
        values, observed_ocean, sigma=float(sigma)
    )
    synthetic = ((codes == 2) | (codes == 3)) & ~observed & ~land & np.isfinite(values)
    eligible = synthetic & estimate_valid
    result = values.copy()
    changed = np.zeros(values.shape, dtype=bool)
    if not np.any(eligible):
        return result, changed, layers

    blend_weight = np.full(values.shape, float(interior_weight), dtype=np.float32)
    blend_weight[layers == 1] = float(first_layer_weight)
    blend_weight[layers == 2] = float(second_layer_weight)
    delta = (estimate - values) * blend_weight
    # Leave one float32 ulp of headroom and check the *persisted* float32
    # value.  At large SST magnitudes, adding 0.10 K can round to 0.100006 K;
    # stepping one representable value back toward the original removes that
    # storage-level overshoot without weakening the requested bound.
    cap = np.nextafter(np.float32(max_change_k), np.float32(0.0)).item()
    delta = np.clip(delta, -float(cap), float(cap))
    candidate = (values + delta).astype(np.float32)
    persisted_delta = candidate - values
    over = np.abs(persisted_delta) > float(max_change_k)
    if np.any(over):
        candidate[over] = np.nextafter(candidate[over], values[over])
        persisted_delta = candidate - values
        still_over = np.abs(persisted_delta) > float(max_change_k)
        if np.any(still_over):
            candidate[still_over] = np.nextafter(candidate[still_over], values[still_over])
    result[eligible] = candidate[eligible]
    changed = eligible & (result != values)
    return result, changed, layers


def _require_source_schema(handle: h5py.File) -> None:
    required = {
        "sst_data",
        "original_obs_mask",
        "original_missing_mask",
        "land_mask",
        "timestamps",
    }
    missing = required.difference(handle.keys())
    if missing:
        raise KeyError(f"source H5 is missing datasets: {sorted(missing)}")
    sst = handle["sst_data"]
    expected = sst.shape
    if len(expected) != 3:
        raise ValueError("sst_data must have shape (frames, height, width)")
    for name in ("original_obs_mask", "original_missing_mask"):
        if handle[name].shape != expected:
            raise ValueError(f"{name} shape does not match sst_data")
    if handle["land_mask"].shape != expected[1:]:
        raise ValueError("land_mask shape does not match sst_data")
    if handle["timestamps"].shape != (expected[0],):
        raise ValueError("timestamps shape does not match sst_data")


def process_h5_file(source: Path | str, output: Path | str) -> dict[str, int | float | str]:
    """Copy and process one H5 file, refusing to overwrite an existing file.

    Processing happens in a sibling ``.part`` file and is published only after
    all frames and audit attributes have been written successfully.  This
    prevents an interrupted job from leaving a path that looks complete.
    """
    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("source and output must be different files")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    partial = output.with_name(output.name + ".part")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse incomplete output: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, partial)

    try:
        with h5py.File(partial, "r+") as handle:
            _require_source_schema(handle)
            if "boundary_smooth_mask" in handle or "boundary_smooth_layer" in handle:
                raise ValueError("output already contains boundary smoothing datasets")
            sst = handle["sst_data"]
            observed_ds = handle["original_obs_mask"]
            natural_ds = handle["original_missing_mask"]
            land = np.asarray(handle["land_mask"][:], dtype=bool)
            frames = int(sst.shape[0])
            ocean_count = int((~land).sum())
            smooth_mask_ds = handle.create_dataset(
                "boundary_smooth_mask",
                shape=sst.shape,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            )
            layer_ds = handle.create_dataset(
                "boundary_smooth_layer",
                shape=sst.shape,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            )
            modified_pixels = 0
            layer1_pixels = 0
            layer2_pixels = 0
            deltas: list[np.ndarray] = []
            for frame_index in range(frames):
                original = np.asarray(sst[frame_index], dtype=np.float32)
                observed = np.asarray(observed_ds[frame_index], dtype=bool)
                natural = np.asarray(natural_ds[frame_index], dtype=bool)
                smoothed, changed, layers = smooth_boundary_frame(
                    original, observed, natural, land
                )
                sst[frame_index] = smoothed
                smooth_mask_ds[frame_index] = changed.astype(np.uint8)
                layer_ds[frame_index] = layers
                modified_pixels += int(changed.sum())
                layer1_pixels += int((changed & (layers == 1)).sum())
                layer2_pixels += int((changed & (layers == 2)).sum())
                if np.any(changed):
                    deltas.append(np.abs(smoothed[changed] - original[changed]).astype(np.float64))
            handle.attrs["boundary_smooth_version"] = VERSION
            handle.attrs["boundary_smooth_layers"] = 2
            handle.attrs["boundary_smooth_first_layer_weight"] = 0.75
            handle.attrs["boundary_smooth_second_layer_weight"] = 0.50
            handle.attrs["boundary_smooth_kernel_1d"] = "[1,4,6,4,1]"
            handle.attrs["boundary_smooth_anchor_mask"] = "original observed ocean only"
            handle.attrs["boundary_smooth_complete"] = np.uint8(1)
            handle.flush()
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    delta_values = np.concatenate(deltas) if deltas else np.empty(0, dtype=np.float64)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "frames": frames,
        "ocean_pixels_per_frame": ocean_count,
        "modified_pixels": modified_pixels,
        "layer1_modified_pixels": layer1_pixels,
        "layer2_modified_pixels": layer2_pixels,
        "modified_fraction_of_ocean": modified_pixels / max(frames * ocean_count, 1),
        "max_abs_change_K": float(delta_values.max()) if delta_values.size else 0.0,
        "p95_abs_change_K": float(np.quantile(delta_values, 0.95)) if delta_values.size else 0.0,
        "p99_abs_change_K": float(np.quantile(delta_values, 0.99)) if delta_values.size else 0.0,
    }


def _require_synthetic_schema(handle: h5py.File) -> None:
    """Validate the provenance fields required by synthetic smoothing."""
    required = {
        "sst_data",
        "original_obs_mask",
        "original_missing_mask",
        "source_code",
        "land_mask",
        "timestamps",
    }
    missing = required.difference(handle.keys())
    if missing:
        raise KeyError(f"source H5 is missing datasets: {sorted(missing)}")
    expected = handle["sst_data"].shape
    if len(expected) != 3:
        raise ValueError("sst_data must have shape (frames, height, width)")
    for name in ("original_obs_mask", "original_missing_mask", "source_code"):
        if handle[name].shape != expected:
            raise ValueError(f"{name} shape does not match sst_data")
    if handle["land_mask"].shape != expected[1:]:
        raise ValueError("land_mask shape does not match sst_data")
    if handle["timestamps"].shape != (expected[0],):
        raise ValueError("timestamps shape does not match sst_data")

    observed = np.asarray(handle["original_obs_mask"][:], dtype=bool)
    land = np.asarray(handle["land_mask"][:], dtype=bool)
    values = np.asarray(handle["sst_data"][:], dtype=np.float32)
    codes = np.asarray(handle["source_code"][:], dtype=np.uint8)
    if np.any(observed & land[None, ...]):
        raise ValueError("original_obs_mask marks land pixels as observed")
    if np.any(observed & (codes != 1)):
        raise ValueError("source_code for original observations must be RAW")
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError("original observations must have finite sst_data values")
    if np.any(land[None, ...] & (codes != 0)):
        raise ValueError("source_code for land pixels must be LAND")
    if np.any(land[None, ...] & np.isfinite(values)):
        raise ValueError("land pixels must remain NaN")


def process_synthetic_h5_file(
    source: Path | str,
    output: Path | str,
    *,
    sigma: float = 0.8,
    first_layer_weight: float = 0.25,
    second_layer_weight: float = 0.12,
    interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> dict[str, int | float | str]:
    """Copy one proxy H5 and conservatively smooth synthetic pixels.

    The operation is atomic and refuses to overwrite an existing destination.
    All provenance datasets are copied from ``source``; only finite synthetic
    ``sst_data`` values are eligible for modification.  Audit masks identify
    the modified pixels and their natural-gap boundary layer.
    """
    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("source and output must be different files")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    partial = output.with_name(output.name + ".part")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse incomplete output: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, partial)
    try:
        with h5py.File(partial, "r+") as handle:
            _require_synthetic_schema(handle)
            sst = handle["sst_data"]
            observed_ds = handle["original_obs_mask"]
            natural_ds = handle["original_missing_mask"]
            codes_ds = handle["source_code"]
            land = np.asarray(handle["land_mask"][:], dtype=bool)
            frames = int(sst.shape[0])
            values_per_frame = int(np.prod(sst.shape[1:]))
            smooth_mask_ds = handle.create_dataset(
                "proxy_smooth_mask",
                shape=sst.shape,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            )
            layer_ds = handle.create_dataset(
                "proxy_smooth_layer",
                shape=sst.shape,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            )
            modified_pixels = 0
            changed_values: list[np.ndarray] = []
            for frame_index in range(frames):
                values = np.asarray(sst[frame_index], dtype=np.float32)
                observed = np.asarray(observed_ds[frame_index], dtype=bool)
                natural = np.asarray(natural_ds[frame_index], dtype=bool)
                source_code = np.asarray(codes_ds[frame_index], dtype=np.uint8)
                smoothed, changed, layers = smooth_synthetic_frame(
                    values,
                    observed,
                    natural,
                    source_code,
                    land,
                    sigma=sigma,
                    first_layer_weight=first_layer_weight,
                    second_layer_weight=second_layer_weight,
                    interior_weight=interior_weight,
                    max_change_k=max_change_k,
                )
                sst[frame_index] = smoothed
                smooth_mask_ds[frame_index] = changed.astype(np.uint8)
                layer_ds[frame_index] = layers
                modified_pixels += int(changed.sum())
                if np.any(changed):
                    changed_values.append(
                        np.abs(smoothed[changed] - values[changed]).astype(np.float64)
                    )
            handle.attrs["proxy_smoothing_version"] = PROXY_SMOOTH_VERSION
            handle.attrs["proxy_smoothing_sigma"] = float(sigma)
            handle.attrs["proxy_smoothing_first_layer_weight"] = float(first_layer_weight)
            handle.attrs["proxy_smoothing_second_layer_weight"] = float(second_layer_weight)
            handle.attrs["proxy_smoothing_interior_weight"] = float(interior_weight)
            handle.attrs["proxy_smoothing_max_change_k"] = float(max_change_k)
            handle.attrs["proxy_smoothing_anchor_mask"] = "finite original observed ocean only"
            handle.attrs["proxy_smoothing_complete"] = np.uint8(1)
            handle.flush()
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    deltas = np.concatenate(changed_values) if changed_values else np.empty(0, dtype=np.float64)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "frames": frames,
        "modified_pixels": modified_pixels,
        "modified_fraction": modified_pixels / max(frames * values_per_frame, 1),
        "max_abs_change_k": float(deltas.max()) if deltas.size else 0.0,
        "p95_abs_change_k": float(np.quantile(deltas, 0.95)) if deltas.size else 0.0,
        "p99_abs_change_k": float(np.quantile(deltas, 0.99)) if deltas.size else 0.0,
    }


def _process_worker(args: tuple[str, str]) -> dict[str, int | float | str]:
    return process_h5_file(Path(args[0]), Path(args[1]))


def process_series_root(
    input_root: Path | str,
    output_root: Path | str,
    *,
    series_ids: Iterable[int] = range(9),
    workers: int = 9,
) -> dict[str, object]:
    """Process selected series in parallel and write one audit manifest."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output root: {output_root}")
    if int(workers) < 1 or int(workers) > 192:
        raise ValueError("workers must lie in [1, 192]")
    ids = tuple(int(value) for value in series_ids)
    if not ids:
        raise ValueError("series_ids must not be empty")
    jobs: list[tuple[str, str]] = []
    for series_id in ids:
        source = input_root / f"jaxa_knn_filled_{series_id:02d}.h5"
        output = output_root / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        jobs.append((str(source), str(output)))
    output_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        results = list(executor.map(_process_worker, jobs))
    manifest = {
        "protocol_version": VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "series": results,
        "series_count": len(results),
        "frames": int(sum(int(item["frames"]) for item in results)),
        "modified_pixels": int(sum(int(item["modified_pixels"]) for item in results)),
    }
    (output_root / "boundary_smooth_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return manifest


def _process_synthetic_worker(args: tuple[str, str, float, float, float, float, float]) -> dict[str, int | float | str]:
    source, output, sigma, first, second, interior, cap = args
    return process_synthetic_h5_file(
        source,
        output,
        sigma=sigma,
        first_layer_weight=first,
        second_layer_weight=second,
        interior_weight=interior,
        max_change_k=cap,
    )


def process_hourly_root(
    input_root: Path | str,
    output_root: Path | str,
    *,
    series_ids: Iterable[int] = range(9),
    workers: int = 9,
    sigma: float = 0.8,
    first_layer_weight: float = 0.25,
    second_layer_weight: float = 0.12,
    interior_weight: float = 0.08,
    max_change_k: float = 0.10,
) -> dict[str, object]:
    """Copy and smooth a complete h00--h23 proxy root atomically per H5 file."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output root: {output_root}")
    if int(workers) < 1 or int(workers) > 192:
        raise ValueError("workers must lie in [1, 192]")
    ids = tuple(int(value) for value in series_ids)
    if not ids:
        raise ValueError("series_ids must not be empty")
    jobs: list[tuple[str, str, float, float, float, float, float]] = []
    for hour in range(24):
        for series_id in ids:
            source = input_root / f"h{hour:02d}" / f"jaxa_knn_filled_{series_id:02d}.h5"
            if not source.is_file():
                continue
            output = output_root / f"h{hour:02d}" / source.name
            jobs.append(
                (
                    str(source),
                    str(output),
                    float(sigma),
                    float(first_layer_weight),
                    float(second_layer_weight),
                    float(interior_weight),
                    float(max_change_k),
                )
            )
    if not jobs:
        raise FileNotFoundError(f"No matching H5 files below {input_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    for hour in range(24):
        (output_root / f"h{hour:02d}").mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        results = list(executor.map(_process_synthetic_worker, jobs))
    manifest = {
        "protocol_version": PROXY_SMOOTH_VERSION,
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "series_ids": list(ids),
        "file_count": len(results),
        "frames": int(sum(int(item["frames"]) for item in results)),
        "modified_pixels": int(sum(int(item["modified_pixels"]) for item in results)),
        "parameters": {
            "sigma": float(sigma),
            "first_layer_weight": float(first_layer_weight),
            "second_layer_weight": float(second_layer_weight),
            "interior_weight": float(interior_weight),
            "max_change_k": float(max_change_k),
        },
        "files": results,
    }
    (output_root / "proxy_smoothing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--series", type=int, nargs="+", default=list(range(9)))
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    manifest = process_series_root(
        args.input_root,
        args.output_root,
        series_ids=args.series,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
