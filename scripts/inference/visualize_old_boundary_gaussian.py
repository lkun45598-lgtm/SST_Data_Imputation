#!/usr/bin/env python3
"""Render a Gaussian-postprocessed counterpart of an existing real-cloud plot.

This is intentionally a one-off, non-destructive visualisation utility.  It
imports the deployment renderer already used for the raw image, applies a
spatial Gaussian filter to its FNO prediction, and writes filtered values only
inside the natural cloud-gap ocean mask.  Observations and land are preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

# When copied to /tmp for the remote run, the repository is no longer on
# Python's default import path.
_repo_root = Path("/home/lz/SST_Data_Imputation")
if _repo_root.exists():
    sys.path.insert(0, str(_repo_root))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap import (
    visualize_realcloud_style as renderer,
)


def gaussian_keep_preserved(
    field: np.ndarray,
    *,
    natural_mask: np.ndarray,
    land_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Smooth only natural-gap ocean pixels, retaining all preserved values."""
    values = np.asarray(field, dtype=np.float32)
    natural = np.asarray(natural_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    if values.shape != natural.shape or values.shape != land.shape:
        raise ValueError("field and masks must have equal shapes")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    ocean_valid = ~land & np.isfinite(values)
    if not ocean_valid.any():
        return values.copy()
    working = values.copy()
    working[~ocean_valid] = float(np.mean(values[ocean_valid]))
    filtered = gaussian_filter(working, sigma=float(sigma))
    write_mask = natural & ocean_valid
    return np.where(write_mask, filtered, values).astype(np.float32, copy=False)


def render_gaussian_counterpart(
    *,
    checkpoint: Path,
    data_path: Path,
    stats_path: Path,
    output: Path,
    frame: int,
    device_name: str,
    sigma: float,
) -> dict[str, object]:
    """Use the existing renderer's inference and layout with masked smoothing."""
    raw_predict = renderer.predict_sample
    smoothing_audit: dict[str, object] = {}

    def predict_with_gaussian(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        prediction = dict(raw_predict(*args, **kwargs))
        sample = args[1]
        if not isinstance(sample, dict):
            raise TypeError("expected the renderer sample as the second argument")
        raw_field = np.asarray(prediction["prediction"], dtype=np.float32)
        smoothed = gaussian_keep_preserved(
            raw_field,
            natural_mask=np.asarray(sample["natural_mask"], dtype=bool),
            land_mask=np.asarray(sample["land_mask"], dtype=bool),
            sigma=sigma,
        )
        natural = np.asarray(sample["natural_mask"], dtype=bool)
        land = np.asarray(sample["land_mask"], dtype=bool)
        changed = smoothed - raw_field
        preserved = ~(natural & ~land)
        if not np.array_equal(smoothed[preserved], raw_field[preserved], equal_nan=True):
            raise RuntimeError("Gaussian post-processing modified a preserved pixel")
        if not np.array_equal(smoothed[land], raw_field[land], equal_nan=True):
            raise RuntimeError("Gaussian post-processing modified a land pixel")
        if not np.isfinite(smoothed[~land]).all():
            raise RuntimeError("Gaussian post-processing introduced non-finite ocean values")
        values = np.abs(changed[natural & ~land])
        smoothing_audit.update(
            {
                "write_pixel_count": int(values.size),
                "mean_abs_change_kelvin": float(values.mean()) if values.size else 0.0,
                "p95_abs_change_kelvin": float(np.percentile(values, 95)) if values.size else 0.0,
                "max_abs_change_kelvin": float(values.max()) if values.size else 0.0,
                "preserved_observed_and_land": True,
            }
        )
        prediction["prediction"] = smoothed
        return prediction

    renderer.predict_sample = predict_with_gaussian
    try:
        manifest = renderer.render(
            checkpoint=checkpoint,
            data_path=data_path,
            stats_path=stats_path,
            output=output,
            frame=frame,
            device_name=device_name,
        )
    finally:
        renderer.predict_sample = raw_predict

    manifest.pop("no_gaussian_postprocessing", None)
    manifest["gaussian_postprocessing"] = {
        "sigma_pixels": float(sigma),
        "filter": "scipy.ndimage.gaussian_filter",
        "write_region": "natural cloud-gap ocean pixels only",
        "filter_context": "full field; observed ocean values participate but are preserved",
        **smoothing_audit,
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=179)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pdf").exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    print(
        json.dumps(
            render_gaussian_counterpart(
                checkpoint=args.checkpoint,
                data_path=args.data_path,
                stats_path=args.stats_path,
                output=args.output,
                frame=args.frame,
                device_name=args.device,
                sigma=args.sigma,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
