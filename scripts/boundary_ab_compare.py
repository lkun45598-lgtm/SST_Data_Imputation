from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.config import RunConfig
from Data_Imputation.retraining.gradient_prior_v2_natural_gap.losses import (
    boundary_ramp_composition,
    boundary_transition_composition,
)
from Data_Imputation.retraining.gradient_prior_v2_natural_gap.train import _load_model
from Data_Imputation.retraining.gradient_prior_v2_natural_gap.visualize_realcloud_style import _make_sample


def edge_values(field: np.ndarray, natural: np.ndarray, land: np.ndarray) -> np.ndarray:
    preserved = (~natural) & (~land)
    vals = []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.full_like(field, np.nan, dtype=np.float32)
        if dy == 1:
            shifted[1:, :] = field[:-1, :]
            valid = natural & (~land) & np.roll(preserved, 1, axis=0)
        elif dy == -1:
            shifted[:-1, :] = field[1:, :]
            valid = natural & (~land) & np.roll(preserved, -1, axis=0)
        elif dx == 1:
            shifted[:, 1:] = field[:, :-1]
            valid = natural & (~land) & np.roll(preserved, 1, axis=1)
        else:
            shifted[:, :-1] = field[:, 1:]
            valid = natural & (~land) & np.roll(preserved, -1, axis=1)
        vals.append(np.abs(field - shifted)[valid])
    return np.concatenate(vals) if vals else np.empty(0, dtype=np.float32)


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    q = np.percentile(values, [50, 90, 95, 99, 100])
    return {"count": int(values.size), "p50": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": float(q[4])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--stats-path", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[150, 179, 223])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    config = RunConfig(data_dir=args.data_path.parent, output_dir=args.checkpoint.parent,
                       pretrained_path=args.checkpoint, gradient_stats_path=args.stats_path,
                       experiment_name=args.checkpoint.stem, epochs=1, device=str(device))
    model = _load_model(config, device)
    model.eval()
    rows = []
    with h5py.File(args.data_path, "r") as handle:
        land_np = np.asarray(handle["land_mask"][:], dtype=bool)
        for frame in args.frames:
            sample = _make_sample(handle, frame, mean=config.norm_mean, std=config.norm_std)
            sst = torch.from_numpy(np.asarray(sample["input_sst_seq"], dtype=np.float32)).unsqueeze(0).to(device)
            masks = torch.from_numpy(np.asarray(sample["mask_seq"], dtype=np.float32)).unsqueeze(0).to(device)
            natural = torch.from_numpy(np.asarray(sample["natural_mask"], dtype=bool)).unsqueeze(0).to(device)
            land = torch.from_numpy(np.asarray(sample["land_mask"], dtype=bool)).unsqueeze(0).to(device)
            proxy = torch.from_numpy(np.nan_to_num((np.asarray(sample["target_proxy"], dtype=np.float32) - config.norm_mean) / config.norm_std, nan=0.0)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.inference_mode():
                raw = model(sst, masks)
                old = boundary_transition_composition(raw, sst[:, -1:], proxy, natural, land)
                new = boundary_ramp_composition(raw, sst[:, -1:], proxy, natural, land,
                                                max_layers=3, max_seam_gradient=0.6 / config.norm_std)
            n = np.asarray(sample["natural_mask"], dtype=bool)
            l = np.asarray(sample["land_mask"], dtype=bool)
            old_k = (old[0, 0].cpu().numpy() * config.norm_std)
            new_k = (new[0, 0].cpu().numpy() * config.norm_std)
            raw_k = raw[0, 0].cpu().numpy() * config.norm_std
            rows.append({"frame": frame, "timestamp": str(sample["timestamp"]),
                         "raw": stats(edge_values(raw_k, n, l)),
                         "old_transition": stats(edge_values(old_k, n, l)),
                         "new_ramp": stats(edge_values(new_k, n, l)),
                         "old_changed_pixels": int(np.count_nonzero(np.abs(old_k - raw_k) > 1e-6)),
                         "new_changed_pixels": int(np.count_nonzero(np.abs(new_k - raw_k) > 1e-6))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
