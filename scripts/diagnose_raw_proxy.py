from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.config import RunConfig
from Data_Imputation.retraining.gradient_prior_v2_natural_gap.losses import boundary_transition_composition
from Data_Imputation.retraining.gradient_prior_v2_natural_gap.train import _load_model
from Data_Imputation.retraining.gradient_prior_v2_natural_gap.visualize_realcloud_style import _make_sample


def main() -> None:
    root = Path("/home/lz/SST_Data_Imputation")
    ckpt = root / "Data_Imputation/experiments/gradient_prior_v2_natural_gap/h12_proxy_smooth_v1_20260904_1740/best_model.pth"
    data = root / "Data_Imputation/experiments/hourly_data_proxy_smooth_v1_20260904_full/h12/jaxa_knn_filled_08.h5"
    stats = root / "Data_Imputation/experiments/gradient_prior_v2_natural_gap/stats/h12_train_monthly_v3.json"
    device = torch.device("cuda:0")
    cfg = RunConfig(data_dir=data.parent, output_dir=ckpt.parent, pretrained_path=ckpt,
                    gradient_stats_path=stats, experiment_name=ckpt.stem, epochs=1,
                    device=str(device))
    model = _load_model(cfg, device)
    model.eval()
    with h5py.File(data, "r") as handle:
        land = np.asarray(handle["land_mask"][:], dtype=bool)
        for frame in (150, 179, 223):
            sample = _make_sample(handle, frame, mean=cfg.norm_mean, std=cfg.norm_std)
            x = torch.from_numpy(np.asarray(sample["input_sst_seq"], dtype=np.float32))[None].to(device)
            masks = torch.from_numpy(np.asarray(sample["mask_seq"], dtype=np.float32))[None].to(device)
            natural = torch.from_numpy(np.asarray(sample["natural_mask"], dtype=bool))[None].to(device)
            lm = torch.from_numpy(np.asarray(sample["land_mask"], dtype=bool))[None].to(device)
            proxy = torch.from_numpy(np.nan_to_num((np.asarray(sample["target_proxy"], dtype=np.float32) - cfg.norm_mean) / cfg.norm_std, nan=0.0))[None, None].to(device)
            with torch.inference_mode():
                raw = model(x, masks)
                composed = boundary_transition_composition(raw, x[:, -1:], proxy, natural, lm)
            raw_k = raw[0, 0].cpu().numpy() * cfg.norm_std + cfg.norm_mean
            comp_k = composed[0, 0].cpu().numpy() * cfg.norm_std + cfg.norm_mean
            proxy_k = np.asarray(sample["target_proxy"], dtype=np.float32)
            nm = np.asarray(sample["natural_mask"], dtype=bool) & ~land
            d_raw = np.abs(raw_k - proxy_k)[nm]
            d_comp = np.abs(comp_k - proxy_k)[nm]
            a, b = raw_k[nm], proxy_k[nm]
            result = {
                "frame": frame, "timestamp": str(sample["timestamp"]),
                "natural_pixels": int(nm.sum()),
                "raw_proxy_mae": float(np.mean(d_raw)),
                "raw_proxy_p50": float(np.percentile(d_raw, 50)),
                "raw_proxy_p90": float(np.percentile(d_raw, 90)),
                "raw_proxy_p99": float(np.percentile(d_raw, 99)),
                "raw_proxy_corr": float(np.corrcoef(a, b)[0, 1]),
                "composed_proxy_mae": float(np.mean(d_comp)),
                "raw_composed_mae": float(np.mean(np.abs(raw_k - comp_k)[nm])),
                "raw_std": float(np.std(a)), "proxy_std": float(np.std(b)),
                "raw_proxy_bias": float(np.mean(a - b)),
            }
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
