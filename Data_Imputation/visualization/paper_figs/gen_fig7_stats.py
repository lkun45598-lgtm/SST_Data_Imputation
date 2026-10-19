"""Generate per-hour statistics for Figure 7 from the 24 final hourly models.

Part A evaluates each model on all eligible Series-8 target fields with the
same 30%, 55%, and 75% real-cloud-shaped extra masks as the main comparison.
Part B directly reconstructs the naturally missing regions in July 2024 and
summarizes the spatial-mean and pointwise diurnal cycles.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm


DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATA_IMPUTATION_DIR))

from inference.real_cloud_mask import RealCloudMaskGenerator, build_cloud_bank
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal


KNN_HOURLY_ROOT = DATA_IMPUTATION_DIR / "experiments/hourly_data"
HOURLY_MODEL_ROOT = DATA_IMPUTATION_DIR / "experiments"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
SERIES_TO_USE = 8
SEED = 0
DONOR_SERIES = [0, 3, 6]
MASK_LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]
DIURNAL_MONTH = "2024-07"
POINT_LAT = 19.0
POINT_LON = 115.0


def load_hourly_model(hour, device):
    checkpoint_path = (
        HOURLY_MODEL_ROOT
        / f"jaxa_finetune_h{hour:02d}_realmask_clean/best_model.pth"
    )
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6,
        cbam_reduction_ratio=16,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_mean = float(checkpoint.get("norm_mean", 299.9221))
    norm_std = float(checkpoint.get("norm_std", 2.6919))
    return model, norm_mean, norm_std


def make_window(sst_all, miss_all, target_index):
    indices = [
        max(0, target_index - (WINDOW_SIZE - 1) + offset)
        for offset in range(WINDOW_SIZE)
    ]
    return (
        np.stack([sst_all[index] for index in indices]).astype(np.float32),
        np.stack([miss_all[index] for index in indices]).astype(np.float32),
    )


def run_model(model, sst_sequence, mask_sequence, norm_mean, norm_std, device):
    normalized = np.nan_to_num(
        (sst_sequence - norm_mean) / norm_std, nan=0.0
    )
    sst_tensor = torch.from_numpy(normalized).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_sequence).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prediction = model(sst_tensor, mask_tensor)
    return prediction.squeeze().cpu().numpy() * norm_std + norm_mean


def predict_extra_mask(
    model,
    sst_sequence,
    missing_sequence,
    artificial_mask,
    norm_mean,
    norm_std,
    device,
):
    model_input = sst_sequence.copy()
    model_input[-1] = np.where(
        artificial_mask > 0, norm_mean, model_input[-1]
    )
    mask_input = missing_sequence.copy().astype(np.float32)
    mask_input[-1] = artificial_mask
    raw_prediction = run_model(
        model, model_input, mask_input, norm_mean, norm_std, device
    )
    return np.where(artificial_mask > 0, raw_prediction, sst_sequence[-1])


def predict_natural_gaps(
    model,
    sst_sequence,
    missing_sequence,
    observed_target,
    land_mask,
    norm_mean,
    norm_std,
    device,
):
    ocean = land_mask == 0
    target_missing = (observed_target == 0) & ocean
    model_input = sst_sequence.copy()
    model_input[-1] = np.where(
        target_missing, norm_mean, model_input[-1]
    )
    mask_input = missing_sequence.copy().astype(np.float32)
    mask_input[-1] = target_missing.astype(np.float32)
    raw_prediction = run_model(
        model, model_input, mask_input, norm_mean, norm_std, device
    )
    reconstruction = np.where(
        target_missing, raw_prediction, sst_sequence[-1]
    )
    return np.where(ocean, reconstruction, np.nan)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)) if values.size else np.nan,
        "std": float(np.std(values)) if values.size else np.nan,
    }


def compute_diurnal_for_hour(
    hour,
    model,
    sst_all,
    obs_all,
    miss_all,
    land,
    timestamps,
    latitude,
    longitude,
    norm_mean,
    norm_std,
    device,
):
    month_indices = [
        index
        for index, timestamp in enumerate(timestamps)
        if timestamp.startswith(DIURNAL_MONTH)
    ]
    point_y = int(np.argmin(np.abs(latitude - POINT_LAT)))
    point_x = int(np.argmin(np.abs(longitude - POINT_LON)))
    ocean = land == 0

    spatial_predictions = []
    spatial_observations = []
    point_predictions = []
    point_observations = []

    for target_index in tqdm(
        month_indices, desc=f"    H={hour:02d} natural gaps", leave=False
    ):
        sst_sequence, missing_sequence = make_window(
            sst_all, miss_all, target_index
        )
        reconstruction = predict_natural_gaps(
            model,
            sst_sequence,
            missing_sequence,
            obs_all[target_index],
            land,
            norm_mean,
            norm_std,
            device,
        )
        valid_reconstruction = ocean & np.isfinite(reconstruction)
        if valid_reconstruction.any():
            spatial_predictions.append(
                float(np.mean(reconstruction[valid_reconstruction]))
            )
        observed = (obs_all[target_index] > 0) & ocean
        if observed.any():
            spatial_observations.append(float(np.mean(sst_all[target_index][observed])))
        if np.isfinite(reconstruction[point_y, point_x]):
            point_predictions.append(float(reconstruction[point_y, point_x]))
        if observed[point_y, point_x]:
            point_observations.append(float(sst_all[target_index, point_y, point_x]))

    spatial_prediction_stats = summarize(spatial_predictions)
    spatial_observation_stats = summarize(spatial_observations)
    point_prediction_stats = summarize(point_predictions)
    point_observation_stats = summarize(point_observations)
    return {
        "n_frames": len(month_indices),
        "spatial_pred_mean": spatial_prediction_stats["mean"],
        "spatial_obs_mean": spatial_observation_stats["mean"],
        "spatial_obs_count": len(spatial_observations),
        "point_pred_mean": point_prediction_stats["mean"],
        "point_pred_std": point_prediction_stats["std"],
        "point_obs_mean": point_observation_stats["mean"],
        "point_obs_count": len(point_observations),
        "point_lat": float(latitude[point_y]),
        "point_lon": float(longitude[point_x]),
    }


def evaluate_hours(device, hours, samples_per_level=None, skip_diurnal=False):
    per_hour_metrics = {}
    per_hour_diurnal = {}
    random_generator = np.random.default_rng(SEED)

    for hour in hours:
        data_dir = KNN_HOURLY_ROOT / f"h{hour:02d}"
        data_path = data_dir / f"jaxa_knn_filled_{SERIES_TO_USE:02d}.h5"
        print(f"\nH={hour:02d}: loading final model and {data_path.name}")
        model, norm_mean, norm_std = load_hourly_model(hour, device)
        with h5py.File(data_path, "r") as data_file:
            sst_all = data_file["sst_data"][:]
            obs_all = data_file["original_obs_mask"][:]
            miss_all = data_file["original_missing_mask"][:]
            land = data_file["land_mask"][:]
            latitude = data_file["latitude"][:]
            longitude = data_file["longitude"][:]
            timestamps = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in data_file["timestamps"][:]
            ]

        ocean = 1 - land
        candidates = np.array(
            [
                index
                for index in range(WINDOW_SIZE - 1, len(sst_all))
                if (obs_all[index] * ocean).sum() > 4000
            ]
        )
        cloud_bank = build_cloud_bank(
            [
                str(data_dir / f"jaxa_knn_filled_{series_id:02d}.h5")
                for series_id in DONOR_SERIES
            ],
            max_donors=400,
            seed=7,
        )

        all_mae = []
        all_rmse = []
        by_level = {}
        for level_name, level in MASK_LEVELS:
            if samples_per_level is None:
                selected_indices = candidates
            else:
                selected_indices = random_generator.choice(
                    candidates,
                    size=min(samples_per_level, len(candidates)),
                    replace=False,
                )
            mask_generator = RealCloudMaskGenerator(
                cloud_bank, seed=int(level * 1000)
            )
            level_mae = []
            level_rmse = []
            for target_index in tqdm(
                selected_indices,
                desc=f"    H={hour:02d} {level_name}",
                leave=False,
            ):
                sst_sequence, missing_sequence = make_window(
                    sst_all, miss_all, target_index
                )
                eligible = obs_all[target_index] * ocean
                artificial_mask = mask_generator.generate(
                    eligible.astype(np.float32), target_ratio=level
                )
                evaluation_mask = (artificial_mask * eligible).astype(bool)
                if not evaluation_mask.any():
                    continue
                prediction = predict_extra_mask(
                    model,
                    sst_sequence,
                    missing_sequence,
                    artificial_mask,
                    norm_mean,
                    norm_std,
                    device,
                )
                difference = prediction[evaluation_mask] - sst_sequence[-1][evaluation_mask]
                level_mae.append(float(np.mean(np.abs(difference))))
                level_rmse.append(float(np.sqrt(np.mean(difference ** 2))))

            all_mae.extend(level_mae)
            all_rmse.extend(level_rmse)
            by_level[level_name] = {
                "target_ratio": level,
                "n": len(level_mae),
                "mae_mean": summarize(level_mae)["mean"],
                "mae_std": summarize(level_mae)["std"],
                "rmse_mean": summarize(level_rmse)["mean"],
                "rmse_std": summarize(level_rmse)["std"],
            }

        per_hour_metrics[hour] = {
            "n": len(all_mae),
            "target_fields": len(candidates),
            "mae_mean": summarize(all_mae)["mean"],
            "mae_std": summarize(all_mae)["std"],
            "rmse_mean": summarize(all_rmse)["mean"],
            "rmse_std": summarize(all_rmse)["std"],
            "by_level": by_level,
        }
        print(
            f"H={hour:02d}: target_fields={len(candidates)}, pairs={len(all_mae)}, "
            f"MAE={per_hour_metrics[hour]['mae_mean']:.4f} K, "
            f"RMSE={per_hour_metrics[hour]['rmse_mean']:.4f} K"
        )

        if not skip_diurnal:
            per_hour_diurnal[hour] = compute_diurnal_for_hour(
                hour,
                model,
                sst_all,
                obs_all,
                miss_all,
                land,
                timestamps,
                latitude,
                longitude,
                norm_mean,
                norm_std,
                device,
            )

        del model
        torch.cuda.empty_cache()

    return per_hour_metrics, per_hour_diurnal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--hours",
        default=None,
        help="Comma-separated hour subset; default evaluates all 24 hours.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Optional smoke-test sample count per level; default uses all fields.",
    )
    parser.add_argument("--out-name", default="fig7_stats.npz")
    parser.add_argument("--skip-diurnal", action="store_true")
    args = parser.parse_args()

    hours = (
        list(range(24))
        if args.hours is None
        else [int(hour) for hour in args.hours.split(",")]
    )
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}, hours={hours}, series={SERIES_TO_USE}")
    per_hour, diurnal = evaluate_hours(
        device,
        hours,
        samples_per_level=args.nsamples,
        skip_diurnal=args.skip_diurnal,
    )
    output_path = CACHE_DIR / args.out_name
    np.savez_compressed(
        output_path,
        per_hour=np.array(per_hour, dtype=object),
        diurnal=np.array(diurnal, dtype=object),
    )
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
