#!/usr/bin/env python3
"""
评估最新回传的 h12 完整模型 (val_mae=0.316), 与论文主结果 0.328/0.337/0.360 对比
协议与 gen_fig6_stats.py 完全一致: 真实云掩码 30/55/75%, series8, N=20, Gaussian平滑sigma=1
"""
import sys
from pathlib import Path
import numpy as np
import torch, h5py
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

DI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DI))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator

DATA_DIR = DI / "experiments/hourly_data/h12"
MODEL_PATH = DI / "experiments/jaxa_finetune_h12_realmask_clean/best_model.pth"
GPU_ID = 0; SERIES_ID = 8; W = 30; SIGMA = 1.0; SEED = 42; N = 20
NORM_MEAN, NORM_STD = 299.9221, 2.6919
LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]


def gauss_filter(sst, land, sigma, fill_region):
    valid = ~np.isnan(sst) & (land == 0)
    if valid.sum() == 0: return sst
    fill = sst.copy(); fill[~valid] = np.nanmean(sst)
    out = gaussian_filter(fill, sigma=sigma)
    write = valid & (fill_region == 1)
    return np.where(write, out, np.where(valid, sst, np.nan))


def predict(model, sst_seq, miss_seq, mask, device):
    si = sst_seq.copy(); si[-1] = np.where(mask > 0, NORM_MEAN, si[-1])
    ms = miss_seq.copy().astype(np.float32); ms[-1] = mask
    sn = np.nan_to_num((si - NORM_MEAN) / NORM_STD, nan=0.0)
    st = torch.from_numpy(sn).unsqueeze(0).float().to(device)
    mt = torch.from_numpy(ms).unsqueeze(0).to(device)
    with torch.no_grad():
        pr = model(st, mt).squeeze().cpu().numpy() * NORM_STD + NORM_MEAN
    return np.where(mask > 0, pr, sst_seq[-1])


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64,
                                  width=64, depth=6, cbam_reduction_ratio=16, use_cbam=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    print(f"h12 模型: val_mae={ckpt.get('val_mae'):.4f}, epoch={ckpt.get('epoch')}")

    with h5py.File(DATA_DIR / f"jaxa_knn_filled_{SERIES_ID:02d}.h5", "r") as f:
        sst_all = f["sst_data"][:]; obs_all = f["original_obs_mask"][:]
        miss_all = f["original_missing_mask"][:]; land = f["land_mask"][:]
    T = sst_all.shape[0]; ocean = 1 - land

    bank = build_cloud_bank([str(DATA_DIR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]],
                            max_donors=400, seed=7)
    rng = np.random.default_rng(SEED)
    cand = np.array([i for i in range(W - 1, T) if obs_all[i].sum() > 5000])

    results = {}
    for name, level in LEVELS:
        idxs = rng.choice(cand, size=min(N, len(cand)), replace=False)
        gen = RealCloudMaskGenerator(bank, seed=int(level * 1000))
        maes, rmses = [], []
        for idx in tqdm(idxs, desc=f"{name}", leave=False):
            sst_seq = np.stack([sst_all[max(0, idx - (W - 1) + t)] for t in range(W)]).astype(np.float32)
            miss_seq = np.stack([miss_all[max(0, idx - (W - 1) + t)] for t in range(W)]).astype(np.float32)
            elig = obs_all[idx] * ocean
            if elig.sum() < 2000: continue
            mask = gen.generate(elig.astype(np.float32), target_ratio=level)
            gt = sst_seq[-1].copy()
            em = (mask * elig).astype(bool)
            if em.sum() < 200: continue
            pred = predict(model, sst_seq, miss_seq, mask, device)
            pred = gauss_filter(pred, land, SIGMA, fill_region=(mask > 0))
            d = pred[em] - gt[em]
            maes.append(float(np.abs(d).mean())); rmses.append(float(np.sqrt((d ** 2).mean())))
        results[name] = (np.mean(maes), np.mean(rmses), len(maes))

    print("\n" + "=" * 60)
    print(f"{'Level':<8}{'新h12 MAE':>12}{'论文 MAE':>12}{'RMSE':>10}{'N':>5}")
    print("-" * 60)
    paper = {"low": 0.328, "mid": 0.337, "high": 0.360}
    for name, _ in LEVELS:
        mae, rmse, n = results[name]
        print(f"{name:<8}{mae:>12.3f}{paper[name]:>12.3f}{rmse:>10.3f}{n:>5}")
    avg = np.mean([results[n][0] for n, _ in LEVELS])
    print("-" * 60)
    print(f"{'平均':<8}{avg:>12.3f}{np.mean(list(paper.values())):>12.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
