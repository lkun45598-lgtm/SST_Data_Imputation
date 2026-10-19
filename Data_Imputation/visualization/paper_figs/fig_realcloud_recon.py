#!/usr/bin/env python3
"""真实JAXA云区(无真值)重建可视化 —— 所有方法并排对比。

与 eval_all_methods.py 的区别：那里是在"有真值的观测区"人工挖空来算指标；
这里是直接填 JAXA 真实缺失的云区(original_missing_mask==1)，没有真值，
纯看各方法在"已知区少、大范围外推"的真实场景下恢复出什么。

每个案例一行：
  [原始观测(带云洞)] [FNO] [DINCAE] [UNet] [ConvLSTM] [OI] [KNN]
云区用真实观测边界勾勒，方便看谁的填充更合理/更平滑/更多细节。
"""
import sys, argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DI))
sys.path.insert(0, str(DI / "baselines"))
sys.path.insert(0, str(DI / "visualization/paper_figs"))

from baselines.eval_all_methods import (
    load_fno, load_deep, run_torch_model, FNO_MEAN, FNO_STD, W, GAUSS_SIGMA)
from baselines.optimal_interpolation import oi_reconstruct
from gen_eval_data import knn_inpaint_2d, gauss_filter
from style import CMAPS, LAND_COLOR

import torch

# 与其他论文图一致的 SST 配色 (RdYlBu_r) + 沙棕陆地
SST_CMAP = CMAPS["sst"]

H12 = DI / "experiments/hourly_data/h12"
OUT = DI / "visualization/paper_figs/output"
OUT.mkdir(parents=True, exist_ok=True)
SERIES_ID = 8


def build_input(sst_all, miss_all, idx):
    """构造30天输入序列 (最后一帧的云区就是要填的目标)。"""
    sst_seq = np.stack([sst_all[max(0, idx - (W - 1) + t)] for t in range(W)])
    miss_seq = np.stack([miss_all[max(0, idx - (W - 1) + t)] for t in range(W)])
    return sst_seq, miss_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frames", type=str, default="40,57,58",
                    help="逗号分隔的帧号(真实大云区帧)")
    ap.add_argument("--out-name", type=str, default="fig_realcloud_recon.png")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    frames = [int(x) for x in args.frames.split(",")]

    # 加载数据
    path = H12 / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:].astype(np.float32)
        obs_all = f["original_obs_mask"][:].astype(np.uint8)
        miss_all = f["original_missing_mask"][:].astype(np.uint8)
        land = f["land_mask"][:].astype(np.uint8)
    ocean = (land == 0)

    # 加载模型
    print("loading models...")
    fno = load_fno(device)
    dincae = load_deep("dincae", device)
    unet = load_deep("unet", device)
    convlstm = load_deep("convlstm", device)

    methods = ["obs", "fno", "dincae", "unet", "convlstm", "oi", "knn"]
    titles = ["JAXA obs (cloud gaps)", "FNO (Ours)", "DINCAE", "U-Net",
              "ConvLSTM", "OI/Kriging", "KNN/IDW"]

    fig, axes = plt.subplots(len(frames), len(methods),
                             figsize=(3.0 * len(methods), 3.0 * len(frames)))
    if len(frames) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(frames):
        print(f"\n=== frame {idx} ===")
        sst_seq, miss_seq = build_input(sst_all, miss_all, idx)
        # 真实云区 = 该帧 ocean 上的缺失(无真值)
        cloud = ((miss_all[idx] == 1) & ocean).astype(np.float32)
        obs_field = np.where(obs_all[idx] == 1, sst_all[idx], np.nan)
        obs_ratio = (obs_all[idx] * ocean).sum() / ocean.sum()
        print(f"  观测覆盖率 {obs_ratio*100:.1f}%, 云区像素 {int(cloud.sum())}")

        # 统一色标范围(用观测区的真实SST范围)
        valid_obs = sst_all[idx][(obs_all[idx] == 1) & ocean]
        vmin, vmax = np.percentile(valid_obs, [2, 98])

        # 各方法填充云区
        preds = {}
        # 深度学习模型: mask=cloud(要填的真实云区)
        for nm, model in [("fno", fno), ("dincae", dincae),
                          ("unet", unet), ("convlstm", convlstm)]:
            mean, std = (FNO_MEAN, FNO_STD)
            out = run_torch_model(model, sst_seq, miss_seq, cloud, mean, std, device)
            # 只在云区写模型输出,观测区保留真实值
            full = np.where(cloud > 0, out, sst_all[idx])
            full = gauss_filter(full, land, GAUSS_SIGMA, fill_region=cloud)
            preds[nm] = np.where(ocean, full, np.nan)

        # OI
        oi_out, _ = oi_reconstruct(sst_all[idx], cloud, obs_all[idx].astype(np.uint8))
        preds["oi"] = np.where(ocean, oi_out, np.nan)
        # KNN
        ki = sst_all[idx].copy(); ki[cloud > 0] = np.nan
        knn_out = knn_inpaint_2d(ki, cloud, obs_all[idx].astype(np.uint8), k=20)
        preds["knn"] = np.where(ocean, knn_out, np.nan)

        # 画图
        for col, m in enumerate(methods):
            ax = axes[row, col]
            if m == "obs":
                img = np.where((obs_all[idx] == 1) & ocean, sst_all[idx], np.nan)
                img = np.where(ocean, img, np.nan)
            else:
                img = preds[m]
            # 陆地铺沙棕底色 (与 fig1/其他论文图一致)
            ax.imshow(np.where(~ocean, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap([LAND_COLOR]),
                      vmin=0, vmax=1, origin="lower")
            im = ax.imshow(img, cmap=SST_CMAP, vmin=vmin, vmax=vmax, origin="lower")
            # 勾勒云区边界
            ax.contour(cloud, levels=[0.5], colors="white", linewidths=0.4, alpha=0.6)
            if row == 0:
                ax.set_title(titles[col], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"frame {idx}\nobs {obs_ratio*100:.0f}%", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Reconstruction of REAL JAXA cloud gaps (no ground truth) — "
                 "white contour = cloud boundary",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out_path = OUT / args.out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
