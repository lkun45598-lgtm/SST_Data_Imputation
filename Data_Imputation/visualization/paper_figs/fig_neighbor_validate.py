#!/usr/bin/env python3
"""用"相邻天真实观测"作伪真值，验证各方法在真实云区填出的结构对不对。

云是移动的：第 D 天被云遮住的像素，往往在 D±1/±2/±3 天是晴空、有真实观测。
拿最近的那一天的真实观测当"伪真值"(pseudo-truth)，就能在 *真实云区* 上
量化每个方法填得准不准、结构对不对 —— 这是无真值场景下唯一诚实的检验。

注意/局限：
  - SST 有日变化 + 中尺度平流，邻天不是精确真值，大尺度可信、细结构会漂移。
  - 只在"当天是云 & 邻天有观测"的重叠像素上评估 (eval region)。

每帧一行：[当天观测] [伪真值(邻天)] [FNO] [DINCAE] [UNet] [ConvLSTM] [OI] [KNN]
标题标注各方法在 eval region 的 MAE(相对伪真值)。
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

SST_CMAP = CMAPS["sst"]
H12 = DI / "experiments/hourly_data/h12"
OUT = DI / "visualization/paper_figs/output"
OUT.mkdir(parents=True, exist_ok=True)
SERIES_ID = 8
NEIGHBOR_OFFSETS = [1, -1, 2, -2, 3, -3]  # 按距离由近到远取邻天


def build_pseudo_truth(sst_all, obs_all, cloud, D):
    """在 cloud 像素上，取最近邻天(有观测)的真实值作伪真值。
    返回 (pt_field, eval_mask): eval_mask=在 cloud 且某邻天有观测的像素。"""
    pt = np.full(cloud.shape, np.nan, np.float32)
    filled = np.zeros(cloud.shape, bool)
    for off in NEIGHBOR_OFFSETS:
        j = D + off
        if not (0 <= j < sst_all.shape[0]):
            continue
        take = cloud & (~filled) & (obs_all[j] == 1)
        pt[take] = sst_all[j][take]
        filled |= take
    return pt, (filled & cloud)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frames", type=str, default="82,48,68")
    ap.add_argument("--out-name", type=str, default="fig_neighbor_validate.png")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    frames = [int(x) for x in args.frames.split(",")]

    path = H12 / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:].astype(np.float32)
        obs_all = f["original_obs_mask"][:].astype(np.uint8)
        miss_all = f["original_missing_mask"][:].astype(np.uint8)
        land = f["land_mask"][:].astype(np.uint8)
    ocean = (land == 0)

    print("loading models...")
    models = {"fno": load_fno(device), "dincae": load_deep("dincae", device),
              "unet": load_deep("unet", device), "convlstm": load_deep("convlstm", device)}

    cols = ["obs", "truth", "fno", "dincae", "unet", "convlstm", "oi", "knn"]
    titles = ["JAXA obs (day D)", "pseudo-truth\n(neighbor day)", "FNO (Ours)",
              "DINCAE", "U-Net", "ConvLSTM", "OI/Kriging", "KNN/IDW"]

    fig, axes = plt.subplots(len(frames), len(cols),
                             figsize=(2.7 * len(cols), 3.0 * len(frames)))
    if len(frames) == 1:
        axes = axes[None, :]

    for row, D in enumerate(frames):
        print(f"\n=== frame {D} ===")
        cloud = ((miss_all[D] == 1) & ocean)
        pt, ev = build_pseudo_truth(sst_all, obs_all, cloud, D)
        print(f"  obs {(obs_all[D]*ocean).sum()/ocean.sum()*100:.1f}%, "
              f"cloud {int(cloud.sum())}, eval(邻天可验证) {int(ev.sum())} "
              f"({ev.sum()/max(cloud.sum(),1)*100:.0f}%)")

        sst_seq = np.stack([sst_all[max(0, D - (W - 1) + t)] for t in range(W)])
        miss_seq = np.stack([miss_all[max(0, D - (W - 1) + t)] for t in range(W)])
        cloudf = cloud.astype(np.float32)

        valid_obs = sst_all[D][(obs_all[D] == 1) & ocean]
        vmin, vmax = np.percentile(valid_obs, [2, 98])

        preds, maes = {}, {}
        for nm, model in models.items():
            out = run_torch_model(model, sst_seq, miss_seq, cloudf, FNO_MEAN, FNO_STD, device)
            full = np.where(cloud, out, sst_all[D])
            full = gauss_filter(full, land, GAUSS_SIGMA, fill_region=cloudf)
            preds[nm] = np.where(ocean, full, np.nan)
        oi_out, _ = oi_reconstruct(sst_all[D], cloudf, obs_all[D].astype(np.uint8))
        preds["oi"] = np.where(ocean, oi_out, np.nan)
        ki = sst_all[D].copy(); ki[cloud] = np.nan
        preds["knn"] = np.where(ocean, knn_inpaint_2d(ki, cloudf, obs_all[D].astype(np.uint8), k=20), np.nan)

        for nm in preds:
            e = preds[nm][ev] - pt[ev]
            maes[nm] = float(np.abs(e[np.isfinite(e)]).mean())
        order = sorted(maes, key=maes.get)
        print("  MAE vs pseudo-truth: " + "  ".join(f"{k}={maes[k]:.3f}" for k in order))

        for col, m in enumerate(cols):
            ax = axes[row, col]
            ax.imshow(np.where(~ocean, 1.0, np.nan),
                      cmap=matplotlib.colors.ListedColormap([LAND_COLOR]),
                      vmin=0, vmax=1, origin="lower")
            if m == "obs":
                img = np.where((obs_all[D] == 1) & ocean, sst_all[D], np.nan)
            elif m == "truth":
                img = np.where(ev, pt, np.nan)  # 只画可验证的邻天真值
            else:
                img = preds[m]
            ax.imshow(img, cmap=SST_CMAP, vmin=vmin, vmax=vmax, origin="lower")
            ax.contour(cloud.astype(float), levels=[0.5], colors="white", linewidths=0.4, alpha=0.6)
            t = titles[col]
            if m in maes:
                best = " ★" if m == order[0] else ""
                t = f"{titles[col]}\nMAE={maes[m]:.3f}{best}"
            if row == 0 or m in maes:
                ax.set_title(t, fontsize=9.5, fontweight="bold" if m == "fno" else "normal")
            if col == 0:
                ax.set_ylabel(f"frame {D}\nobs {(obs_all[D]*ocean).sum()/ocean.sum()*100:.0f}%  "
                              f"val {ev.sum()/max(cloud.sum(),1)*100:.0f}%", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Structure check on REAL cloud gaps via neighbor-day observations "
                 "(pseudo-truth) — MAE(°C) vs pseudo-truth, ★=best",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    out_path = OUT / args.out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
