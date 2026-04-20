#!/usr/bin/env python3
"""
批量推理 + 可视化：h01 ~ h22 所有已训练的逐小时模型

对每个小时：
  - 加载 experiments/jaxa_finetune_h{hh}/best_model.pth
  - 从 experiments/hourly_data/h{hh}/ 取 series 8 的4个均匀样本
  - 生成4连图保存到 experiments/jaxa_finetune_h{hh}/inference_vis/
  - 汇总各小时 MAE / RMSE

用法:
  python batch_infer_hourly.py [--hours 1-22] [--gpu 6] [--n_samples 4]
"""

import sys
import os
import argparse
import torch
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── 项目路径 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2] / 'Data_Imputation'
sys.path.insert(0, str(PROJECT_ROOT))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.jaxa_inference_dataset import JAXAFinetuneDataset

EXP_ROOT = PROJECT_ROOT / 'experiments'

# ── Matplotlib 全局设置 ───────────────────────────────────────────────────────
def setup_matplotlib():
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1.5, labelsize=14)
    plt.rcParams.update({
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

def fmt_lon(x, pos): return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"
def fmt_lat(y, pos): return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


# ── 4连图 ──────────────────────────────────────────────────────────────────────
def draw_four_panel(input_sst_norm, artificial_mask, gt_norm, pred_norm,
                    land_mask, lon_coords, lat_coords,
                    norm_mean, norm_std, hour, sample_idx, save_path):
    to_c = lambda x: x * norm_std + norm_mean - 273.15

    input_c = to_c(input_sst_norm)
    gt_c    = to_c(gt_norm)
    pred_c  = to_c(pred_norm)
    abs_err = np.abs(pred_c - gt_c)

    ocean      = (land_mask == 0)
    miss_ocean = (artificial_mask == 1) & ocean

    mae      = float(abs_err[miss_ocean].mean())
    rmse     = float(np.sqrt((abs_err[miss_ocean]**2).mean()))
    max_err  = float(abs_err[miss_ocean].max())
    miss_pct = float(miss_ocean.sum() / ocean.sum() * 100)

    lon_g, lat_g = np.meshgrid(lon_coords, lat_coords)

    vmin = np.percentile(gt_c[ocean], 2)
    vmax = np.percentile(gt_c[ocean], 98)
    cmap_sst   = 'RdYlBu_r'
    cmap_err   = 'hot_r'
    land_color = '#D2B48C'

    fig = plt.figure(figsize=(28, 7))
    gs  = gridspec.GridSpec(1, 6, figure=fig,
                            width_ratios=[1, 1, 1, 0.08, 1, 0.08],
                            wspace=0.15, left=0.04, right=0.98,
                            top=0.85, bottom=0.15)

    ax_in  = fig.add_subplot(gs[0, 0])
    ax_gt  = fig.add_subplot(gs[0, 1])
    ax_pr  = fig.add_subplot(gs[0, 2])
    ax_er  = fig.add_subplot(gs[0, 4])

    land_disp = np.ma.masked_where(land_mask == 0, land_mask.astype(float))

    def _base(ax, show_yticks=True):
        ax.set_facecolor('skyblue')
        ax.pcolormesh(lon_g, lat_g, land_disp,
                      cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.locator_params(axis='x', nbins=4)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_box_aspect(1)
        if show_yticks:
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
            ax.locator_params(axis='y', nbins=5)
            ax.set_ylabel('Latitude', fontsize=12)
        else:
            ax.set_yticks([])

    # Panel 1: Input
    _base(ax_in, show_yticks=True)
    inp = input_c.copy(); inp[artificial_mask == 1] = np.nan
    ax_in.pcolormesh(lon_g, lat_g, np.ma.masked_where(land_mask > 0, inp),
                     cmap=cmap_sst, vmin=vmin, vmax=vmax, shading='auto')
    ax_in.set_title(f'Input SST (Day 30)\n{miss_pct:.1f}% Missing',
                    fontsize=13, fontweight='bold', pad=10)

    # Panel 2: Ground Truth
    _base(ax_gt, show_yticks=False)
    ax_gt.set_facecolor('lightgray')
    ax_gt.pcolormesh(lon_g, lat_g, np.ma.masked_where(land_mask > 0, gt_c),
                     cmap=cmap_sst, vmin=vmin, vmax=vmax, shading='auto')
    ax_gt.set_title('Ground Truth SST\n(KNN-filled)', fontsize=13, fontweight='bold', pad=10)

    # Panel 3: Prediction
    _base(ax_pr, show_yticks=False)
    ax_pr.set_facecolor('lightgray')
    im2 = ax_pr.pcolormesh(lon_g, lat_g, np.ma.masked_where(land_mask > 0, pred_c),
                           cmap=cmap_sst, vmin=vmin, vmax=vmax, shading='auto')
    ax_pr.set_title('FNO-CBAM Reconstruction', fontsize=13, fontweight='bold', pad=10)

    cax2c = fig.add_subplot(gs[0, 3]); cax2c.axis('off')
    cax2  = inset_axes(cax2c, width="50%", height="90%", loc='center',
                       bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax2c.transAxes)
    plt.colorbar(im2, cax=cax2).set_label('SST (°C)', fontsize=11)

    # Panel 4: Error
    _base(ax_er, show_yticks=False)
    ax_er.set_facecolor('white')
    err_disp = abs_err.copy(); err_disp[artificial_mask == 0] = np.nan
    im3 = ax_er.pcolormesh(lon_g, lat_g,
                            np.ma.masked_where((land_mask > 0) | (artificial_mask == 0), err_disp),
                            cmap=cmap_err, vmin=0, vmax=0.5, shading='auto')
    ax_er.set_title('|Error| (Missing Region Only)', fontsize=13, fontweight='bold', pad=10)
    ax_er.text(0.02, 0.98,
               f'MAE:  {mae:.3f}°C\nRMSE: {rmse:.3f}°C\nMax:  {max_err:.3f}°C',
               transform=ax_er.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    cax3c = fig.add_subplot(gs[0, 5]); cax3c.axis('off')
    cax3  = inset_axes(cax3c, width="50%", height="90%", loc='center',
                       bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax3c.transAxes)
    plt.colorbar(im3, cax=cax3).set_label('|Error| (°C)', fontsize=11)

    fig.text(0.5, 0.98, f'H={hour:02d}  |  Sample {sample_idx+1:02d}',
             ha='center', va='top', fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return mae, rmse, max_err


# ── 单个小时推理 ───────────────────────────────────────────────────────────────
def run_hour(hour: int, device: torch.device, n_samples: int = 4):
    hh = f'{hour:02d}'
    model_path = EXP_ROOT / f'jaxa_finetune_h{hh}' / 'best_model.pth'
    data_dir   = EXP_ROOT / 'hourly_data' / f'h{hh}'
    coords_f   = data_dir / 'jaxa_knn_filled_08.h5'
    out_dir    = EXP_ROOT / f'jaxa_finetune_h{hh}' / 'inference_vis'

    if not model_path.exists():
        print(f'  [H={hh}] 模型不存在，跳过')
        return None
    if not coords_f.exists():
        print(f'  [H={hh}] 验证数据不存在，跳过')
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    norm_mean = float(ckpt['norm_mean'])
    norm_std  = float(ckpt['norm_std'])
    model.eval()

    # 加载坐标
    with h5py.File(coords_f, 'r') as f:
        lat_coords = f['latitude'][:]
        lon_coords = f['longitude'][:]

    # 加载数据集（series 8 = 验证集）
    dataset = JAXAFinetuneDataset(
        data_dir=str(data_dir),
        series_ids=[8],
        window_size=30,
        mask_ratio=0.2,
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=norm_mean,
        std=norm_std,
        cache_size=50,
        seed=42,
    )

    if len(dataset) == 0:
        print(f'  [H={hh}] 数据集为空，跳过')
        return None

    step = max(1, len(dataset) // n_samples)
    indices = [i * step for i in range(n_samples)]

    maes, rmses = [], []
    with torch.no_grad():
        for idx, s_idx in enumerate(indices):
            batch = dataset[s_idx]

            sst_seq  = torch.from_numpy(batch['input_sst_seq']).unsqueeze(0).to(device).float()
            mask_seq = torch.from_numpy(batch['mask_seq']).unsqueeze(0).to(device).float()

            pred = model(sst_seq, mask_seq)   # (1,1,H,W)

            input_sst_norm  = sst_seq[0, -1].cpu().numpy()
            artificial_mask = mask_seq[0, -1].cpu().numpy()
            gt_norm         = batch['ground_truth_sst']
            pred_norm       = pred[0, 0].cpu().numpy()
            land_mask_np    = batch['land_mask']

            save_path = out_dir / f'sample_{idx:02d}.png'
            mae, rmse, max_err = draw_four_panel(
                input_sst_norm, artificial_mask, gt_norm, pred_norm,
                land_mask_np, lon_coords, lat_coords,
                norm_mean, norm_std, hour, idx, save_path
            )
            maes.append(mae)
            rmses.append(rmse)

    avg_mae  = float(np.mean(maes))
    avg_rmse = float(np.mean(rmses))
    print(f'  [H={hh}] MAE={avg_mae:.3f}°C  RMSE={avg_rmse:.3f}°C  → {out_dir}')
    return {'hour': hour, 'mae': avg_mae, 'rmse': avg_rmse}


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', default='1-22',
                        help='要推理的小时范围，如 1-22 或 1,3,5')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=4,
                        help='每个小时可视化的样本数')
    args = parser.parse_args()

    # 解析 --hours
    if '-' in args.hours:
        lo, hi = args.hours.split('-')
        hours = list(range(int(lo), int(hi) + 1))
    else:
        hours = [int(x) for x in args.hours.split(',')]

    setup_matplotlib()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print(f'批量逐小时推理  |  GPU: {device}  |  hours: {hours}')
    print('=' * 65)

    results = []
    for h in hours:
        print(f'\n--- H={h:02d} ---')
        r = run_hour(h, device, n_samples=args.n_samples)
        if r:
            results.append(r)

    # 汇总表
    print('\n' + '=' * 65)
    print(f'{"小时":>6}  {"MAE (°C)":>10}  {"RMSE (°C)":>10}')
    print('-' * 35)
    for r in results:
        print(f'  H={r["hour"]:02d}    {r["mae"]:>10.3f}  {r["rmse"]:>10.3f}')
    if results:
        maes  = [r['mae']  for r in results]
        rmses = [r['rmse'] for r in results]
        print('-' * 35)
        print(f'  平均    {np.mean(maes):>10.3f}  {np.mean(rmses):>10.3f}')
    print('=' * 65)


if __name__ == '__main__':
    main()
