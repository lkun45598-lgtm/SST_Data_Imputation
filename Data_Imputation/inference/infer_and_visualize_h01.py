#!/usr/bin/env python3
"""
H=1 微调模型推理 + 可视化（4连图，复刻h=0风格）

输出路径: experiments/jaxa_finetune_h01/inference_vis/sample_XX.png
误差统计: 只在人工挖空的海洋区域计算
"""

import sys
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.jaxa_inference_dataset import JAXAFinetuneDataset

# ============================================================
# 路径配置
# ============================================================
MODEL_PATH    = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h01/best_model.pth'
DATA_DIR      = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01'
COORDS_FILE   = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01/jaxa_knn_filled_08.h5'
OUTPUT_DIR    = Path('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h01/inference_vis')

# 4个测试样本的时间索引（均匀分布于验证集）
N_SAMPLES = 4


# ============================================================
# Matplotlib 全局配置（复刻h=0风格）
# ============================================================
def setup_matplotlib():
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1.5, labelsize=14)
    plt.rc('lines', linewidth=1.5)
    plt.rcParams.update({
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.pad': 5, 'ytick.major.pad': 5,
    })


def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


# ============================================================
# 4连图绘制
# ============================================================
def draw_four_panel(input_sst_norm, artificial_mask, ground_truth_norm,
                    pred_norm, land_mask, lon_coords, lat_coords,
                    norm_mean, norm_std, sample_idx, save_path):
    """
    4连图：Input | Ground Truth | Prediction | Error（只在挖空区）

    Args:
        input_sst_norm:    (H, W) 第30天输入SST，已归一化，挖空处已填0
        artificial_mask:   (H, W) 1=人工挖空, 0=有观测
        ground_truth_norm: (H, W) 完整真值，已归一化
        pred_norm:         (H, W) 模型预测，已归一化
        land_mask:         (H, W) 1=陆地
        lon_coords:        (W,)
        lat_coords:        (H,)
        norm_mean/std:     Kelvin归一化参数
        sample_idx:        样本序号（用于标题）
        save_path:         保存路径
    """
    # ---------- 反归一化 → Kelvin → Celsius ----------
    to_celsius = lambda x: x * norm_std + norm_mean - 273.15

    input_celsius  = to_celsius(input_sst_norm)
    gt_celsius     = to_celsius(ground_truth_norm)
    pred_celsius   = to_celsius(pred_norm)

    abs_error = np.abs(pred_celsius - gt_celsius)

    # ---------- 掩码准备 ----------
    ocean_mask   = (land_mask == 0)
    missing_ocean = (artificial_mask == 1) & ocean_mask

    # ---------- 指标（只在挖空海洋区域） ----------
    mae       = float(abs_error[missing_ocean].mean())
    rmse      = float(np.sqrt((abs_error[missing_ocean]**2).mean()))
    max_err   = float(abs_error[missing_ocean].max())
    miss_rate = float(missing_ocean.sum() / ocean_mask.sum() * 100)

    # ---------- 坐标网格 ----------
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # ---------- SST色阶范围（2~98百分位，GT海洋区） ----------
    vmin_sst = np.percentile(gt_celsius[ocean_mask], 2)
    vmax_sst = np.percentile(gt_celsius[ocean_mask], 98)
    cmap_sst   = 'RdYlBu_r'
    cmap_error = 'hot_r'
    land_color = '#D2B48C'

    # ---------- 布局 ----------
    fig = plt.figure(figsize=(28, 7))
    gs  = gridspec.GridSpec(1, 6, figure=fig,
                            width_ratios=[1, 1, 1, 0.08, 1, 0.08],
                            wspace=0.15, hspace=0.1,
                            left=0.04, right=0.98, top=0.85, bottom=0.15)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_gt    = fig.add_subplot(gs[0, 1])
    ax_pred  = fig.add_subplot(gs[0, 2])
    ax_error = fig.add_subplot(gs[0, 4])

    land_display = np.ma.masked_where(land_mask == 0, land_mask.astype(float))

    # ===== Panel 1: Input SST =====
    ax = ax_input
    ax.set_facecolor('skyblue')                         # 挖空区=云层蓝
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    input_display = input_celsius.copy()
    input_display[artificial_mask == 1] = np.nan        # 挖空处显示背景色
    input_masked = np.ma.masked_where(land_mask > 0, input_display)
    ax.pcolormesh(lon_grid, lat_grid, input_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title(f'Input SST (Day 30)\n{miss_rate:.1f}% Missing',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)

    # ===== Panel 2: Ground Truth =====
    ax = ax_gt
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    gt_masked = np.ma.masked_where(land_mask > 0, gt_celsius)
    ax.pcolormesh(lon_grid, lat_grid, gt_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Ground Truth SST\n(Complete, KNN-filled)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_yticks([])
    ax.set_box_aspect(1)

    # ===== Panel 3: Prediction =====
    ax = ax_pred
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    pred_masked = np.ma.masked_where(land_mask > 0, pred_celsius)
    im2 = ax.pcolormesh(lon_grid, lat_grid, pred_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('FNO-CBAM Reconstruction\n(Model Output)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_yticks([])
    ax.set_box_aspect(1)

    # SST colorbar
    cax2_cont = fig.add_subplot(gs[0, 3])
    cax2_cont.axis('off')
    cax2 = inset_axes(cax2_cont, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1),
                      bbox_transform=cax2_cont.transAxes)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('SST (°C)', fontsize=11)

    # ===== Panel 4: Absolute Error（只显示挖空区） =====
    ax = ax_error
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    error_display = abs_error.copy()
    error_display[artificial_mask == 0] = np.nan        # 非挖空区不显示误差
    error_masked = np.ma.masked_where((land_mask > 0) | (artificial_mask == 0), error_display)
    im3 = ax.pcolormesh(lon_grid, lat_grid, error_masked,
                        cmap=cmap_error, vmin=0, vmax=0.5, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Absolute Error (Missing Region Only)\n|Reconstruction - Ground Truth|',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_yticks([])
    ax.set_box_aspect(1)

    # 统计文字
    stats_text = f'MAE:  {mae:.3f}°C\nRMSE: {rmse:.3f}°C\nMax:  {max_err:.3f}°C'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Error colorbar
    cax3_cont = fig.add_subplot(gs[0, 5])
    cax3_cont.axis('off')
    cax3 = inset_axes(cax3_cont, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1),
                      bbox_transform=cax3_cont.transAxes)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('|Error| (°C)', fontsize=11)

    # 顶部日期标签
    fig.text(0.5, 0.98, f'H=1 Inference  |  Sample {sample_idx+1:02d}',
             ha='center', va='top', fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return mae, rmse, max_err


# ============================================================
# 主流程
# ============================================================
def main():
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0')
    print("=" * 70)
    print("H=1 模型推理 + 4连图可视化")
    print("=" * 70)

    # ---------- 加载模型 ----------
    print(f"\n加载模型: {MODEL_PATH}")
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    norm_mean = float(ckpt['norm_mean'])
    norm_std  = float(ckpt['norm_std'])
    model.eval()
    print(f"  ✓ Epoch {ckpt['epoch']}, val_mae={ckpt['val_mae']:.3f}K")
    print(f"  ✓ 归一化: mean={norm_mean:.2f}K, std={norm_std:.2f}K")

    # ---------- 加载经纬度坐标 ----------
    with h5py.File(COORDS_FILE, 'r') as f:
        lat_coords = f['latitude'][:]
        lon_coords = f['longitude'][:]
    print(f"\n坐标范围: lon {lon_coords.min():.1f}°~{lon_coords.max():.1f}°E, "
          f"lat {lat_coords.min():.1f}°~{lat_coords.max():.1f}°N")

    # ---------- 加载测试数据集 ----------
    print("\n加载测试数据集 (series 8)...")
    test_dataset = JAXAFinetuneDataset(
        data_dir=DATA_DIR,
        series_ids=[8],
        window_size=30,
        mask_ratio=0.2,        # 与训练时保持一致
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=norm_mean,
        std=norm_std,
        cache_size=50,
        seed=42
    )
    print(f"  ✓ {len(test_dataset)} 个样本")

    # 均匀选取4个样本
    step = len(test_dataset) // N_SAMPLES
    sample_indices = [i * step for i in range(N_SAMPLES)]

    # ---------- 推理 + 绘图 ----------
    print(f"\n开始推理 ({N_SAMPLES} 个样本)...")
    all_mae, all_rmse = [], []

    with torch.no_grad():
        for idx, s_idx in enumerate(sample_indices):
            batch = test_dataset[s_idx]

            sst_seq  = torch.from_numpy(batch['input_sst_seq']).unsqueeze(0).to(device).float()
            mask_seq = torch.from_numpy(batch['mask_seq']).unsqueeze(0).to(device).float()
            land_mask_t = torch.from_numpy(batch['land_mask']).to(device).float()

            pred = model(sst_seq, mask_seq)   # (1, 1, H, W)

            # 取第30天的输入/掩码
            input_sst_norm  = sst_seq[0, -1].cpu().numpy()          # (H, W)
            artificial_mask = mask_seq[0, -1].cpu().numpy()         # (H, W) 1=挖空
            gt_norm         = batch['ground_truth_sst']              # (H, W) numpy
            pred_norm       = pred[0, 0].cpu().numpy()               # (H, W)
            land_mask_np    = land_mask_t.cpu().numpy()              # (H, W)

            save_path = OUTPUT_DIR / f'sample_{idx:02d}.png'
            mae, rmse, max_err = draw_four_panel(
                input_sst_norm, artificial_mask, gt_norm, pred_norm,
                land_mask_np, lon_coords, lat_coords,
                norm_mean, norm_std, idx, save_path
            )

            all_mae.append(mae)
            all_rmse.append(rmse)
            print(f"  Sample {idx+1}: MAE={mae:.3f}°C, RMSE={rmse:.3f}°C, Max={max_err:.3f}°C → {save_path.name}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print(f"平均 MAE : {np.mean(all_mae):.3f}°C")
    print(f"平均 RMSE: {np.mean(all_rmse):.3f}°C")
    print(f"图片保存目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
