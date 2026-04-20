#!/usr/bin/env python3
"""
H=1 推理 - 使用原始缺失模式，复刻h=0 fill_jaxa.py的可视化风格
4连横图：Filtered JAXA SST | KNN Filled SST | FNO-CBAM Filled SST | Model-KNN Difference

输出路径: experiments/jaxa_finetune_h01/inference_original_missing_vis/
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
MODEL_PATH = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h01/best_model.pth'
DATA_DIR = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01'
OUTPUT_DIR = Path('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h01/inference_original_missing_vis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def load_model(model_path, device):
    """加载模型"""
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80, modes2=64,
        width=64, depth=6,
        cbam_reduction_ratio=16
    )
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    norm_mean = checkpoint.get('norm_mean', None)
    norm_std = checkpoint.get('norm_std', None)
    return model, norm_mean, norm_std


def inference_sequence(model, sst_seq, mask_seq, device):
    """模型推理"""
    sst_input = torch.from_numpy(sst_seq[np.newaxis, :, :, :]).float().to(device)
    mask_input = torch.from_numpy(mask_seq[np.newaxis, :, :, :]).float().to(device)
    with torch.no_grad():
        pred = model(sst_input, mask_input)
    return pred[0, 0].cpu().numpy()


# ============================================================
# 4连横图（复刻 fill_jaxa.py 风格）
# ============================================================
def create_four_panel_plot(original_sst_celsius, knn_filled_celsius, model_filled_celsius,
                           original_missing_mask, land_mask,
                           lon_coords, lat_coords, timestamp_str, save_path):
    """
    复刻h=0 fill_jaxa.py的4连横图风格

    1. Filtered JAXA SST (with missing as skyblue)
    2. KNN Filled SST (Coarse Fill)
    3. FNO-CBAM Filled SST (Model Output)
    4. Model - KNN Difference (Missing Regions Only)
    """
    setup_matplotlib()

    fig = plt.figure(figsize=(28, 7))
    gs = gridspec.GridSpec(1, 6, figure=fig,
                          width_ratios=[1, 1, 1, 0.08, 1, 0.08],
                          wspace=0.15, hspace=0.1,
                          left=0.04, right=0.98, top=0.85, bottom=0.15)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_knn = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    ax_diff = fig.add_subplot(gs[0, 4])

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # 颜色设置
    cmap_sst = 'RdYlBu_r'
    cmap_diff = 'RdBu_r'
    land_color = '#D2B48C'

    # 数据范围：收集所有3张图的海洋像素，取全局min/max
    ocean_mask = land_mask == 0
    all_ocean = np.concatenate([
        original_sst_celsius[ocean_mask & (original_missing_mask == 0)],
        knn_filled_celsius[ocean_mask],
        model_filled_celsius[ocean_mask],
    ])
    all_ocean = all_ocean[~np.isnan(all_ocean)]
    if len(all_ocean) > 0:
        vmin_sst = np.floor(np.percentile(all_ocean, 0.5))
        vmax_sst = np.ceil(np.percentile(all_ocean, 99.5))
        # 保证至少5°C范围
        if vmax_sst - vmin_sst < 5:
            mid = (vmin_sst + vmax_sst) / 2
            vmin_sst = mid - 3
            vmax_sst = mid + 3
    else:
        vmin_sst, vmax_sst = 20, 32

    # 陆地显示
    land_display = np.ma.masked_where(land_mask == 0, land_mask)

    # 缺失率
    ocean_pixels = np.sum(ocean_mask)
    missing_pixels = np.sum((original_missing_mask > 0) & ocean_mask)
    missing_rate = missing_pixels / ocean_pixels * 100 if ocean_pixels > 0 else 0

    # ===== 1. Filtered JAXA SST (缺失区域为skyblue) =====
    ax = ax_orig
    ax.set_facecolor('skyblue')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    orig_display = original_sst_celsius.copy()
    orig_display[original_missing_mask > 0] = np.nan
    orig_masked = np.ma.masked_where((land_mask > 0) | np.isnan(orig_display), orig_display)

    ax.pcolormesh(lon_grid, lat_grid, orig_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title(f'Filtered JAXA SST\n(Missing: {missing_rate:.1f}%)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_box_aspect(1)

    # ===== 2. KNN Filled SST =====
    ax = ax_knn
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    knn_masked = np.ma.masked_where(land_mask > 0, knn_filled_celsius)
    ax.pcolormesh(lon_grid, lat_grid, knn_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('KNN Filled SST\n(Coarse Fill)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 3. FNO-CBAM Filled SST =====
    ax = ax_model
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    model_masked = np.ma.masked_where(land_mask > 0, model_filled_celsius)
    im3 = ax.pcolormesh(lon_grid, lat_grid, model_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('FNO-CBAM Filled SST\n(Hybrid Input)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # SST Colorbar
    cax3_container = fig.add_subplot(gs[0, 3])
    cax3_container.axis('off')
    cax3 = inset_axes(cax3_container, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax3_container.transAxes)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('SST (°C)', fontsize=11)

    # ===== 4. Model - KNN Difference (Missing Regions Only) =====
    ax = ax_diff
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    diff = model_filled_celsius - knn_filled_celsius
    diff_display = diff.copy()
    diff_display[(original_missing_mask == 0) | (land_mask > 0)] = np.nan
    diff_masked = np.ma.masked_where(np.isnan(diff_display), diff_display)

    vmax_diff = 2.0
    im4 = ax.pcolormesh(lon_grid, lat_grid, diff_masked,
                        cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Model - KNN Difference\n(Missing Regions Only)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # 差异统计
    if np.sum(~np.isnan(diff_display)) > 0:
        mean_diff = np.nanmean(diff_display)
        std_diff = np.nanstd(diff_display)
        ax.text(0.02, 0.98, f'Mean: {mean_diff:.3f}°C\nStd: {std_diff:.3f}°C',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Diff Colorbar
    cax4_container = fig.add_subplot(gs[0, 5])
    cax4_container.axis('off')
    cax4 = inset_axes(cax4_container, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax4_container.transAxes)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.set_label('Diff (°C)', fontsize=11)

    # 标题
    fig.text(0.5, 0.98, f'Date: {timestamp_str}',
            ha='center', va='top', fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 打印统计
    if np.sum(~np.isnan(diff_display)) > 0:
        print(f"  Saved: {save_path}")
        print(f"  Missing: {missing_rate:.1f}%, Diff Mean: {mean_diff:.3f}°C, Std: {std_diff:.3f}°C")

    return mean_diff if np.sum(~np.isnan(diff_display)) > 0 else 0.0, \
           std_diff if np.sum(~np.isnan(diff_display)) > 0 else 0.0


# ============================================================
# 主流程
# ============================================================
def main():
    print("="*70)
    print("H=1 推理 - 原始缺失模式（复刻h=0可视化风格）")
    print("="*70)

    setup_matplotlib()

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model, ckpt_mean, ckpt_std = load_model(MODEL_PATH, DEVICE)

    # 加载数据
    print(f"\n加载数据集: {DATA_DIR}")
    dataset = JAXAFinetuneDataset(
        data_dir=DATA_DIR,
        series_ids=[1],
        window_size=30,
        mask_ratio=0.0,
        normalize=True,
        cache_size=50
    )

    norm_mean = dataset.mean
    norm_std = dataset.std
    print(f"  归一化参数: mean={norm_mean:.4f}, std={norm_std:.4f}")

    # 获取坐标
    h5_path = Path(DATA_DIR) / 'jaxa_knn_filled_01.h5'
    with h5py.File(h5_path, 'r') as f:
        lat_coords = f['latitude'][:]
        lon_coords = f['longitude'][:]
        timestamps = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]

    # 选择样本（均匀分布）
    n_samples = 4
    total_frames = len(dataset)
    sample_indices = [int(i * total_frames / (n_samples + 1)) for i in range(1, n_samples + 1)]

    all_results = []

    for sample_num, frame_idx in enumerate(sample_indices, 1):
        print(f"\n处理样本 {sample_num}/{n_samples} (frame={frame_idx})...")

        # 获取数据
        sample = dataset[frame_idx]

        sst_seq = sample['input_sst_seq']       # (30, H, W), 已归一化
        mask_seq = sample['mask_seq']            # (30, H, W), 1=missing
        ground_truth_sst = sample['ground_truth_sst']  # (H, W), 已归一化
        original_obs_mask = sample['original_obs_mask']  # (H, W), 1=observed
        land_mask = sample['land_mask']          # (H, W), 1=land

        # 原始缺失掩码
        original_missing_mask = (1 - original_obs_mask).astype(np.float32)

        # 推理
        pred_norm = inference_sequence(model, sst_seq, mask_seq, DEVICE)

        # 反归一化 → Kelvin → Celsius
        input_celsius = sst_seq[-1] * norm_std + norm_mean - 273.15
        knn_celsius = ground_truth_sst * norm_std + norm_mean - 273.15
        model_celsius = pred_norm * norm_std + norm_mean - 273.15

        # output composition: 观测区保留输入，缺失区用模型预测
        composed_celsius = np.where(original_missing_mask > 0, model_celsius, input_celsius)

        # 时间戳
        ts = timestamps[min(frame_idx, len(timestamps)-1)] if frame_idx < len(timestamps) else f"frame_{frame_idx}"

        # 绘图
        save_path = OUTPUT_DIR / f'sample_{sample_num:02d}_{ts[:10].replace("-","")}.png'
        mean_diff, std_diff = create_four_panel_plot(
            original_sst_celsius=input_celsius,
            knn_filled_celsius=knn_celsius,
            model_filled_celsius=composed_celsius,
            original_missing_mask=original_missing_mask,
            land_mask=land_mask,
            lon_coords=lon_coords,
            lat_coords=lat_coords,
            timestamp_str=ts[:10] if len(ts) >= 10 else ts,
            save_path=save_path
        )

        all_results.append({
            'sample': sample_num, 'frame': frame_idx,
            'mean_diff': mean_diff, 'std_diff': std_diff
        })

    # 汇总
    print("\n" + "="*70)
    print("推理完成！")
    print("="*70)
    print("\nModel-KNN差异统计（缺失区域）：")
    for res in all_results:
        print(f"  样本{res['sample']}: Mean={res['mean_diff']:.4f}°C, Std={res['std_diff']:.4f}°C")

    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
