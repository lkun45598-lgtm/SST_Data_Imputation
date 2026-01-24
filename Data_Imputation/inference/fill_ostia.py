#!/usr/bin/env python3
"""
使用训练好的FNO-CBAM模型填充OSTIA SST缺失值

用于评估预训练模型在OSTIA验证集上的效果

输入方案:
- 30天时间序列: 使用最近邻插值填充缺失区域
- mask序列: 30天的缺失掩码

输出: 模型填充后的SST数据 + 可视化 + 评估指标

作者: Claude Code
日期: 2026-01-23
"""

import os
import sys
import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from datasets.ostia_dataset import SSTDatasetTemporal

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# 默认路径配置
DEFAULT_CONFIG = {
    'val_data_path': '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/data_for_agent/processed_data/processed_sst_valid.h5',
    'model_path': '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/ostia_pretrain/best_model.pth',
    'output_dir': '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/ostia_filled_output',
    'vis_dir': '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/ostia_filled_visualization',
    'gpu_id': 0,
    'num_samples': None,  # None表示处理全部
    'vis_interval': 10,   # 每10个样本可视化一次
}


# ============================================================================
# Visualization Functions
# ============================================================================

def setup_matplotlib():
    """配置matplotlib高质量绘图"""
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1.5, labelsize=14)
    plt.rc('lines', linewidth=1.5)
    params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
    }
    plt.rcParams.update(params)


def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


def create_four_panel_plot(input_sst, ground_truth, model_output, missing_mask, land_mask,
                           lon_coords, lat_coords, sample_idx, save_path, norm_mean, norm_std):
    """
    创建4连图可视化（与fill_jaxa.py格式一致）

    1. Input SST (Day 30, with interpolated missing)
    2. Ground Truth SST
    3. FNO-CBAM Output SST
    4. Error (Model - GT) in missing regions
    """
    setup_matplotlib()

    fig = plt.figure(figsize=(28, 7))
    gs = gridspec.GridSpec(1, 6, figure=fig,
                          width_ratios=[1, 1, 1, 0.08, 1, 0.08],
                          wspace=0.15, hspace=0.1,
                          left=0.04, right=0.98, top=0.85, bottom=0.15)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    ax_error = fig.add_subplot(gs[0, 4])

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # 反归一化转为摄氏度
    input_celsius = input_sst * norm_std + norm_mean - 273.15
    gt_celsius = ground_truth * norm_std + norm_mean - 273.15
    model_celsius = model_output * norm_std + norm_mean - 273.15

    # 颜色设置
    cmap_sst = 'RdYlBu_r'
    cmap_error = 'RdBu_r'
    land_color = '#D2B48C'

    # 数据范围
    ocean_mask = land_mask == 0
    valid_data = gt_celsius[ocean_mask]
    if len(valid_data) > 0:
        vmin_sst = np.percentile(valid_data, 2)
        vmax_sst = np.percentile(valid_data, 98)
    else:
        vmin_sst, vmax_sst = 20, 32

    # 陆地显示
    land_display = np.ma.masked_where(land_mask == 0, land_mask)

    # 缺失率
    ocean_pixels = np.sum(ocean_mask)
    missing_pixels = np.sum((missing_mask > 0) & ocean_mask)
    missing_rate = missing_pixels / ocean_pixels * 100 if ocean_pixels > 0 else 0

    # ===== 1. Input SST (Day 30) =====
    ax = ax_input
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    input_masked = np.ma.masked_where(land_mask > 0, input_celsius)
    ax.pcolormesh(lon_grid, lat_grid, input_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title(f'Input SST (Day 30)\n(Missing: {missing_rate:.1f}%)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_box_aspect(1)

    # ===== 2. Ground Truth =====
    ax = ax_gt
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    gt_masked = np.ma.masked_where(land_mask > 0, gt_celsius)
    ax.pcolormesh(lon_grid, lat_grid, gt_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Ground Truth SST',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 3. Model Output =====
    ax = ax_model
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    model_masked = np.ma.masked_where(land_mask > 0, model_celsius)
    im3 = ax.pcolormesh(lon_grid, lat_grid, model_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('FNO-CBAM Output',
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

    # ===== 4. Error in Missing Regions =====
    ax = ax_error
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    error = model_celsius - gt_celsius
    error_display = error.copy()
    error_display[(missing_mask == 0) | (land_mask > 0)] = np.nan
    error_masked = np.ma.masked_where(np.isnan(error_display), error_display)

    vmax_error = 2.0
    im4 = ax.pcolormesh(lon_grid, lat_grid, error_masked,
                        cmap=cmap_error, vmin=-vmax_error, vmax=vmax_error, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Model - GT Error\n(Missing Regions Only)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # 误差统计
    if np.sum(~np.isnan(error_display)) > 0:
        mae = np.nanmean(np.abs(error_display))
        rmse = np.sqrt(np.nanmean(error_display**2))
        ax.text(0.02, 0.98, f'MAE: {mae:.3f}°C\nRMSE: {rmse:.3f}°C',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Error Colorbar
    cax4_container = fig.add_subplot(gs[0, 5])
    cax4_container.axis('off')
    cax4 = inset_axes(cax4_container, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax4_container.transAxes)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.set_label('Error (°C)', fontsize=11)

    # 标题
    fig.text(0.5, 0.98, f'OSTIA Sample Index: {sample_idx}',
            ha='center', va='top', fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Model Functions
# ============================================================================

def load_model(model_path: str, device: torch.device):
    """加载模型并返回归一化参数"""
    print(f"加载模型: {model_path}")

    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6,
        cbam_reduction_ratio=16
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_mean = checkpoint.get('norm_mean', 299.9221)
    norm_std = checkpoint.get('norm_std', 2.6919)

    epoch = checkpoint.get('epoch', 'N/A')
    val_mae = checkpoint.get('val_mae', checkpoint.get('mae', 'N/A'))
    if isinstance(val_mae, float):
        print(f"  ✓ 模型加载成功 (Epoch {epoch}, MAE: {val_mae:.4f}K)")
    else:
        print(f"  ✓ 模型加载成功 (Epoch {epoch})")
    print(f"  归一化参数: mean={norm_mean:.4f}, std={norm_std:.4f}")

    return model, norm_mean, norm_std


def run_inference(model, input_sst_seq, mask_seq, device):
    """
    运行模型推理

    Args:
        model: FNO-CBAM模型
        input_sst_seq: [30, H, W] 归一化后的输入序列
        mask_seq: [30, H, W] 缺失掩码序列
        device: 计算设备

    Returns:
        output: [H, W] 模型输出（归一化后）
    """
    # 转为tensor
    sst_tensor = torch.from_numpy(input_sst_seq).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sst_tensor, mask_tensor)

    return output.squeeze().cpu().numpy()


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_metrics(pred, gt, mask, land_mask):
    """
    计算评估指标（仅在缺失区域）

    Args:
        pred: 预测值 (归一化后)
        gt: 真值 (归一化后)
        mask: 缺失掩码 (1=缺失)
        land_mask: 陆地掩码

    Returns:
        dict: MAE, RMSE, VRMSE
    """
    # 只在海洋缺失区域计算
    valid = (mask > 0) & (land_mask == 0)

    if valid.sum() == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'vrmse': np.nan}

    pred_valid = pred[valid]
    gt_valid = gt[valid]

    mae = np.mean(np.abs(pred_valid - gt_valid))
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

    # VRMSE: 考虑空间方差的RMSE
    gt_var = np.var(gt_valid)
    if gt_var > 0:
        vrmse = rmse / np.sqrt(gt_var)
    else:
        vrmse = np.nan

    return {'mae': mae, 'rmse': rmse, 'vrmse': vrmse}


# ============================================================================
# Main
# ============================================================================

def main(args):
    print("=" * 70)
    print("OSTIA SST 缺失值填充 (预训练模型评估)")
    print("=" * 70)

    # 设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    vis_dir = Path(args.vis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model, norm_mean, norm_std = load_model(args.model_path, device)

    # 从H5文件读取经纬度坐标
    print(f"\n加载验证数据: {args.val_data_path}")
    with h5py.File(args.val_data_path, 'r') as f:
        lat_coords = f['latitude'][:]
        lon_coords = f['longitude'][:]
    print(f"  经度范围: [{lon_coords.min():.2f}°E, {lon_coords.max():.2f}°E]")
    print(f"  纬度范围: [{lat_coords.min():.2f}°N, {lat_coords.max():.2f}°N]")

    # 加载数据集
    dataset = SSTDatasetTemporal(
        hdf5_path=args.val_data_path,
        normalize=True,
        mean=norm_mean,
        std=norm_std,
        window_size=30
    )

    # 确定处理范围
    num_samples = len(dataset)
    if args.num_samples is not None:
        num_samples = min(args.num_samples, num_samples)
    print(f"处理样本数: {num_samples}")

    # 统计
    all_metrics = []
    total_processed = 0
    total_failed = 0

    # 处理每个样本
    for idx in tqdm(range(num_samples), desc="Processing"):
        try:
            # 获取数据
            sample = dataset[idx]
            input_sst_seq = sample['input_sst_seq']  # [30, H, W]
            mask_seq = sample['mask_seq']  # [30, H, W]
            gt_sst = sample['ground_truth_sst']  # [H, W]
            missing_mask = sample['missing_mask']  # [H, W]
            land_mask = sample['land_mask']  # [H, W]

            # 模型推理
            output = run_inference(model, input_sst_seq, mask_seq, device)

            # 输出组合：观测区域保留输入，缺失区域用模型预测
            final_output = input_sst_seq[-1].copy()  # 第30天的输入
            final_output[missing_mask > 0] = output[missing_mask > 0]

            # 计算指标
            metrics = compute_metrics(final_output, gt_sst, missing_mask, land_mask)
            if not np.isnan(metrics['mae']):
                all_metrics.append(metrics)

            # 可视化
            if idx % args.vis_interval == 0:
                vis_path = vis_dir / f'ostia_filled_{idx:05d}.png'
                create_four_panel_plot(
                    input_sst=input_sst_seq[-1],
                    ground_truth=gt_sst,
                    model_output=final_output,
                    missing_mask=missing_mask,
                    land_mask=land_mask,
                    lon_coords=lon_coords,
                    lat_coords=lat_coords,
                    sample_idx=idx,
                    save_path=vis_path,
                    norm_mean=norm_mean,
                    norm_std=norm_std
                )

            total_processed += 1

        except Exception as e:
            print(f"\n处理样本 {idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1
            continue

    # 汇总统计
    print(f"\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"成功处理: {total_processed} 样本")
    print(f"失败: {total_failed} 样本")

    if all_metrics:
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics])
        avg_vrmse = np.mean([m['vrmse'] for m in all_metrics if not np.isnan(m['vrmse'])])

        # 转换为温度单位
        avg_mae_k = avg_mae * norm_std
        avg_rmse_k = avg_rmse * norm_std

        print(f"\n缺失区域评估指标 (归一化):")
        print(f"  MAE:   {avg_mae:.6f}")
        print(f"  RMSE:  {avg_rmse:.6f}")
        print(f"  VRMSE: {avg_vrmse:.6f}")

        print(f"\n缺失区域评估指标 (Kelvin):")
        print(f"  MAE:   {avg_mae_k:.4f} K")
        print(f"  RMSE:  {avg_rmse_k:.4f} K")

        # 保存结果
        results_path = output_dir / 'evaluation_results.txt'
        with open(results_path, 'w') as f:
            f.write("OSTIA SST Missing Value Imputation Evaluation\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Data: {args.val_data_path}\n")
            f.write(f"Samples: {total_processed}\n")
            f.write(f"Region: Lon [{lon_coords.min():.2f}, {lon_coords.max():.2f}]E, ")
            f.write(f"Lat [{lat_coords.min():.2f}, {lat_coords.max():.2f}]N\n")
            f.write(f"\nMetrics (Normalized):\n")
            f.write(f"  MAE:   {avg_mae:.6f}\n")
            f.write(f"  RMSE:  {avg_rmse:.6f}\n")
            f.write(f"  VRMSE: {avg_vrmse:.6f}\n")
            f.write(f"\nMetrics (Kelvin):\n")
            f.write(f"  MAE:   {avg_mae_k:.4f} K\n")
            f.write(f"  RMSE:  {avg_rmse_k:.4f} K\n")
            f.write(f"\nNormalization: mean={norm_mean:.4f}, std={norm_std:.4f}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n结果已保存到: {results_path}")

    print(f"\n可视化目录: {vis_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OSTIA SST Missing Value Imputation')
    parser.add_argument('--val_data_path', type=str, default=DEFAULT_CONFIG['val_data_path'],
                       help='Path to validation HDF5 file')
    parser.add_argument('--model_path', type=str, default=DEFAULT_CONFIG['model_path'],
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                       help='Output directory for results')
    parser.add_argument('--vis_dir', type=str, default=DEFAULT_CONFIG['vis_dir'],
                       help='Directory for visualization outputs')
    parser.add_argument('--gpu_id', type=int, default=DEFAULT_CONFIG['gpu_id'],
                       help='GPU ID to use')
    parser.add_argument('--num_samples', type=int, default=DEFAULT_CONFIG['num_samples'],
                       help='Number of samples to process (None for all)')
    parser.add_argument('--vis_interval', type=int, default=DEFAULT_CONFIG['vis_interval'],
                       help='Visualization interval')

    args = parser.parse_args()
    main(args)
