#!/usr/bin/env python3
"""
使用训练好的FNO-CBAM模型填充JAXA SST缺失值

混合输入方案:
- 前29天: KNN填充的完整数据 (提供丰富的历史时序信息)
- 第30天: 高斯滤波数据 + 缺失区域用均值填充 (与训练时一致)

输出: 模型填充后的SST数据 (nc格式) + 可视化

作者: Claude Code
日期: 2026-01-23
"""

import os
import sys
import torch
import numpy as np
import h5py
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import imageio

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# 数据路径
KNN_FILLED_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled')
FILTERED_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filtered')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filled_output')
VIS_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filled_visualization')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_8years/best_model.pth'

# 模型参数
WINDOW_SIZE = 30  # 30天输入序列
GPU_ID = 4  # 使用GPU 4
NUM_TEST_SAMPLES = 30  # 测试样本数 (设为None则处理全部)

# 要处理的序列ID（对应jaxa_knn_filled_XX.h5 和 jaxa_filtered_XX.h5）
SERIES_IDS = [0]  # 只测试第一个序列


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


def create_four_panel_plot(original_sst, knn_filled_sst, model_filled_sst,
                           original_missing_mask, land_mask,
                           lon_coords, lat_coords, timestamp, save_path):
    """
    创建4连图可视化

    1. Original JAXA SST (with missing) - 滤波后的原始观测
    2. KNN Filled SST - KNN粗填充结果
    3. Model Filled SST - 模型填充结果
    4. Difference (Model - KNN) in missing regions
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

    # 转换为摄氏度
    original_celsius = original_sst - 273.15
    knn_celsius = knn_filled_sst - 273.15
    model_celsius = model_filled_sst - 273.15

    # 颜色设置
    cmap_sst = 'RdYlBu_r'
    cmap_diff = 'RdBu_r'
    land_color = '#D2B48C'

    # 数据范围
    ocean_mask = land_mask == 0
    valid_data = knn_celsius[ocean_mask]
    if len(valid_data) > 0:
        vmin_sst = np.percentile(valid_data, 2)
        vmax_sst = np.percentile(valid_data, 98)
    else:
        vmin_sst, vmax_sst = 20, 32

    # 陆地显示
    land_display = np.ma.masked_where(land_mask == 0, land_mask)

    # 计算缺失率
    ocean_pixels = np.sum(ocean_mask)
    missing_pixels = np.sum((original_missing_mask > 0) & ocean_mask)
    missing_rate = missing_pixels / ocean_pixels * 100 if ocean_pixels > 0 else 0

    # ===== 1. Original SST (显示原始观测，缺失区域为浅蓝色) =====
    ax = ax_orig
    ax.set_facecolor('skyblue')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    orig_display = original_celsius.copy()
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

    knn_masked = np.ma.masked_where(land_mask > 0, knn_celsius)
    ax.pcolormesh(lon_grid, lat_grid, knn_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('KNN Filled SST\n(Coarse Fill)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 3. Model Filled SST =====
    ax = ax_model
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    model_masked = np.ma.masked_where(land_mask > 0, model_celsius)
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

    # ===== 4. Difference in Missing Regions =====
    ax = ax_diff
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    diff = model_celsius - knn_celsius
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
    fig.text(0.5, 0.98, f'Date: {timestamp[:10]}',
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

    print(f"  ✓ 模型加载成功 (Epoch {checkpoint.get('epoch', 'N/A')}, MAE: {checkpoint.get('val_mae', 'N/A'):.4f}K)")
    print(f"  归一化参数: mean={norm_mean:.4f}, std={norm_std:.4f}")

    return model, norm_mean, norm_std


def fill_sst_with_model_hybrid(model, sst_knn_seq, sst_filtered_day30, mask_seq,
                                land_mask, norm_mean, norm_std, device):
    """
    混合输入方案填充SST缺失值

    Args:
        model: FNO-CBAM模型
        sst_knn_seq: [29, H, W] 前29天KNN填充后的SST序列 (Kelvin)
        sst_filtered_day30: [H, W] 第30天滤波后的SST (有NaN)
        mask_seq: [30, H, W] 30天的原始缺失掩码 (1=缺失, 0=有效)
        land_mask: [H, W] 陆地掩码
        norm_mean, norm_std: 归一化参数
        device: 计算设备

    Returns:
        filled_sst: [H, W] 填充后的SST (Kelvin)
    """
    H, W = sst_filtered_day30.shape

    # 构建30天输入序列
    sst_seq = np.zeros((WINDOW_SIZE, H, W), dtype=np.float32)

    # 前29天: KNN填充数据
    sst_seq[:WINDOW_SIZE-1] = sst_knn_seq

    # 第30天: 滤波数据 + 缺失区域用均值填充
    sst_day30 = sst_filtered_day30.copy()
    sst_day30 = np.where(np.isnan(sst_day30), norm_mean, sst_day30)
    sst_seq[-1] = sst_day30

    # 归一化
    sst_norm = (sst_seq - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)

    # 转为tensor
    sst_tensor = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)

    # 反归一化
    pred_kelvin = pred.squeeze().cpu().numpy() * norm_std + norm_mean

    # 输出组合：观测区域保留滤波值，缺失区域用模型预测
    last_day_mask = mask_seq[-1]
    filled_sst = sst_filtered_day30.copy()
    filled_sst = np.where(last_day_mask > 0, pred_kelvin, filled_sst)
    filled_sst = np.where(np.isnan(filled_sst), pred_kelvin, filled_sst)  # 处理剩余NaN
    filled_sst[land_mask > 0] = np.nan

    return filled_sst


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_knn_data(h5_path):
    """加载KNN填充后的H5文件数据"""
    with h5py.File(h5_path, 'r') as f:
        sst_data = f['sst_data'][:]
        original_missing_mask = f['original_missing_mask'][:]
        land_mask = f['land_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]

    return {
        'sst_data': sst_data,
        'original_missing_mask': original_missing_mask,
        'land_mask': land_mask,
        'lat': lat,
        'lon': lon,
        'timestamps': timestamps
    }


def load_filtered_data(h5_path):
    """加载滤波后的H5文件数据"""
    with h5py.File(h5_path, 'r') as f:
        sst_data = f['sst_data'][:]
        missing_mask = f['missing_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]

    return {
        'sst_data': sst_data,
        'missing_mask': missing_mask,
        'lat': lat,
        'lon': lon,
        'timestamps': timestamps
    }


def get_hybrid_30day_sequence(sst_knn, mask_knn, sst_filtered, mask_filtered, idx, window_size=30):
    """
    获取混合的30天序列

    前29天: KNN填充数据
    第30天: 滤波数据
    """
    H, W = sst_knn.shape[1], sst_knn.shape[2]

    sst_knn_seq = np.zeros((window_size - 1, H, W), dtype=np.float32)
    mask_seq = np.zeros((window_size, H, W), dtype=np.float32)

    # 前29天
    for t in range(window_size - 1):
        src_idx = idx - (window_size - 1) + t
        if src_idx < 0:
            src_idx = 0
        sst_knn_seq[t] = sst_knn[src_idx]
        mask_seq[t] = mask_knn[src_idx]

    # 第30天
    sst_filtered_day30 = sst_filtered[idx].copy()
    mask_seq[-1] = mask_filtered[idx]

    return sst_knn_seq, sst_filtered_day30, mask_seq


# ============================================================================
# Output Functions
# ============================================================================

def save_filled_nc(output_path: Path, filled_sst: np.ndarray, knn_sst: np.ndarray,
                   filtered_sst: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   timestamp: str, original_missing_mask: np.ndarray):
    """保存填充后的SST为NetCDF文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        f.createDimension('lat', len(lat))
        f.createDimension('lon', len(lon))
        f.createDimension('time', 1)

        lat_var = f.createVariable('lat', 'f4', ('lat',))
        lon_var = f.createVariable('lon', 'f4', ('lon',))
        time_var = f.createVariable('time', 'S32', ('time',))
        sst_filled_var = f.createVariable('sst_filled', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        sst_knn_var = f.createVariable('sst_knn', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        sst_filtered_var = f.createVariable('sst_filtered', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        mask_var = f.createVariable('original_missing_mask', 'u1', ('time', 'lat', 'lon'))

        lat_var[:] = lat
        lon_var[:] = lon
        time_var[:] = np.array([timestamp], dtype='S32')
        sst_filled_var[0, :, :] = filled_sst
        sst_knn_var[0, :, :] = knn_sst
        sst_filtered_var[0, :, :] = filtered_sst
        mask_var[0, :, :] = original_missing_mask

        f.title = 'JAXA SST Filled by FNO-CBAM Model (Hybrid Input)'
        f.source = 'Hybrid input: KNN[Day1-29] + Filtered+Mean[Day30]'
        f.model = 'FNO_CBAM_SST_Temporal (JAXA 8-year fine-tuned)'
        f.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        sst_filled_var.units = 'Kelvin'
        sst_filled_var.long_name = 'Sea Surface Temperature (Model Filled)'
        sst_knn_var.units = 'Kelvin'
        sst_knn_var.long_name = 'Sea Surface Temperature (KNN Filled)'
        sst_filtered_var.units = 'Kelvin'
        sst_filtered_var.long_name = 'Sea Surface Temperature (Filtered, with missing)'


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("JAXA SST 缺失值填充 (混合输入方案)")
    print("前29天: KNN填充 | 第30天: 滤波+均值填充")
    print("=" * 70)

    # 设备
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    model, norm_mean, norm_std = load_model(MODEL_PATH, device)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # 处理统计
    total_processed = 0
    total_failed = 0
    vis_paths = []

    # 遍历每个序列
    for series_id in SERIES_IDS:
        knn_path = KNN_FILLED_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'
        filtered_path = FILTERED_DIR / f'jaxa_filtered_{series_id:02d}.h5'

        if not knn_path.exists():
            print(f"\n跳过不存在的KNN文件: {knn_path}")
            continue
        if not filtered_path.exists():
            print(f"\n跳过不存在的滤波文件: {filtered_path}")
            continue

        print(f"\n处理序列 {series_id}")

        # 加载数据
        knn_data = load_knn_data(knn_path)
        filtered_data = load_filtered_data(filtered_path)

        sst_knn = knn_data['sst_data']
        mask_knn = knn_data['original_missing_mask']
        land_mask = knn_data['land_mask']
        lat = knn_data['lat']
        lon = knn_data['lon']
        timestamps = knn_data['timestamps']

        sst_filtered = filtered_data['sst_data']
        mask_filtered = filtered_data['missing_mask']

        num_frames = min(sst_knn.shape[0], sst_filtered.shape[0])
        print(f"  帧数: {num_frames}")

        # 确定处理范围
        if NUM_TEST_SAMPLES is not None:
            start_idx = WINDOW_SIZE - 1
            indices = list(range(start_idx, min(start_idx + NUM_TEST_SAMPLES, num_frames)))
        else:
            indices = list(range(num_frames))

        # 可视化间隔
        vis_interval = max(1, len(indices) // 10)

        for i, idx in enumerate(tqdm(indices, desc=f"Series {series_id}")):
            try:
                # 获取混合30天序列
                sst_knn_seq, sst_filtered_day30, mask_seq = get_hybrid_30day_sequence(
                    sst_knn, mask_knn, sst_filtered, mask_filtered, idx, WINDOW_SIZE
                )

                # 模型填充
                filled_sst = fill_sst_with_model_hybrid(
                    model, sst_knn_seq, sst_filtered_day30, mask_seq,
                    land_mask, norm_mean, norm_std, device
                )

                # 保存NC文件
                timestamp = timestamps[idx]
                output_file = OUTPUT_DIR / f'series_{series_id:02d}' / f'jaxa_filled_{timestamp.replace(":", "").replace("-", "")}.nc'

                # 准备保存的数据
                knn_sst_day = sst_knn[idx].copy()
                knn_sst_day[land_mask > 0] = np.nan

                filtered_sst_day = sst_filtered_day30.copy()
                filtered_sst_day[land_mask > 0] = np.nan

                save_filled_nc(
                    output_file,
                    filled_sst=filled_sst,
                    knn_sst=knn_sst_day,
                    filtered_sst=filtered_sst_day,
                    lat=lat,
                    lon=lon,
                    timestamp=timestamp,
                    original_missing_mask=mask_seq[-1].astype(np.uint8)
                )

                # 可视化
                if i % vis_interval == 0:
                    vis_path = VIS_DIR / f'series_{series_id:02d}_{timestamp.replace(":", "").replace("-", "")}.png'
                    create_four_panel_plot(
                        original_sst=filtered_sst_day,
                        knn_filled_sst=knn_sst_day,
                        model_filled_sst=filled_sst,
                        original_missing_mask=mask_seq[-1],
                        land_mask=land_mask,
                        lon_coords=lon,
                        lat_coords=lat,
                        timestamp=timestamp,
                        save_path=vis_path
                    )
                    vis_paths.append(str(vis_path))

                total_processed += 1

            except Exception as e:
                print(f"\n  处理帧 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1
                continue

    # 生成GIF
    if vis_paths:
        print("\n生成GIF动画...")
        gif_path = VIS_DIR / 'jaxa_filled_animation.gif'
        with imageio.get_writer(gif_path, mode='I', fps=0.5) as writer:
            for img_path in vis_paths:
                image = imageio.v2.imread(img_path)
                writer.append_data(image)
        print(f"GIF保存在: {gif_path}")

    # 统计
    print(f"\n" + "=" * 70)
    print("填充完成!")
    print("=" * 70)
    print(f"成功处理: {total_processed} 帧")
    print(f"失败: {total_failed} 帧")
    print(f"NC输出目录: {OUTPUT_DIR}")
    print(f"可视化目录: {VIS_DIR}")


if __name__ == '__main__':
    main()
