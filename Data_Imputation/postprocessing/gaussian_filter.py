#!/usr/bin/env python3
"""
高斯滤波后处理脚本

对FNO-CBAM模型填充后的JAXA SST数据应用高斯滤波，平滑边缘不连续性。

输入: jaxa_filled_output/ 目录下的NC文件
输出: jaxa_filled_smoothed/ 目录下的NC文件

作者: Claude Code
日期: 2026-01-23
"""

import os
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap


# ============================================================================
# Configuration
# ============================================================================

# 输入输出路径
INPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filled_output/series_00')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filled_smoothed/series_00')
VIS_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_smoothed_visualization')

# 高斯滤波参数
GAUSSIAN_SIGMA = 1.0  # 高斯滤波标准差，值越大越平滑

# 可视化
VIS_INTERVAL = 5  # 每隔多少帧可视化一次


# ============================================================================
# Gaussian Filter Function
# ============================================================================

def apply_gaussian_filter(sst_data, land_mask, sigma=1.0):
    """
    对SST数据应用高斯滤波

    Args:
        sst_data: [H, W] SST数据 (Kelvin)，陆地为NaN
        land_mask: [H, W] 陆地掩码 (1=陆地, 0=海洋)
        sigma: 高斯滤波的标准差

    Returns:
        filtered_sst: [H, W] 滤波后的SST数据
    """
    sst = sst_data.copy()

    # 标记有效像素（非NaN且非陆地）
    mask_valid = ~np.isnan(sst) & (land_mask == 0)

    if mask_valid.sum() == 0:
        return sst

    # 临时填充NaN以便滤波
    sst_for_filter = sst.copy()
    mean_val = np.nanmean(sst)
    sst_for_filter[~mask_valid] = mean_val

    # 应用高斯滤波
    filtered = gaussian_filter(sst_for_filter, sigma=sigma)

    # 只在有效海洋区域保留滤波结果
    result = np.where(mask_valid, filtered, np.nan)

    return result


# ============================================================================
# Visualization
# ============================================================================

def format_lon(x, pos):
    return f"{abs(x):.1f}{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.1f}{'N' if y >= 0 else 'S'}"


def visualize_comparison(original_sst, smoothed_sst, land_mask, missing_mask,
                         lon, lat, timestamp, save_path):
    """
    可视化对比：原始填充 vs 高斯滤波后
    """
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 5, figure=fig,
                          width_ratios=[1, 1, 0.06, 1, 0.06],
                          wspace=0.12, left=0.04, right=0.98, top=0.85, bottom=0.15)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_smooth = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[0, 3])

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 转换为摄氏度
    orig_celsius = original_sst - 273.15
    smooth_celsius = smoothed_sst - 273.15
    diff = smooth_celsius - orig_celsius

    cmap_sst = 'RdYlBu_r'
    cmap_diff = 'RdBu_r'
    land_color = '#D2B48C'

    # 数据范围
    ocean_mask = land_mask == 0
    valid_data = orig_celsius[ocean_mask & ~np.isnan(orig_celsius)]
    if len(valid_data) > 0:
        vmin_sst = np.percentile(valid_data, 2)
        vmax_sst = np.percentile(valid_data, 98)
    else:
        vmin_sst, vmax_sst = 20, 32

    land_display = np.ma.masked_where(land_mask == 0, land_mask)

    # 1. 原始填充结果
    ax = ax_orig
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    orig_masked = np.ma.masked_where((land_mask > 0) | np.isnan(orig_celsius), orig_celsius)
    ax.pcolormesh(lon_grid, lat_grid, orig_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title('FNO-CBAM Filled\n(Original)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_box_aspect(1)

    # 2. 高斯滤波后
    ax = ax_smooth
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    smooth_masked = np.ma.masked_where((land_mask > 0) | np.isnan(smooth_celsius), smooth_celsius)
    im2 = ax.pcolormesh(lon_grid, lat_grid, smooth_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(f'Gaussian Filtered\n(sigma={GAUSSIAN_SIGMA})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # SST Colorbar
    cax2 = fig.add_subplot(gs[0, 2])
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('SST (C)')

    # 3. 差异图（只在缺失区域）
    ax = ax_diff
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # 只显示缺失区域的差异
    diff_display = diff.copy()
    diff_display[(missing_mask == 0) | (land_mask > 0)] = np.nan
    diff_masked = np.ma.masked_where(np.isnan(diff_display), diff_display)

    im3 = ax.pcolormesh(lon_grid, lat_grid, diff_masked,
                        cmap=cmap_diff, vmin=-0.5, vmax=0.5, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Smoothed - Original\n(Missing Region)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # 差异统计
    if np.sum(~np.isnan(diff_display)) > 0:
        mean_diff = np.nanmean(diff_display)
        std_diff = np.nanstd(diff_display)
        ax.text(0.02, 0.98, f'Mean: {mean_diff:.4f}C\nStd: {std_diff:.4f}C',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Diff Colorbar
    cax3 = fig.add_subplot(gs[0, 4])
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('Diff (C)')

    fig.text(0.5, 0.98, f'Gaussian Filter Post-processing: {timestamp[:10]}',
            ha='center', va='top', fontsize=16, fontweight='bold')

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main Processing
# ============================================================================

def process_nc_file(input_path: Path, output_path: Path, sigma: float):
    """
    处理单个NC文件
    """
    # 读取输入文件
    with nc.Dataset(input_path, 'r') as f_in:
        lat = f_in.variables['lat'][:]
        lon = f_in.variables['lon'][:]
        time_var = f_in.variables['time'][:]
        sst_filled = f_in.variables['sst_filled'][0, :, :]
        sst_knn = f_in.variables['sst_knn'][0, :, :]
        sst_filtered = f_in.variables['sst_filtered'][0, :, :]
        missing_mask = f_in.variables['original_missing_mask'][0, :, :]

        # 从filled数据推断land_mask
        land_mask = np.isnan(sst_knn).astype(np.uint8)

    # 应用高斯滤波
    sst_smoothed = apply_gaussian_filter(sst_filled, land_mask, sigma=sigma)

    # 保存输出文件
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(output_path, 'w', format='NETCDF4') as f_out:
        f_out.createDimension('lat', len(lat))
        f_out.createDimension('lon', len(lon))
        f_out.createDimension('time', 1)

        lat_var = f_out.createVariable('lat', 'f4', ('lat',))
        lon_var = f_out.createVariable('lon', 'f4', ('lon',))
        time_var_out = f_out.createVariable('time', 'S32', ('time',))
        sst_smoothed_var = f_out.createVariable('sst_smoothed', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        sst_original_var = f_out.createVariable('sst_filled_original', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        sst_knn_var = f_out.createVariable('sst_knn', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        mask_var = f_out.createVariable('original_missing_mask', 'u1', ('time', 'lat', 'lon'))

        lat_var[:] = lat
        lon_var[:] = lon
        time_var_out[:] = time_var
        sst_smoothed_var[0, :, :] = sst_smoothed
        sst_original_var[0, :, :] = sst_filled
        sst_knn_var[0, :, :] = sst_knn
        mask_var[0, :, :] = missing_mask

        f_out.title = 'JAXA SST Filled by FNO-CBAM Model (Gaussian Smoothed)'
        f_out.source = 'Post-processed from FNO-CBAM filled output'
        f_out.postprocess = f'Gaussian filter (sigma={sigma})'
        f_out.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        sst_smoothed_var.units = 'Kelvin'
        sst_smoothed_var.long_name = 'Sea Surface Temperature (Gaussian Smoothed)'
        sst_original_var.units = 'Kelvin'
        sst_original_var.long_name = 'Sea Surface Temperature (Original Filled)'

    return {
        'lat': lat, 'lon': lon,
        'sst_original': sst_filled,
        'sst_smoothed': sst_smoothed,
        'land_mask': land_mask,
        'missing_mask': missing_mask,
        'timestamp': time_var[0].decode('utf-8') if isinstance(time_var[0], bytes) else str(time_var[0])
    }


def main():
    print("=" * 70)
    print("JAXA SST 高斯滤波后处理")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"高斯滤波 sigma: {GAUSSIAN_SIGMA}")
    print("=" * 70)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # 查找所有NC文件
    nc_files = sorted(INPUT_DIR.rglob('*.nc'))

    if not nc_files:
        print(f"未找到NC文件: {INPUT_DIR}")
        return

    print(f"找到 {len(nc_files)} 个NC文件")

    total_processed = 0
    total_failed = 0

    for i, input_path in enumerate(tqdm(nc_files, desc="处理中")):
        try:
            # 构建输出路径（保持目录结构）
            rel_path = input_path.relative_to(INPUT_DIR)
            output_path = OUTPUT_DIR / rel_path.parent / input_path.name.replace('.nc', '_smoothed.nc')

            # 处理文件
            result = process_nc_file(input_path, output_path, GAUSSIAN_SIGMA)

            # 可视化
            if i % VIS_INTERVAL == 0:
                vis_name = input_path.stem + '_comparison.png'
                vis_path = VIS_DIR / vis_name
                visualize_comparison(
                    original_sst=result['sst_original'],
                    smoothed_sst=result['sst_smoothed'],
                    land_mask=result['land_mask'],
                    missing_mask=result['missing_mask'],
                    lon=result['lon'],
                    lat=result['lat'],
                    timestamp=result['timestamp'],
                    save_path=vis_path
                )

            total_processed += 1

        except Exception as e:
            print(f"\n处理失败 {input_path}: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1
            continue

    print(f"\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"成功处理: {total_processed} 个文件")
    print(f"失败: {total_failed} 个文件")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"可视化目录: {VIS_DIR}")


if __name__ == '__main__':
    main()
