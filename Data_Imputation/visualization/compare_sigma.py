#!/usr/bin/env python3
"""
对比不同sigma值的高斯滤波效果 - 全部放在一张图
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap


# 配置
INPUT_FILE = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filled_output/series_00/jaxa_filled_20170808T000000.nc')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/sigma_comparison')
SIGMA_VALUES = [0.1, 0.5, 1.0]


def setup_matplotlib():
    """配置matplotlib高质量绘图"""
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=1.5, labelsize=12)
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


def apply_gaussian_filter(sst_data, land_mask, sigma):
    """应用高斯滤波"""
    sst = sst_data.copy()
    mask_valid = ~np.isnan(sst) & (land_mask == 0)
    if mask_valid.sum() == 0:
        return sst
    sst_for_filter = sst.copy()
    mean_val = np.nanmean(sst)
    sst_for_filter[~mask_valid] = mean_val
    filtered = gaussian_filter(sst_for_filter, sigma=sigma)
    result = np.where(mask_valid, filtered, np.nan)
    return result


def format_lon(x, pos):
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}"


def main():
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取数据
    with nc.Dataset(INPUT_FILE, 'r') as f:
        lat = f.variables['lat'][:]
        lon = f.variables['lon'][:]
        sst_filled = f.variables['sst_filled'][0, :, :]
        sst_knn = f.variables['sst_knn'][0, :, :]
        missing_mask = f.variables['original_missing_mask'][0, :, :]

    land_mask = np.isnan(sst_knn).astype(np.uint8)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 转换为摄氏度
    orig_celsius = sst_filled - 273.15

    # 数据范围
    ocean_mask = land_mask == 0
    valid_data = orig_celsius[ocean_mask & ~np.isnan(orig_celsius)]
    vmin_sst = np.percentile(valid_data, 2)
    vmax_sst = np.percentile(valid_data, 98)

    land_display = np.ma.masked_where(land_mask == 0, land_mask)
    land_color = '#D2B48C'
    cmap_sst = 'RdYlBu_r'

    # 预计算所有滤波结果
    all_results = {'Original': orig_celsius}
    for sigma in SIGMA_VALUES:
        sst_smoothed = apply_gaussian_filter(sst_filled, land_mask, sigma)
        all_results[f'σ={sigma}'] = sst_smoothed - 273.15

    # 创建图 - 1行4列布局
    # Original, σ=0.1, σ=0.5, σ=1.0, colorbar
    fig = plt.figure(figsize=(24, 6))

    gs = gridspec.GridSpec(1, 5, figure=fig,
                          width_ratios=[1, 1, 1, 1, 0.06],
                          wspace=0.10, hspace=0.1,
                          left=0.04, right=0.96, top=0.88, bottom=0.12)

    # 定义每个subplot的位置和标题
    plot_configs = [
        (0, 0, 'Original (No Filter)'),
        (0, 1, 'Gaussian σ=0.1'),
        (0, 2, 'Gaussian σ=0.5'),
        (0, 3, 'Gaussian σ=1.0'),
    ]

    data_keys = ['Original', 'σ=0.1', 'σ=0.5', 'σ=1.0']

    im = None
    for i, (row, col, title) in enumerate(plot_configs):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('lightgray')

        # 画陆地
        ax.pcolormesh(lon_grid, lat_grid, land_display,
                      cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

        # 画SST数据
        data = all_results[data_keys[i]]
        data_masked = np.ma.masked_where((land_mask > 0) | np.isnan(data), data)
        im = ax.pcolormesh(lon_grid, lat_grid, data_masked,
                          cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')

        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.locator_params(axis='x', nbins=4)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
        ax.set_xlabel('Longitude', fontsize=11)

        if col == 0:
            ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
            ax.locator_params(axis='y', nbins=5)
            ax.set_ylabel('Latitude', fontsize=11)
        else:
            ax.set_yticks([])

        ax.set_aspect('equal')

    # 最后一格放colorbar
    cax = fig.add_subplot(gs[0, 4])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('SST (°C)', fontsize=13)

    # 总标题
    fig.suptitle('Gaussian Filter Comparison: Different Sigma Values  |  Date: 2017-08-08',
                fontsize=16, fontweight='bold', y=0.96)

    # 保存
    save_path = OUTPUT_DIR / 'gaussian_sigma_comparison_all.png'
    plt.savefig(save_path, dpi=512, bbox_inches='tight')
    plt.close()
    print(f"图片保存: {save_path}")


if __name__ == '__main__':
    main()
