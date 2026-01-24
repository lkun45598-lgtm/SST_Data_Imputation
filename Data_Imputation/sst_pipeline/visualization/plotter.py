#!/usr/bin/env python3
"""
SST Pipeline 可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Optional, Dict, Tuple


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


def format_lon(x, pos):
    """经度格式化"""
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    """纬度格式化"""
    return f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}"


class Plotter:
    """可视化绘图器"""

    def __init__(self, dpi: int = 300, figsize: Tuple = (20, 5),
                 cmap: str = 'RdYlBu_r', land_color: str = '#D2B48C'):
        """
        初始化绘图器

        Args:
            dpi: 图像分辨率
            figsize: 图像大小
            cmap: SST颜色映射
            land_color: 陆地颜色
        """
        self.dpi = dpi
        self.figsize = figsize
        self.cmap = cmap
        self.land_color = land_color
        setup_matplotlib()

    def plot_reconstruction(self, result: Dict, metadata: Dict,
                           output_path: Optional[Path] = None,
                           title_prefix: str = "") -> None:
        """
        绘制重建结果对比图

        Args:
            result: 包含预测结果的字典
            metadata: 元数据字典
            output_path: 输出路径
            title_prefix: 标题前缀
        """
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        target_date = metadata.get('target_date', '')

        # 创建网格
        if lat is not None and lon is not None:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        else:
            H, W = result['composed'].shape
            lon_grid, lat_grid = np.meshgrid(np.arange(W), np.arange(H))

        # 准备数据
        land_mask = result['land_mask']
        land_display = np.ma.masked_where(land_mask == 0, land_mask)

        # 获取各个数据
        knn_filled = result.get('knn_filled')
        day30_input = result.get('day30_input')
        model_output = result.get('predicted_raw')
        final_output = result.get('composed')
        final_gaussian = result.get('gaussian_filtered')

        # 转换为摄氏度
        def to_celsius(data):
            if data is None:
                return None
            d = data.copy()
            if np.nanmean(d[~np.isnan(d)]) > 100:  # 开尔文
                d = d - 273.15
            return d

        knn_filled = to_celsius(knn_filled)
        day30_input = to_celsius(day30_input)
        model_output = to_celsius(model_output)
        final_output = to_celsius(final_output)
        final_gaussian = to_celsius(final_gaussian)

        # 确定数据范围
        all_data = [d for d in [knn_filled, day30_input, model_output, final_output, final_gaussian] if d is not None]
        if all_data:
            combined = np.concatenate([d[~np.isnan(d)].flatten() for d in all_data])
            vmin = np.percentile(combined, 2)
            vmax = np.percentile(combined, 98)
        else:
            vmin, vmax = 15, 30

        # 创建4面板图
        fig = plt.figure(figsize=(24, 6))
        gs = gridspec.GridSpec(1, 5, figure=fig,
                              width_ratios=[1, 1, 1, 1, 0.06],
                              wspace=0.10, hspace=0.1,
                              left=0.04, right=0.96, top=0.88, bottom=0.12)

        panels = []
        if knn_filled is not None:
            panels.append(('KNN Filled (Day 30)', knn_filled))
        if day30_input is not None:
            panels.append(('Model Input (Masked)', day30_input))
        if model_output is not None:
            panels.append(('Model Output', model_output))
        if final_gaussian is not None:
            panels.append(('Final (Gaussian σ=1.0)', final_gaussian))
        elif final_output is not None:
            panels.append(('Final Output', final_output))

        # 确保有4个面板
        while len(panels) < 4:
            panels.append(('Empty', np.full_like(land_mask, np.nan, dtype=float)))

        im = None
        for i, (title, data) in enumerate(panels[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor('lightgray')

            # 画陆地
            ax.pcolormesh(lon_grid, lat_grid, land_display,
                         cmap=ListedColormap([self.land_color]),
                         vmin=0, vmax=1, shading='auto')

            # 画SST数据
            data_masked = np.ma.masked_where((land_mask > 0) | np.isnan(data), data)
            im = ax.pcolormesh(lon_grid, lat_grid, data_masked,
                              cmap=self.cmap, vmin=vmin, vmax=vmax, shading='auto')

            ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
            ax.locator_params(axis='x', nbins=4)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
            ax.set_xlabel('Longitude', fontsize=11)

            if i == 0:
                ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
                ax.locator_params(axis='y', nbins=5)
                ax.set_ylabel('Latitude', fontsize=11)
            else:
                ax.set_yticks([])

            ax.set_aspect('equal')

        # colorbar
        cax = fig.add_subplot(gs[0, 4])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('SST (°C)', fontsize=13)

        # 总标题
        fig.suptitle(f'{title_prefix}SST Reconstruction | Date: {target_date}',
                    fontsize=16, fontweight='bold', y=0.96)

        # 保存
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[Visualization] 图像已保存: {output_path}")

        plt.close()


def plot_reconstruction(result: Dict, metadata: Dict,
                       output_path: Optional[Path] = None,
                       dpi: int = 300) -> None:
    """绘制重建结果的便捷函数"""
    plotter = Plotter(dpi=dpi)
    plotter.plot_reconstruction(result, metadata, output_path)


def create_plotter(config) -> Plotter:
    """从配置创建绘图器的工厂函数"""
    return Plotter(
        dpi=config.visualization.dpi,
        figsize=config.visualization.figsize,
        cmap=config.visualization.cmap,
        land_color=config.visualization.land_color
    )
