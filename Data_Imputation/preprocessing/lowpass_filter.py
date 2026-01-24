#!/usr/bin/env python3
"""
JAXA SST Low-Pass Filter Processor
工业级低通滤波：去除高频噪声

处理流程位置:
    时间加权融合(23小时) → [本脚本:滤波] → KNN插值

支持的滤波方法:
    1. gaussian - 高斯滤波（默认，平滑效果好）
    2. median - 中值滤波（去除椒盐噪声）
    3. uniform - 均值滤波（简单移动平均）
    4. bilateral - 双边滤波（保边平滑）

输入: jaxa_weighted_aligned/*.h5 (时间加权填充后，约40%缺失)
输出: jaxa_filtered/*.h5 (滤波后，缺失区域不变)

使用方法:
    # 测试单帧可视化
    python jaxa_lowpass_filter.py --mode test

    # 处理所有序列
    python jaxa_lowpass_filter.py --mode full --method gaussian --sigma 1.5

    # 处理单个序列
    python jaxa_lowpass_filter.py --mode single --series 0 --method gaussian --sigma 1.5

作者: Claude Code
日期: 2026-01-20
"""

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_weighted_aligned')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filtered')

# Default filter parameters
DEFAULT_METHOD = 'gaussian'
DEFAULT_SIGMA = 1.5  # For gaussian filter
DEFAULT_KERNEL_SIZE = 3  # For median/uniform filter
DEFAULT_NUM_WORKERS = 32


# ============================================================
# Filter Functions
# ============================================================

def apply_gaussian_filter(data: np.ndarray, mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    高斯低通滤波 - 工业标准平滑滤波器

    特点:
    - 频率响应为高斯函数，平滑过渡
    - 对连续场（如温度场）效果好
    - sigma越大，滤波越强

    Args:
        data: SST数据 (H, W)，缺失位置为NaN
        mask: 缺失掩码，1=缺失，0=有效
        sigma: 高斯核标准差，建议1.0-3.0

    Returns:
        filtered_data: 滤波后的数据，缺失位置保持NaN
    """
    # 复制数据
    filtered = data.copy()

    # 将NaN替换为局部均值或0
    valid_mask = (mask == 0) & (~np.isnan(data))
    if valid_mask.sum() == 0:
        return filtered

    # 临时填充NaN为有效区域均值
    temp_data = data.copy()
    mean_val = np.nanmean(data[valid_mask])
    temp_data[~valid_mask] = mean_val

    # 应用高斯滤波
    filtered_temp = ndimage.gaussian_filter(temp_data, sigma=sigma, mode='reflect')

    # 只保留原始有效区域的滤波结果
    filtered[valid_mask] = filtered_temp[valid_mask]

    return filtered


def apply_median_filter(data: np.ndarray, mask: np.ndarray, size: int = 3) -> np.ndarray:
    """
    中值滤波 - 去除椒盐噪声和异常值

    特点:
    - 非线性滤波，保边能力强
    - 对异常值（outliers）鲁棒
    - 常用于预处理

    Args:
        data: SST数据 (H, W)
        mask: 缺失掩码
        size: 滤波器窗口大小（奇数）

    Returns:
        filtered_data: 滤波后的数据
    """
    filtered = data.copy()
    valid_mask = (mask == 0) & (~np.isnan(data))

    if valid_mask.sum() == 0:
        return filtered

    # 临时填充
    temp_data = data.copy()
    mean_val = np.nanmean(data[valid_mask])
    temp_data[~valid_mask] = mean_val

    # 应用中值滤波
    filtered_temp = ndimage.median_filter(temp_data, size=size, mode='reflect')

    # 只保留有效区域
    filtered[valid_mask] = filtered_temp[valid_mask]

    return filtered


def apply_uniform_filter(data: np.ndarray, mask: np.ndarray, size: int = 3) -> np.ndarray:
    """
    均值滤波（移动平均）- 简单快速的平滑滤波

    特点:
    - 线性滤波，计算简单
    - 等权重平均
    - 对高频噪声有效

    Args:
        data: SST数据 (H, W)
        mask: 缺失掩码
        size: 滤波器窗口大小

    Returns:
        filtered_data: 滤波后的数据
    """
    filtered = data.copy()
    valid_mask = (mask == 0) & (~np.isnan(data))

    if valid_mask.sum() == 0:
        return filtered

    temp_data = data.copy()
    mean_val = np.nanmean(data[valid_mask])
    temp_data[~valid_mask] = mean_val

    # 应用均值滤波
    filtered_temp = ndimage.uniform_filter(temp_data, size=size, mode='reflect')

    filtered[valid_mask] = filtered_temp[valid_mask]

    return filtered


def apply_bilateral_filter(data: np.ndarray, mask: np.ndarray,
                           sigma_space: float = 1.5, sigma_color: float = 2.0) -> np.ndarray:
    """
    双边滤波 - 保边平滑滤波器

    特点:
    - 同时考虑空间距离和值域差异
    - 在平滑区域滤波，在边缘保持
    - 计算较慢但效果好

    注意: 这里使用简化版本（先高斯后对比）

    Args:
        data: SST数据 (H, W)
        mask: 缺失掩码
        sigma_space: 空间高斯标准差
        sigma_color: 值域高斯标准差（开尔文）

    Returns:
        filtered_data: 滤波后的数据
    """
    filtered = data.copy()
    valid_mask = (mask == 0) & (~np.isnan(data))

    if valid_mask.sum() == 0:
        return filtered

    temp_data = data.copy()
    mean_val = np.nanmean(data[valid_mask])
    temp_data[~valid_mask] = mean_val

    # 简化双边滤波：高斯滤波 + 梯度加权混合
    gaussian_filtered = ndimage.gaussian_filter(temp_data, sigma=sigma_space, mode='reflect')

    # 计算梯度幅值（边缘检测）
    grad_y = ndimage.sobel(temp_data, axis=0)
    grad_x = ndimage.sobel(temp_data, axis=1)
    gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)

    # 边缘权重：梯度大的地方保留原值
    edge_weight = 1 - np.exp(-gradient_magnitude**2 / (2 * sigma_color**2))
    edge_weight = np.clip(edge_weight, 0, 1)

    # 混合：边缘保留原值，平滑区域用滤波值
    filtered_temp = edge_weight * temp_data + (1 - edge_weight) * gaussian_filtered

    filtered[valid_mask] = filtered_temp[valid_mask]

    return filtered


def get_filter_function(method: str):
    """获取滤波函数"""
    filters = {
        'gaussian': apply_gaussian_filter,
        'median': apply_median_filter,
        'uniform': apply_uniform_filter,
        'bilateral': apply_bilateral_filter,
    }
    if method not in filters:
        raise ValueError(f"Unknown filter method: {method}. Choose from {list(filters.keys())}")
    return filters[method]


# ============================================================
# Frame Processing
# ============================================================

def filter_single_frame(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    处理单帧的滤波

    Args:
        args: (frame_idx, sst_frame, missing_mask, method, params)

    Returns:
        (frame_idx, filtered_frame)
    """
    frame_idx, sst_frame, missing_mask, method, params = args

    filter_func = get_filter_function(method)

    if method == 'gaussian':
        filtered = filter_func(sst_frame, missing_mask, sigma=params.get('sigma', 1.5))
    elif method in ['median', 'uniform']:
        filtered = filter_func(sst_frame, missing_mask, size=params.get('size', 3))
    elif method == 'bilateral':
        filtered = filter_func(sst_frame, missing_mask,
                              sigma_space=params.get('sigma_space', 1.5),
                              sigma_color=params.get('sigma_color', 2.0))
    else:
        filtered = sst_frame.copy()

    return frame_idx, filtered


# ============================================================
# Series Processing
# ============================================================

def process_series(input_path: Path, output_path: Path,
                   method: str = DEFAULT_METHOD,
                   params: Dict = None,
                   num_workers: int = DEFAULT_NUM_WORKERS) -> Dict:
    """
    处理单个时间序列

    Args:
        input_path: 输入h5文件路径
        output_path: 输出h5文件路径
        method: 滤波方法
        params: 滤波参数
        num_workers: 并行worker数

    Returns:
        统计信息字典
    """
    if params is None:
        params = {'sigma': 1.5, 'size': 3}

    print(f"\n{'='*70}")
    print(f"Processing: {input_path.name}")
    print(f"Method: {method}, Params: {params}")
    print(f"{'='*70}")

    start_time = time.time()

    # 读取数据
    with h5py.File(input_path, 'r') as f:
        sst_data = f['sst_data'][:]  # (T, H, W)
        missing_mask = f['missing_mask'][:]  # (T, H, W)
        fill_mask = f['fill_mask'][:]  # (T, H, W)
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        timestamps = f['timestamps'][:]

        # 读取属性
        attrs = dict(f.attrs)

    T, H, W = sst_data.shape
    print(f"Data shape: ({T}, {H}, {W})")

    # 统计滤波前的数据范围
    valid_data = sst_data[missing_mask == 0]
    before_stats = {
        'mean': float(np.nanmean(valid_data)),
        'std': float(np.nanstd(valid_data)),
        'min': float(np.nanmin(valid_data)),
        'max': float(np.nanmax(valid_data)),
    }
    print(f"Before filtering - Mean: {before_stats['mean']:.2f}K, Std: {before_stats['std']:.3f}K")

    # 准备并行任务
    tasks = []
    for t in range(T):
        tasks.append((t, sst_data[t], missing_mask[t], method, params))

    # 并行处理
    filtered_sst_data = np.zeros_like(sst_data)

    print(f"\nFiltering with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(filter_single_frame, task): task[0] for task in tasks}

        with tqdm(total=T, desc="Filtering") as pbar:
            for future in as_completed(futures):
                frame_idx, filtered_frame = future.result()
                filtered_sst_data[frame_idx] = filtered_frame
                pbar.update(1)

    # 统计滤波后的数据
    valid_filtered = filtered_sst_data[missing_mask == 0]
    after_stats = {
        'mean': float(np.nanmean(valid_filtered)),
        'std': float(np.nanstd(valid_filtered)),
        'min': float(np.nanmin(valid_filtered)),
        'max': float(np.nanmax(valid_filtered)),
    }
    print(f"After filtering - Mean: {after_stats['mean']:.2f}K, Std: {after_stats['std']:.3f}K")

    # 计算滤波影响
    diff = filtered_sst_data - sst_data
    valid_diff = diff[missing_mask == 0]
    diff_stats = {
        'mean_abs_change': float(np.mean(np.abs(valid_diff))),
        'max_abs_change': float(np.max(np.abs(valid_diff))),
        'rmse': float(np.sqrt(np.mean(valid_diff**2))),
    }
    print(f"Filter effect - Mean abs change: {diff_stats['mean_abs_change']:.4f}K, RMSE: {diff_stats['rmse']:.4f}K")

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # 主数据
        f.create_dataset('sst_data', data=filtered_sst_data.astype(np.float32),
                        compression='gzip', compression_opts=4)
        f.create_dataset('missing_mask', data=missing_mask,
                        compression='gzip', compression_opts=4)
        f.create_dataset('fill_mask', data=fill_mask,
                        compression='gzip', compression_opts=4)

        # 坐标
        f.create_dataset('latitude', data=latitude)
        f.create_dataset('longitude', data=longitude)
        f.create_dataset('timestamps', data=timestamps)

        # 复制原有属性
        for key, value in attrs.items():
            f.attrs[key] = value

        # 添加滤波属性
        f.attrs['filter_method'] = method
        f.attrs['filter_params'] = json.dumps(params)
        f.attrs['filter_date'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        f.attrs['before_stats'] = json.dumps(before_stats)
        f.attrs['after_stats'] = json.dumps(after_stats)
        f.attrs['diff_stats'] = json.dumps(diff_stats)

    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"\nSaved to: {output_path}")
    print(f"Time: {elapsed:.1f}s, Size: {file_size:.1f}MB")

    return {
        'series_id': attrs.get('series_id', -1),
        'method': method,
        'params': params,
        'before_stats': before_stats,
        'after_stats': after_stats,
        'diff_stats': diff_stats,
        'elapsed_time': elapsed,
        'file_size_mb': file_size,
    }


# ============================================================
# Visualization & Testing
# ============================================================

def visualize_filter_comparison(input_path: Path, frame_idx: int = 0,
                                methods: List[str] = None,
                                output_path: Path = None):
    """
    可视化不同滤波方法的对比

    Args:
        input_path: 输入h5文件
        frame_idx: 帧索引
        methods: 要对比的滤波方法列表
        output_path: 输出图片路径
    """
    if methods is None:
        methods = ['gaussian', 'median', 'uniform']

    print(f"\nVisualizing filter comparison for frame {frame_idx}...")

    # 读取数据
    with h5py.File(input_path, 'r') as f:
        sst_frame = f['sst_data'][frame_idx]
        missing_mask = f['missing_mask'][frame_idx]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    # 应用各种滤波
    results = {'Original': sst_frame}
    params_list = {
        'gaussian': {'sigma': 1.5},
        'median': {'size': 3},
        'uniform': {'size': 3},
        'bilateral': {'sigma_space': 1.5, 'sigma_color': 2.0},
    }

    for method in methods:
        filter_func = get_filter_function(method)
        params = params_list.get(method, {})

        if method == 'gaussian':
            filtered = filter_func(sst_frame, missing_mask, **params)
        elif method in ['median', 'uniform']:
            filtered = filter_func(sst_frame, missing_mask, **params)
        elif method == 'bilateral':
            filtered = filter_func(sst_frame, missing_mask, **params)

        results[f'{method.capitalize()}'] = filtered

    # 绘图
    n_plots = len(results) + len(methods)  # 原图+滤波图+差值图
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    # 数据范围
    valid_mask = (missing_mask == 0) & (~np.isnan(sst_frame))
    vmin = np.percentile(sst_frame[valid_mask], 1)
    vmax = np.percentile(sst_frame[valid_mask], 99)

    plot_idx = 0

    # 绘制原图和滤波结果
    for name, data in results.items():
        ax = axes[plot_idx]

        # 屏蔽缺失值
        plot_data = data.copy()
        plot_data[missing_mask == 1] = np.nan

        im = ax.imshow(plot_data, vmin=vmin, vmax=vmax, cmap='RdYlBu_r', aspect='auto')
        ax.set_title(name, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='SST (K)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # 计算统计
        valid_data = data[valid_mask]
        stats_text = f"Mean: {np.mean(valid_data):.2f}K\nStd: {np.std(valid_data):.3f}K"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx += 1

    # 绘制差值图
    for method in methods:
        ax = axes[plot_idx]

        diff = results[f'{method.capitalize()}'] - results['Original']
        diff[missing_mask == 1] = np.nan

        diff_max = np.percentile(np.abs(diff[valid_mask]), 99)
        im = ax.imshow(diff, vmin=-diff_max, vmax=diff_max, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{method.capitalize()} - Original', fontsize=12)
        plt.colorbar(im, ax=ax, label='Diff (K)')

        # 统计
        valid_diff = diff[valid_mask]
        stats_text = f"MAE: {np.mean(np.abs(valid_diff)):.4f}K\nRMSE: {np.sqrt(np.mean(valid_diff**2)):.4f}K"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx += 1

    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path is None:
        output_path = OUTPUT_DIR / 'filter_comparison.png'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison to: {output_path}")


def run_test():
    """运行测试模式"""
    print("="*70)
    print("JAXA Low-Pass Filter - Test Mode")
    print("="*70)

    # 找到输入文件
    input_files = sorted(INPUT_DIR.glob('jaxa_weighted_series_*.h5'))

    if not input_files:
        print(f"No input files found in {INPUT_DIR}")
        print("Please run jaxa_temporal_weighted_filling.py first.")
        return

    print(f"Found {len(input_files)} input files")
    print(f"Using first file for testing: {input_files[0].name}")

    # 可视化对比
    visualize_filter_comparison(
        input_files[0],
        frame_idx=10,  # 使用第10帧
        methods=['gaussian', 'median', 'uniform'],
        output_path=OUTPUT_DIR / 'filter_comparison_test.png'
    )

    # 测试不同sigma值的高斯滤波
    print("\n" + "="*70)
    print("Testing different Gaussian sigma values...")
    print("="*70)

    with h5py.File(input_files[0], 'r') as f:
        sst_frame = f['sst_data'][10]
        missing_mask = f['missing_mask'][10]

    valid_mask = (missing_mask == 0) & (~np.isnan(sst_frame))
    original_std = np.std(sst_frame[valid_mask])

    print(f"\nOriginal Std: {original_std:.4f}K\n")
    print(f"{'Sigma':<10} {'Filtered Std':<15} {'MAE':<12} {'RMSE':<12}")
    print("-"*50)

    for sigma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        filtered = apply_gaussian_filter(sst_frame, missing_mask, sigma=sigma)
        filtered_std = np.std(filtered[valid_mask])
        diff = filtered - sst_frame
        mae = np.mean(np.abs(diff[valid_mask]))
        rmse = np.sqrt(np.mean(diff[valid_mask]**2))

        print(f"{sigma:<10} {filtered_std:<15.4f} {mae:<12.4f} {rmse:<12.4f}")

    print("\nTest completed!")
    print(f"Output: {OUTPUT_DIR}/filter_comparison_test.png")


def process_full_dataset(method: str = DEFAULT_METHOD,
                         params: Dict = None,
                         num_workers: int = DEFAULT_NUM_WORKERS):
    """处理完整数据集"""
    print("="*70)
    print("JAXA Low-Pass Filter - Full Processing")
    print("="*70)
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Method: {method}")
    print(f"Parameters: {params}")
    print(f"Workers: {num_workers}")

    # 找到输入文件
    input_files = sorted(INPUT_DIR.glob('jaxa_weighted_series_*.h5'))

    if not input_files:
        print(f"\nNo input files found in {INPUT_DIR}")
        print("Please run jaxa_temporal_weighted_filling.py first.")
        return

    print(f"\nFound {len(input_files)} input files")

    # 处理每个文件
    all_results = []
    for input_path in input_files:
        output_path = OUTPUT_DIR / f"jaxa_filtered_{input_path.stem.split('_')[-1]}.h5"

        try:
            result = process_series(input_path, output_path, method, params, num_workers)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            continue

    # 汇总
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    print(f"\n{'Series':<10} {'Method':<12} {'MAE(K)':<12} {'RMSE(K)':<12} {'Time(s)':<10}")
    print("-"*60)

    for r in all_results:
        print(f"#{r['series_id']:<9} {r['method']:<12} "
              f"{r['diff_stats']['mean_abs_change']:<12.4f} "
              f"{r['diff_stats']['rmse']:<12.4f} "
              f"{r['elapsed_time']:<10.1f}")

    # 保存统计
    stats_file = OUTPUT_DIR / 'filter_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'method': method,
            'params': params,
            'results': all_results,
            'creation_date': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }, f, indent=2)

    print(f"\nStatistics saved to: {stats_file}")
    print("\nAll processing completed!")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='JAXA SST Low-Pass Filter Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run test mode (visualization)
  python jaxa_lowpass_filter.py --mode test

  # Process all series with Gaussian filter
  python jaxa_lowpass_filter.py --mode full --method gaussian --sigma 1.5

  # Process single series
  python jaxa_lowpass_filter.py --mode single --series 0 --method median --size 3

Filter methods:
  gaussian  - Gaussian low-pass filter (smooth)
  median    - Median filter (robust to outliers)
  uniform   - Uniform/box filter (simple moving average)
  bilateral - Bilateral filter (edge-preserving smooth)
"""
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'full', 'single'],
        default='test',
        help='Execution mode'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['gaussian', 'median', 'uniform', 'bilateral'],
        default=DEFAULT_METHOD,
        help=f'Filter method (default: {DEFAULT_METHOD})'
    )

    parser.add_argument(
        '--sigma',
        type=float,
        default=DEFAULT_SIGMA,
        help=f'Sigma for Gaussian filter (default: {DEFAULT_SIGMA})'
    )

    parser.add_argument(
        '--size',
        type=int,
        default=DEFAULT_KERNEL_SIZE,
        help=f'Kernel size for median/uniform filter (default: {DEFAULT_KERNEL_SIZE})'
    )

    parser.add_argument(
        '--series',
        type=int,
        default=0,
        help='Series ID for single mode (default: 0)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f'Number of workers (default: {DEFAULT_NUM_WORKERS})'
    )

    args = parser.parse_args()

    # 构建参数
    if args.method == 'gaussian':
        params = {'sigma': args.sigma}
    elif args.method in ['median', 'uniform']:
        params = {'size': args.size}
    elif args.method == 'bilateral':
        params = {'sigma_space': args.sigma, 'sigma_color': 2.0}
    else:
        params = {}

    if args.mode == 'test':
        run_test()
    elif args.mode == 'full':
        process_full_dataset(args.method, params, args.workers)
    elif args.mode == 'single':
        input_path = INPUT_DIR / f'jaxa_weighted_series_{args.series:02d}.h5'
        output_path = OUTPUT_DIR / f'jaxa_filtered_{args.series:02d}.h5'

        if not input_path.exists():
            print(f"Error: {input_path} not found")
            sys.exit(1)

        process_series(input_path, output_path, args.method, params, args.workers)


if __name__ == '__main__':
    main()
