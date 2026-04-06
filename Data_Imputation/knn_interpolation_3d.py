"""
3D Iterative KNN Interpolation for SST Missing Value Imputation
在时空域进行3D KNN插值，考虑时间和空间两个维度
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import h5py
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def compute_local_missing_density_3d(missing_coords, all_missing_coords, radius=10):
    """
    计算每个缺失点附近距离r范围内的缺失密度（3D版本）

    Args:
        missing_coords: 缺失点坐标 (N, 3) - [time, y, x]
        all_missing_coords: 所有缺失点坐标 (M, 3) - [time, y, x]
        radius: 搜索半径

    Returns:
        density: 每个缺失点的局部缺失密度 (N,)
    """
    tree = cKDTree(all_missing_coords)
    # 查询半径范围内的点数
    density = tree.query_ball_point(missing_coords, radius, return_length=True)
    return density


def knn_interpolate_iterative_3d(sst_data, missing_mask, land_mask,
                                   k=50, radius=20, power=1,
                                   time_weight=1.0, space_weight=1.0):
    """
    3D迭代式KNN插值（时空联合插值）

    算法流程：
    1. 提取所有时空缺失点
    2. 计算每个缺失点附近的缺失密度（3D）
    3. 按缺失密度从小到大排序（优先填充容易的点）
    4. 逐个进行3D KNN插值，每次填充后更新数据

    Args:
        sst_data: 输入SST数据 (T, H, W)，缺失位置为NaN
        missing_mask: 缺失掩码，1=缺失，0=有效 (T, H, W)
        land_mask: 陆地掩码，1=陆地，0=海洋 (H, W)
        k: KNN中的K值
        radius: 计算局部缺失密度的半径
        power: 距离权重的指数
        time_weight: 时间维度的权重
        space_weight: 空间维度的权重

    Returns:
        filled_sst: 填充后的SST数据
        stats: 统计信息字典
    """
    print(f"Starting 3D iterative KNN interpolation...")
    print(f"Parameters: k={k}, radius={radius}, power={power}")
    print(f"Weights: time={time_weight}, space={space_weight}")

    # 复制数据，避免修改原始数据
    filled_sst = sst_data.copy()

    # 获取数据形状
    T, H, W = sst_data.shape
    print(f"Data shape: T={T}, H={H}, W={W}")

    # 创建3D坐标网格
    t_coords, y_coords, x_coords = np.meshgrid(
        np.arange(T), np.arange(H), np.arange(W), indexing='ij'
    )

    # 步骤1: 提取所有缺失点（只考虑海洋区域）
    # 扩展land_mask到3D
    land_mask_3d = np.broadcast_to(land_mask[np.newaxis, :, :], (T, H, W))
    ocean_mask_3d = (land_mask_3d == 0)
    missing_ocean_mask = (missing_mask == 1) & ocean_mask_3d

    missing_t, missing_y, missing_x = np.where(missing_ocean_mask)
    # 注意：对坐标进行加权，使时间和空间维度具有不同的权重
    missing_coords = np.column_stack([
        missing_t * time_weight,
        missing_y * space_weight,
        missing_x * space_weight
    ])
    n_missing = len(missing_coords)

    print(f"\nTotal missing ocean pixels (3D): {n_missing}")
    print(f"Total ocean pixels (3D): {ocean_mask_3d.sum()}")
    print(f"Missing ratio: {n_missing / ocean_mask_3d.sum() * 100:.2f}%")

    # 调试信息：检查初始数据中的NaN情况
    initial_nan_count = np.isnan(filled_sst[ocean_mask_3d]).sum()
    print(f"\nDebug: Initial NaN count in ocean area: {initial_nan_count}")
    print(f"Debug: Missing mask count in ocean: {(missing_mask & ocean_mask_3d).sum()}")

    # 步骤2: 计算每个缺失点的局部缺失密度（3D）
    print(f"\nComputing 3D local missing density (radius={radius})...")
    densities = compute_local_missing_density_3d(missing_coords, missing_coords, radius=radius)

    # 步骤3: 按密度排序（从小到大）
    sorted_indices = np.argsort(densities)
    sorted_missing_t = missing_t[sorted_indices]
    sorted_missing_y = missing_y[sorted_indices]
    sorted_missing_x = missing_x[sorted_indices]
    sorted_densities = densities[sorted_indices]

    print(f"Density range: [{sorted_densities.min()}, {sorted_densities.max()}]")
    print(f"Median density: {np.median(sorted_densities):.1f}")

    # 步骤4: 迭代填充
    print(f"\nStarting iterative filling...")
    start_time = time.time()

    # 创建工作掩码，用于追踪哪些点已填充
    is_filled = ~missing_ocean_mask.copy()  # 初始时，非缺失点已填充

    # 统计信息
    failed_count = 0
    filled_count = 0

    # 每隔一定间隔打印进度
    log_interval = max(1, n_missing // 20)

    for i in range(n_missing):
        t, y, x = sorted_missing_t[i], sorted_missing_y[i], sorted_missing_x[i]

        # 提取当前所有已有值的点（包括原始有效点和已填充点）
        valid_t, valid_y, valid_x = np.where(is_filled & ocean_mask_3d)

        if len(valid_t) == 0:
            # 没有有效点，无法插值
            failed_count += 1
            continue

        valid_coords = np.column_stack([
            valid_t * time_weight,
            valid_y * space_weight,
            valid_x * space_weight
        ])
        valid_values = filled_sst[valid_t, valid_y, valid_x]

        # 关键修复：过滤掉值为NaN的点
        valid_mask = ~np.isnan(valid_values)
        valid_coords = valid_coords[valid_mask]
        valid_values = valid_values[valid_mask]

        if len(valid_values) == 0:
            # 过滤后没有有效点，无法插值
            failed_count += 1
            continue

        # 计算到当前缺失点的3D距离（已经加权）
        current_coord = np.array([t * time_weight, y * space_weight, x * space_weight])
        distances = np.sqrt(np.sum((valid_coords - current_coord)**2, axis=1))

        # 找到K个最近邻
        if len(distances) <= k:
            # 如果有效点少于k个，使用所有点
            nearest_indices = np.arange(len(distances))
        else:
            # 找到k个最近的点
            nearest_indices = np.argpartition(distances, k)[:k]

        nearest_distances = distances[nearest_indices]
        nearest_values = valid_values[nearest_indices]

        # 反距离加权插值
        epsilon = 1e-10
        weights = 1.0 / (nearest_distances**power + epsilon)
        weights_sum = weights.sum()

        if weights_sum > 0:
            interpolated_value = np.sum(weights * nearest_values) / weights_sum
            filled_sst[t, y, x] = interpolated_value
            is_filled[t, y, x] = True
            filled_count += 1
        else:
            failed_count += 1

        # 打印进度
        if (i + 1) % log_interval == 0 or i == n_missing - 1:
            elapsed = time.time() - start_time
            progress = (i + 1) / n_missing * 100
            eta = elapsed / (i + 1) * (n_missing - i - 1)
            print(f"Progress: {i+1}/{n_missing} ({progress:.1f}%) | "
                  f"Filled: {filled_count} | Failed: {failed_count} | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    total_time = time.time() - start_time
    print(f"\n3D Interpolation completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successfully filled: {filled_count}/{n_missing} ({filled_count/n_missing*100:.2f}%)")
    print(f"Failed: {failed_count}/{n_missing}")

    # 调试信息：检查填充后还有多少NaN
    final_nan_count = np.isnan(filled_sst[ocean_mask_3d]).sum()
    print(f"\nDebug: Final NaN count in ocean area: {final_nan_count}")
    print(f"Debug: Successfully reduced NaN by: {initial_nan_count - final_nan_count}")

    # 统计信息
    stats = {
        'n_missing': n_missing,
        'filled_count': filled_count,
        'failed_count': failed_count,
        'total_time': total_time,
        'k': k,
        'radius': radius,
        'power': power,
        'time_weight': time_weight,
        'space_weight': space_weight
    }

    return filled_sst, stats


def visualize_results_3d(ground_truth_sst, filled_sst, missing_mask,
                          land_mask, latitude, longitude, dates, stats, save_path=None):
    """
    可视化3D结果 - 显示多个时间点的2D切片

    Args:
        ground_truth_sst: Ground truth SST (T, H, W)
        filled_sst: KNN填充后的SST (T, H, W)
        missing_mask: 缺失掩码 (T, H, W)
        land_mask: 陆地掩码 (H, W)
        latitude: 纬度数组
        longitude: 经度数组
        dates: 日期字符串列表
        stats: 统计信息
        save_path: 保存路径
    """
    T = ground_truth_sst.shape[0]

    # 创建图形：T行 x 4列
    fig, axes = plt.subplots(T, 4, figsize=(24, 6*T))

    # 设置经纬度范围
    extent = [longitude.min(), longitude.max(), latitude.min(), latitude.max()]

    # 计算SST的显示范围（只考虑海洋区域，使用开尔文单位）
    ocean_mask = (land_mask == 0)
    valid_sst = ground_truth_sst[:, ocean_mask]
    valid_sst = valid_sst[~np.isnan(valid_sst)]
    vmin, vmax = np.percentile(valid_sst, [1, 99])

    for t in range(T):
        date_str = dates[t]
        missing_ratio = (missing_mask[t] & ocean_mask).sum() / ocean_mask.sum() * 100

        # 1. Input SST (带缺失)
        input_with_missing = ground_truth_sst[t].copy()
        input_with_missing[missing_mask[t] == 1] = np.nan

        ax = axes[t, 0] if T > 1 else axes[0]
        im = ax.imshow(input_with_missing, cmap='jet', vmin=vmin, vmax=vmax,
                       extent=extent, origin='lower', aspect='auto')
        ax.set_title(f'Input SST (Day {t+1})\n{date_str}\n{missing_ratio:.1f}% Missing',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SST [K]', fontsize=9)

        # 2. Ground Truth SST
        ax = axes[t, 1] if T > 1 else axes[1]
        im = ax.imshow(ground_truth_sst[t], cmap='jet', vmin=vmin, vmax=vmax,
                       extent=extent, origin='lower', aspect='auto')
        ax.set_title(f'Ground Truth SST\n{date_str}\n(Complete, No Missing)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SST [K]', fontsize=9)

        # 3. 3D KNN插值重建结果
        ax = axes[t, 2] if T > 1 else axes[2]
        im = ax.imshow(filled_sst[t], cmap='jet', vmin=vmin, vmax=vmax,
                       extent=extent, origin='lower', aspect='auto')
        ax.set_title(f'3D KNN Reconstruction\n{date_str}\n(Model Output)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SST [K]', fontsize=9)

        # 4. 绝对误差
        error = np.abs(filled_sst[t] - ground_truth_sst[t])
        error[land_mask == 1] = np.nan

        # 计算误差统计
        mae = np.nanmean(error)
        rmse = np.sqrt(np.nanmean(error**2))
        max_error = np.nanmax(error)

        ax = axes[t, 3] if T > 1 else axes[3]
        im = ax.imshow(error, cmap='hot_r', vmin=0, vmax=0.5,
                       extent=extent, origin='lower', aspect='auto')
        ax.set_title(f'Absolute Error\n{date_str}\n|3D KNN - Ground Truth|',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Error [K]', fontsize=9)

        # 在子图上添加误差统计文本
        textstr = f'MAE: {mae:.3f}K\nRMSE: {rmse:.3f}K\nMax: {max_error:.3f}K'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 添加总体标题
    fig.suptitle(f'3D KNN Interpolation Results (Time Range: {dates[0]} to {dates[-1]})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.close()


def main():
    # ========== 开始计时 ==========
    program_start_time = time.time()
    print(f"\nProgram started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 数据文件路径
    h5_file_path = '/data/sst_data/sst_missing_value_imputation/processed_data/processed_sst_valid.h5'

    # 输出目录
    output_dir = '/home/yc/SST_gap_filling_project/knn_3d_results'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("3D KNN Interpolation for SST Missing Value Imputation")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    # 时间范围：2015-01-08后的一个星期（7天）
    # 2015-01-08的索引是860
    start_index = 860
    n_days = 7
    date_indices = list(range(start_index, start_index + n_days))

    print(f"\nTime range: 2015-01-08 to 2015-01-14 ({n_days} days)")
    print(f"Date indices: {date_indices}")

    # 3D KNN参数
    k = 15              # K近邻数量
    radius = 15         # 局部缺失密度计算半径
    power = 1           # 距离权重指数
    time_weight = 0.5   # 时间维度权重（降低时间权重，使空间相关性更重要）
    space_weight = 1.0  # 空间维度权重

    print(f"\n3D KNN Parameters:")
    print(f"  k (neighbors): {k}")
    print(f"  radius (density): {radius}")
    print(f"  power (distance weight): {power}")
    print(f"  time_weight: {time_weight}")
    print(f"  space_weight: {space_weight}")

    # 从h5文件读取数据
    print(f"\nLoading data from: {h5_file_path}")
    data_loading_start = time.time()
    with h5py.File(h5_file_path, 'r') as f:
        # 读取一周的数据
        ground_truth_sst = f['ground_truth_sst'][start_index:start_index+n_days]
        missing_mask = f['missing_mask'][start_index:start_index+n_days]
        effective_cloud_mask = f['effective_cloud_mask'][start_index:start_index+n_days]
        land_mask = f['land_mask'][:]
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        time_iso = [f['time_iso'][i].decode() if isinstance(f['time_iso'][i], bytes)
                    else f['time_iso'][i] for i in date_indices]
    data_loading_time = time.time() - data_loading_start

    print(f"\nData loaded (Time: {data_loading_time:.2f}s):")
    print(f"  Shape: {ground_truth_sst.shape}")
    print(f"  Latitude range: [{latitude.min():.2f}, {latitude.max():.2f}]")
    print(f"  Longitude range: [{longitude.min():.2f}, {longitude.max():.2f}]")
    print(f"  Time range: {time_iso[0]} to {time_iso[-1]}")

    # 提取日期字符串（只保留日期部分）
    dates = [t.split('T')[0] for t in time_iso]

    ocean_mask = (land_mask == 0)
    total_missing = effective_cloud_mask.sum()
    total_ocean = ocean_mask.sum() * n_days
    print(f"  Total missing pixels (3D): {total_missing}")
    print(f"  Total ocean pixels (3D): {total_ocean}")
    print(f"  Overall missing ratio: {total_missing / total_ocean * 100:.2f}%")

    # 创建输入SST（使用ground_truth_sst作为基础）
    input_sst = ground_truth_sst.copy()

    # 执行3D KNN插值
    print(f"\n{'-'*70}")
    filled_sst, stats = knn_interpolate_iterative_3d(
        sst_data=input_sst,
        missing_mask=effective_cloud_mask,
        land_mask=land_mask,
        k=k,
        radius=radius,
        power=power,
        time_weight=time_weight,
        space_weight=space_weight
    )
    print(f"{'-'*70}")

    # 计算整体误差统计
    error = np.abs(filled_sst - ground_truth_sst)
    ocean_mask_3d = np.broadcast_to(ocean_mask[np.newaxis, :, :], ground_truth_sst.shape)
    error[~ocean_mask_3d] = np.nan

    overall_mae = np.nanmean(error)
    overall_rmse = np.sqrt(np.nanmean(error**2))

    print(f"\nOverall Error Statistics (all {n_days} days):")
    print(f"  MAE: {overall_mae:.4f}K")
    print(f"  RMSE: {overall_rmse:.4f}K")
    print(f"  Max Error: {np.nanmax(error):.4f}K")

    # 逐天误差统计
    print(f"\nPer-day Error Statistics:")
    print(f"{'Date':<15} {'Missing%':<12} {'MAE(K)':<10} {'RMSE(K)':<10}")
    print("-" * 50)
    for t, date in enumerate(dates):
        error_t = error[t]
        missing_ratio = (effective_cloud_mask[t] & ocean_mask).sum() / ocean_mask.sum() * 100
        mae_t = np.nanmean(error_t)
        rmse_t = np.sqrt(np.nanmean(error_t**2))
        print(f"{date:<15} {missing_ratio:<12.2f} {mae_t:<10.4f} {rmse_t:<10.4f}")

    # 可视化
    print(f"\nCreating visualization...")
    visualization_start = time.time()
    save_path = os.path.join(output_dir, f'knn_3d_reconstruction_{dates[0]}_to_{dates[-1]}.png')
    visualize_results_3d(
        ground_truth_sst=ground_truth_sst,
        filled_sst=filled_sst,
        missing_mask=effective_cloud_mask,
        land_mask=land_mask,
        latitude=latitude,
        longitude=longitude,
        dates=dates,
        stats=stats,
        save_path=save_path
    )
    visualization_time = time.time() - visualization_start
    print(f"Visualization time: {visualization_time:.2f}s")

    # 保存填充后的数据
    print(f"\nSaving filled data...")
    saving_start = time.time()
    data_save_path = os.path.join(output_dir, f'knn_3d_filled_{dates[0]}_to_{dates[-1]}.npz')
    np.savez(data_save_path,
             filled_sst=filled_sst,
             ground_truth_sst=ground_truth_sst,
             missing_mask=effective_cloud_mask,
             land_mask=land_mask,
             latitude=latitude,
             longitude=longitude,
             dates=dates,
             overall_mae=overall_mae,
             overall_rmse=overall_rmse,
             stats=stats)
    saving_time = time.time() - saving_start
    print(f"Filled data saved to: {data_save_path}")
    print(f"Data saving time: {saving_time:.2f}s")

    # 保存汇总结果
    summary_path = os.path.join(output_dir, 'summary_3d.txt')
    with open(summary_path, 'w') as f:
        f.write("3D KNN Interpolation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Time Range: {dates[0]} to {dates[-1]} ({n_days} days)\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  k = {k}\n")
        f.write(f"  radius = {radius}\n")
        f.write(f"  power = {power}\n")
        f.write(f"  time_weight = {time_weight}\n")
        f.write(f"  space_weight = {space_weight}\n\n")
        f.write(f"Overall Statistics:\n")
        f.write(f"  Overall MAE: {overall_mae:.4f}K\n")
        f.write(f"  Overall RMSE: {overall_rmse:.4f}K\n")
        f.write(f"  Successfully filled: {stats['filled_count']}/{stats['n_missing']}\n\n")
        f.write(f"Time Statistics:\n")
        f.write(f"  Data loading time: {data_loading_time:.2f}s\n")
        f.write(f"  Interpolation time: {stats['total_time']:.2f}s\n")
        f.write(f"  Visualization time: {visualization_time:.2f}s\n")
        f.write(f"  Data saving time: {saving_time:.2f}s\n")
        f.write(f"  Total program time: {time.time() - program_start_time:.2f}s\n\n")
        f.write(f"{'Date':<15} {'Missing%':<12} {'MAE(K)':<10} {'RMSE(K)':<10}\n")
        f.write("-" * 50 + "\n")
        for t, date in enumerate(dates):
            error_t = error[t]
            missing_ratio = (effective_cloud_mask[t] & ocean_mask).sum() / ocean_mask.sum() * 100
            mae_t = np.nanmean(error_t)
            rmse_t = np.sqrt(np.nanmean(error_t**2))
            f.write(f"{date:<15} {missing_ratio:<12.2f} {mae_t:<10.4f} {rmse_t:<10.4f}\n")
    print(f"\nSummary saved to: {summary_path}")

    # ========== 程序总时间统计 ==========
    total_program_time = time.time() - program_start_time
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"Data loading:      {data_loading_time:>8.2f}s  ({data_loading_time/total_program_time*100:>5.1f}%)")
    print(f"3D KNN interpolation: {stats['total_time']:>8.2f}s  ({stats['total_time']/total_program_time*100:>5.1f}%)")
    print(f"Visualization:     {visualization_time:>8.2f}s  ({visualization_time/total_program_time*100:>5.1f}%)")
    print(f"Data saving:       {saving_time:>8.2f}s  ({saving_time/total_program_time*100:>5.1f}%)")
    print("-" * 70)
    print(f"TOTAL TIME:        {total_program_time:>8.2f}s")
    print("=" * 70)
    print(f"\nProgram ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n3D KNN interpolation completed successfully!")


if __name__ == '__main__':
    main()
