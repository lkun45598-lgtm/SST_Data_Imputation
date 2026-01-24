#!/usr/bin/env python3
"""
JAXA 渐进式KNN填充处理器

方案：
1. 扫描所有缺失点
2. 计算每个缺失点附近半径r内的缺失率d
3. 按d排序（小到大），缺失少的先填充
4. 逐个填充，每次填充后更新数据，下一个点利用刚填充的值

作者: Claude Code
日期: 2026-01-20
"""

import numpy as np
import h5py
from pathlib import Path
from scipy.spatial import cKDTree
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def compute_missing_density(missing_coords, radius=20):
    """
    计算每个缺失点周围的缺失密度

    Args:
        missing_coords: (N, 2) 缺失点坐标
        radius: 搜索半径

    Returns:
        density: (N,) 每个点周围半径r内的缺失点数量
    """
    if len(missing_coords) == 0:
        return np.array([])

    # 构建KDTree
    tree = cKDTree(missing_coords)

    # 查询每个点周围半径r内的点数
    counts = tree.query_ball_point(missing_coords, r=radius, return_length=True)

    return np.array(counts)


def progressive_knn_fill_single_frame(sst_data, missing_mask, k=20, radius=20, power=2):
    """
    渐进式KNN填充单帧数据

    Args:
        sst_data: SST数据 (H, W)
        missing_mask: 缺失掩码 (H, W), 1=缺失
        k: KNN的K值
        radius: 计算缺失密度的半径
        power: 距离权重指数

    Returns:
        filled_sst: 填充后的数据
        filled_count: 填充的像素数
    """
    filled_sst = sst_data.copy()
    H, W = sst_data.shape

    # 1. 提取缺失点坐标
    missing_y, missing_x = np.where(missing_mask == 1)
    n_missing = len(missing_y)

    if n_missing == 0:
        return filled_sst, 0

    missing_coords = np.column_stack([missing_y, missing_x])

    # 2. 计算每个缺失点的缺失密度d
    density = compute_missing_density(missing_coords, radius=radius)

    # 3. 按密度排序（小到大，边缘的先填充）
    sort_idx = np.argsort(density)
    sorted_coords = missing_coords[sort_idx]

    # 4. 逐个填充
    filled_count = 0

    for i in range(len(sorted_coords)):
        y, x = sorted_coords[i]

        # 检查是否已经被填充（理论上不会，但保险起见）
        if not np.isnan(filled_sst[y, x]) and missing_mask[y, x] == 0:
            continue

        # 获取当前有效点
        valid_mask = ~np.isnan(filled_sst)
        valid_y, valid_x = np.where(valid_mask)
        n_valid = len(valid_y)

        if n_valid == 0:
            continue

        valid_coords = np.column_stack([valid_y, valid_x])
        valid_values = filled_sst[valid_y, valid_x]

        # 构建KDTree（每次都要重建，因为有效点在变化）
        tree = cKDTree(valid_coords)

        # 查询k个最近邻
        actual_k = min(k, n_valid)
        query_point = np.array([[y, x]])
        distances, indices = tree.query(query_point, k=actual_k)

        distances = distances.flatten()
        indices = indices.flatten()

        # 反距离加权
        epsilon = 1e-10
        weights = 1.0 / (distances ** power + epsilon)
        neighbor_values = valid_values[indices]

        # 加权平均
        interpolated_value = np.sum(weights * neighbor_values) / np.sum(weights)

        # 填充
        filled_sst[y, x] = interpolated_value
        filled_count += 1

    return filled_sst, filled_count


def progressive_knn_fill_single_frame_optimized(sst_data, missing_mask, k=20, radius=20, power=2, rebuild_interval=100):
    """
    优化版渐进式KNN填充 - 减少KDTree重建次数

    Args:
        sst_data: SST数据 (H, W)
        missing_mask: 缺失掩码 (H, W), 1=缺失
        k: KNN的K值
        radius: 计算缺失密度的半径
        power: 距离权重指数
        rebuild_interval: 每填充多少个点重建一次KDTree

    Returns:
        filled_sst: 填充后的数据
        filled_count: 填充的像素数
    """
    filled_sst = sst_data.copy()
    H, W = sst_data.shape

    # 1. 提取缺失点坐标
    missing_y, missing_x = np.where(missing_mask == 1)
    n_missing = len(missing_y)

    if n_missing == 0:
        return filled_sst, 0

    missing_coords = np.column_stack([missing_y, missing_x])

    # 2. 计算每个缺失点的缺失密度d
    density = compute_missing_density(missing_coords, radius=radius)

    # 3. 按密度排序（小到大，边缘的先填充）
    sort_idx = np.argsort(density)
    sorted_coords = missing_coords[sort_idx]

    # 4. 逐个填充
    filled_count = 0
    tree = None
    valid_coords = None
    valid_values = None

    # 记录新填充的点
    newly_filled_coords = []
    newly_filled_values = []

    for i in range(len(sorted_coords)):
        y, x = sorted_coords[i]

        # 每隔一定间隔或第一次时重建KDTree
        if tree is None or (i > 0 and i % rebuild_interval == 0):
            # 获取当前所有有效点
            valid_mask = ~np.isnan(filled_sst)
            valid_y, valid_x = np.where(valid_mask)
            n_valid = len(valid_y)

            if n_valid == 0:
                continue

            valid_coords = np.column_stack([valid_y, valid_x])
            valid_values = filled_sst[valid_y, valid_x]
            tree = cKDTree(valid_coords)
            newly_filled_coords = []
            newly_filled_values = []

        # 查询k个最近邻
        actual_k = min(k, len(valid_coords))
        query_point = np.array([[y, x]])
        distances, indices = tree.query(query_point, k=actual_k)

        distances = distances.flatten()
        indices = indices.flatten()

        # 检查是否有新填充的点更近
        # 计算到新填充点的距离
        if newly_filled_coords:
            new_coords = np.array(newly_filled_coords)
            new_values = np.array(newly_filled_values)
            new_distances = np.sqrt(np.sum((new_coords - np.array([y, x])) ** 2, axis=1))

            # 合并旧邻居和新填充点
            all_distances = np.concatenate([distances, new_distances])
            all_values = np.concatenate([valid_values[indices], new_values])

            # 取最近的k个
            k_nearest_idx = np.argsort(all_distances)[:actual_k]
            distances = all_distances[k_nearest_idx]
            neighbor_values = all_values[k_nearest_idx]
        else:
            neighbor_values = valid_values[indices]

        # 反距离加权
        epsilon = 1e-10
        weights = 1.0 / (distances ** power + epsilon)

        # 加权平均
        interpolated_value = np.sum(weights * neighbor_values) / np.sum(weights)

        # 填充
        filled_sst[y, x] = interpolated_value
        filled_count += 1

        # 记录新填充的点
        newly_filled_coords.append([y, x])
        newly_filled_values.append(interpolated_value)

    return filled_sst, filled_count


def process_single_frame_wrapper(args):
    """包装函数用于多进程"""
    frame_idx, sst_frame, missing_frame, k, radius, power = args

    filled_sst, filled_count = progressive_knn_fill_single_frame_optimized(
        sst_frame, missing_frame, k=k, radius=radius, power=power, rebuild_interval=50
    )

    return frame_idx, filled_sst, filled_count


def process_jaxa_series(input_h5_path, output_h5_path, k=20, radius=20, power=2, num_workers=32):
    """
    处理单个JAXA时间序列

    Args:
        input_h5_path: 输入h5文件路径
        output_h5_path: 输出h5文件路径
        k: KNN参数
        radius: 缺失密度计算半径
        power: 距离权重指数
        num_workers: 并行worker数
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_h5_path}")
    print(f"{'='*70}")

    start_time = time.time()

    # 读取数据
    with h5py.File(input_h5_path, 'r') as f:
        sst_data = f['sst_data'][:]  # (T, H, W)
        missing_mask = f['missing_mask'][:]  # (T, H, W)
        fill_mask = f['fill_mask'][:]  # (T, H, W)
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        timestamps = f['timestamps'][:]

        series_id = f.attrs['series_id']
        start_year = f.attrs['start_year']
        num_frames = f.attrs['num_frames']

    T, H, W = sst_data.shape
    print(f"\nData shape: ({T}, {H}, {W})")
    print(f"Series ID: {series_id}, Start Year: {start_year}")

    # 计算陆地掩码
    land_mask = np.all(np.isnan(sst_data), axis=0).astype(np.uint8)
    land_pixel_count = land_mask.sum()
    print(f"Land mask: {land_pixel_count} pixels ({land_pixel_count/(H*W)*100:.2f}%)")

    ocean_mask = 1 - land_mask
    original_missing_count = np.sum(missing_mask)
    total_pixels = T * H * W
    original_missing_rate = original_missing_count / total_pixels * 100
    print(f"Original missing rate: {original_missing_rate:.2f}%")

    print(f"\nStarting Progressive KNN interpolation with {num_workers} workers...")
    print(f"Parameters: k={k}, radius={radius}, power={power}")

    filled_sst_data = np.zeros_like(sst_data)
    total_filled = 0

    # 准备任务
    tasks = []
    for t in range(T):
        ocean_missing = missing_mask[t] * ocean_mask
        tasks.append((t, sst_data[t], ocean_missing, k, radius, power))

    # 并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_frame_wrapper, task): task[0]
                   for task in tasks}

        with tqdm(total=T, desc="Progressive KNN") as pbar:
            for future in futures:
                frame_idx, filled_frame, filled_count = future.result()
                filled_sst_data[frame_idx] = filled_frame
                total_filled += filled_count
                pbar.update(1)

    # 验证填充结果
    ocean_nan_count = np.sum(np.isnan(filled_sst_data) & (ocean_mask[np.newaxis, :, :] == 1))
    print(f"\nFilling completed!")
    print(f"Total pixels filled: {total_filled}")
    print(f"Remaining ocean NaN count: {ocean_nan_count}")

    # 如果还有NaN，用全局均值填充
    if ocean_nan_count > 0:
        print(f"Filling remaining {ocean_nan_count} ocean NaN with global mean...")
        global_mean = np.nanmean(filled_sst_data)
        for t in range(T):
            nan_ocean = np.isnan(filled_sst_data[t]) & (ocean_mask == 1)
            filled_sst_data[t][nan_ocean] = global_mean

    # 陆地保持NaN
    for t in range(T):
        filled_sst_data[t][land_mask == 1] = np.nan

    # 计算original_obs_mask
    original_obs_mask = ((fill_mask == 0) & (missing_mask == 0)).astype(np.uint8)

    # 保存结果
    print(f"\nSaving to: {output_h5_path}")
    Path(output_h5_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('sst_data', data=filled_sst_data.astype(np.float32),
                        compression='gzip', compression_opts=4)
        f.create_dataset('land_mask', data=land_mask,
                        compression='gzip', compression_opts=4)
        f.create_dataset('original_obs_mask', data=original_obs_mask,
                        compression='gzip', compression_opts=4)
        f.create_dataset('temporal_fill_mask', data=fill_mask,
                        compression='gzip', compression_opts=4)
        f.create_dataset('original_missing_mask', data=missing_mask,
                        compression='gzip', compression_opts=4)
        f.create_dataset('latitude', data=latitude)
        f.create_dataset('longitude', data=longitude)
        f.create_dataset('timestamps', data=timestamps)

        f.attrs['series_id'] = series_id
        f.attrs['start_year'] = start_year
        f.attrs['num_frames'] = num_frames
        f.attrs['knn_k'] = k
        f.attrs['knn_radius'] = radius
        f.attrs['knn_power'] = power
        f.attrs['knn_method'] = 'progressive'
        f.attrs['original_missing_rate'] = original_missing_rate
        f.attrs['pixels_filled_by_knn'] = int(total_filled)
        f.attrs['land_pixel_count'] = int(land_pixel_count)
        f.attrs['creation_date'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    elapsed = time.time() - start_time
    file_size = Path(output_h5_path).stat().st_size / 1024 / 1024

    print(f"\nCompleted!")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Land pixels: {land_pixel_count} ({land_pixel_count/(H*W)*100:.2f}%)")
    print(f"  Original obs mask ratio: {original_obs_mask.sum() / total_pixels * 100:.2f}%")

    return {
        'series_id': series_id,
        'original_missing_rate': original_missing_rate,
        'pixels_filled': total_filled,
        'elapsed_time': elapsed,
        'file_size_mb': file_size
    }


def main():
    """主函数"""
    input_dir = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filtered')
    output_dir = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled')

    # KNN参数
    k = 20
    radius = 20
    power = 2
    num_workers = 216  # 使用216核心

    print("="*70)
    print("JAXA Progressive KNN Fill Processor")
    print("="*70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"KNN parameters: k={k}, radius={radius}, power={power}")
    print(f"Method: Progressive (edge-to-center)")
    print(f"Workers: {num_workers}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for series_id in range(3):
        input_path = input_dir / f'jaxa_filtered_{series_id:02d}.h5'
        output_path = output_dir / f'jaxa_knn_filled_{series_id:02d}.h5'

        if input_path.exists():
            result = process_jaxa_series(
                str(input_path),
                str(output_path),
                k=k,
                radius=radius,
                power=power,
                num_workers=num_workers
            )
            results.append(result)
        else:
            print(f"Warning: {input_path} not found, skipping...")

    # 汇总
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"\n{'Series':<10} {'Orig Missing%':<15} {'Pixels Filled':<15} {'Time(s)':<10} {'Size(MB)':<10}")
    print("-"*60)

    total_filled = 0
    total_time = 0

    for r in results:
        print(f"#{r['series_id']:<9} {r['original_missing_rate']:<15.2f} {r['pixels_filled']:<15,} "
              f"{r['elapsed_time']:<10.1f} {r['file_size_mb']:<10.1f}")
        total_filled += r['pixels_filled']
        total_time += r['elapsed_time']

    print("-"*60)
    print(f"{'Total':<10} {'':<15} {total_filled:<15,} {total_time:<10.1f}")

    print("\nAll processing completed!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
