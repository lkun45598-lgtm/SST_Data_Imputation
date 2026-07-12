#!/usr/bin/env python3
"""
JAXA SST 逐时预处理脚本

对指定小时 h (0-23) 的 JAXA 逐时观测数据，执行与 h=0 完全相同的三阶段预处理：
  Stage 1: 时间加权填充（从原始 nc 中提取 hour=h 的帧，跨天序列）
  Stage 2: 高斯低通滤波
  Stage 3: 渐进式 KNN 空间填充

输出与现有 jaxa_knn_filled_XX.h5 格式完全一致，可直接用于模型推理。

用法:
    python hourly_preprocess.py --hour 1 --series 0 --workers 32
    python hourly_preprocess.py --hour 1 --series all --workers 64
"""

import argparse
import sys
import os
import time
import numpy as np
import h5py
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

JAXA_ROOT = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
OUTPUT_BASE = Path('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data')

# 各年度序列的起始日期 (与 h=0 数据对齐)
SERIES_START_DATES = {
    0: datetime(2017, 7, 6),
    1: datetime(2016, 7, 6),
    2: datetime(2021, 7, 5),
    3: datetime(2018, 7, 6),
    4: datetime(2019, 7, 7),
    5: datetime(2020, 7, 5),
    6: datetime(2022, 7, 5),
    7: datetime(2023, 7, 5),
    8: datetime(2024, 7, 4),
}

SERIES_NUM_DAYS = {
    0: 360, 1: 317, 2: 365, 3: 345, 4: 356,
    5: 363, 6: 365, 7: 365, 8: 267,
}

LOOKBACK_WINDOW = 48  # hours
DEFAULT_WORKERS = 32
GAUSSIAN_SIGMA = 1.5
KNN_K = 20
KNN_RADIUS = 20
KNN_POWER = 2
KNN_REBUILD_INTERVAL = 50


# ============================================================
# Stage 1: 时间加权填充
# ============================================================

def load_jaxa_frame(target_time):
    """加载单帧 JAXA 原始 nc 数据"""
    date_str = target_time.strftime('%Y%m')
    day_str = target_time.strftime('%d')
    file_str = target_time.strftime('%Y%m%d%H%M%S')
    file_path = JAXA_ROOT / date_str / day_str / f'{file_str}.nc'

    if not file_path.exists():
        return None, None, None

    try:
        ds = xr.open_dataset(file_path)
        sst = ds.sea_surface_temperature.values
        if len(sst.shape) == 3:
            sst = sst[0]
        lat = ds.lat.values
        lon = ds.lon.values
        ds.close()
        return sst, lat, lon
    except Exception:
        return None, None, None


def temporal_weighted_fill_frame(target_time, start_time, hour_offset):
    """
    对单帧做时间加权填充

    向前看 LOOKBACK_WINDOW 小时内同小时的历史帧,
    用 weight=1/dt 加权平均填充缺失像素。
    """
    target_sst, lat, lon = load_jaxa_frame(target_time)
    if target_sst is None:
        return None, None, None, None

    original_sst = target_sst.copy()
    filled_sst = target_sst.copy()

    missing_mask = np.isnan(target_sst)
    if not missing_mask.any():
        return filled_sst, original_sst, missing_mask, target_time

    # 收集历史帧 (向前看最多 LOOKBACK_WINDOW/24 天的同小时数据)
    max_lookback_days = LOOKBACK_WINDOW // 24 + 1
    history = {}
    for d in range(1, max_lookback_days + 1):
        hist_time = target_time - timedelta(days=d)
        if hist_time < start_time:
            break
        hist_sst, _, _ = load_jaxa_frame(hist_time)
        if hist_sst is not None:
            history[d] = hist_sst  # d = 天数距离

    if not history:
        return filled_sst, original_sst, missing_mask, target_time

    # 对每个缺失像素做加权填充
    missing_y, missing_x = np.where(missing_mask)
    for idx in range(len(missing_y)):
        y, x = missing_y[idx], missing_x[idx]
        weights = []
        values = []
        for d, hist_sst in history.items():
            if not np.isnan(hist_sst[y, x]):
                w = 1.0 / d
                weights.append(w)
                values.append(hist_sst[y, x])
        if weights:
            filled_sst[y, x] = sum(w * v for w, v in zip(weights, values)) / sum(weights)

    return filled_sst, original_sst, missing_mask, target_time


def stage1_temporal_fill(series_id, hour_offset, num_workers):
    """Stage 1: 对整个年度序列做时间加权填充"""
    start_date = SERIES_START_DATES[series_id]
    num_days = SERIES_NUM_DAYS[series_id]

    # 起始时间加上小时偏移
    start_time = start_date.replace(hour=hour_offset)

    print(f"\n  Stage 1: 时间加权填充")
    print(f"    起始: {start_time}, 共 {num_days} 天")

    sst_list = []
    missing_mask_list = []
    fill_mask_list = []
    timestamps = []
    lat, lon = None, None

    for day in tqdm(range(num_days), desc="    加权填充"):
        target_time = start_time + timedelta(days=day)
        filled_sst, original_sst, missing_before, t_time = \
            temporal_weighted_fill_frame(target_time, start_time, hour_offset)

        if filled_sst is None:
            continue

        # 获取坐标 (首次)
        if lat is None:
            _, lat, lon = load_jaxa_frame(target_time)

        missing_after = np.isnan(filled_sst)
        fill_mask = (~missing_after & missing_before).astype(np.uint8)  # 被填充的像素

        sst_list.append(filled_sst)
        missing_mask_list.append(missing_after.astype(np.uint8))
        fill_mask_list.append(fill_mask)
        timestamps.append(target_time.isoformat())

    sst_data = np.array(sst_list, dtype=np.float32)
    missing_masks = np.array(missing_mask_list, dtype=np.uint8)
    fill_masks = np.array(fill_mask_list, dtype=np.uint8)

    filled_rate = np.mean([m.mean() for m in missing_mask_list]) * 100 if missing_mask_list else 0

    print(f"    填充后缺失率: {filled_rate:.1f}%")
    print(f"    有效帧数: {len(sst_list)}")

    return sst_data, missing_masks, fill_masks, lat, lon, np.array(timestamps, dtype='S32')


# ============================================================
# Stage 2: 高斯低通滤波
# ============================================================

def gaussian_filter_frame(data, missing_mask, fill_mask, sigma=GAUSSIAN_SIGMA):
    """对单帧做高斯滤波。

    只对"时间填充像素"(fill_mask==1)写回滤波值；原始观测像素与缺失像素保持不变。
    高斯卷积的输入仍使用真实观测值，使填充区能借真值邻居平滑，但结果不覆盖观测。
    """
    valid_mask = ~np.isnan(data) & (missing_mask == 0)
    if valid_mask.sum() == 0:
        return data

    data_for_filter = data.copy()
    mean_val = np.nanmean(data)
    data_for_filter[~valid_mask] = mean_val

    filtered = ndimage.gaussian_filter(data_for_filter, sigma=sigma)
    # 只在填充像素处使用滤波值；观测/缺失保留原值(真值不被滤波)
    write = valid_mask & (fill_mask == 1)
    result = np.where(write, filtered, data)
    return result


def stage2_filter(sst_data, missing_masks, fill_masks):
    """Stage 2: 对所有帧做高斯低通滤波(仅平滑时间填充像素，保留原始观测)"""
    print(f"\n  Stage 2: 高斯低通滤波 (sigma={GAUSSIAN_SIGMA}, 仅填充区)")

    filtered_data = np.zeros_like(sst_data)
    for t in tqdm(range(len(sst_data)), desc="    滤波"):
        filtered_data[t] = gaussian_filter_frame(sst_data[t], missing_masks[t], fill_masks[t])

    return filtered_data


# ============================================================
# Stage 3: 渐进式 KNN 填充
# ============================================================

def compute_missing_density(missing_coords, radius=KNN_RADIUS):
    """计算缺失密度"""
    if len(missing_coords) == 0:
        return np.array([])
    tree = cKDTree(missing_coords)
    counts = tree.query_ball_point(missing_coords, r=radius, return_length=True)
    return np.array(counts)


def knn_fill_single_frame(sst_data, missing_mask, k=KNN_K, radius=KNN_RADIUS,
                           power=KNN_POWER, rebuild_interval=KNN_REBUILD_INTERVAL):
    """渐进式KNN填充单帧"""
    filled_sst = sst_data.copy()

    missing_y, missing_x = np.where(missing_mask == 1)
    n_missing = len(missing_y)

    if n_missing == 0:
        return filled_sst, 0

    missing_coords = np.column_stack([missing_y, missing_x])
    density = compute_missing_density(missing_coords, radius=radius)
    sort_idx = np.argsort(density)
    sorted_coords = missing_coords[sort_idx]

    filled_count = 0
    tree = None
    valid_coords = None
    valid_values = None
    newly_filled_coords = []
    newly_filled_values = []

    for i in range(len(sorted_coords)):
        y, x = sorted_coords[i]

        if tree is None or (i > 0 and i % rebuild_interval == 0):
            valid_mask = ~np.isnan(filled_sst)
            valid_y, valid_x = np.where(valid_mask)
            if len(valid_y) == 0:
                continue
            valid_coords = np.column_stack([valid_y, valid_x])
            valid_values = filled_sst[valid_y, valid_x]
            tree = cKDTree(valid_coords)
            newly_filled_coords = []
            newly_filled_values = []

        actual_k = min(k, len(valid_coords))
        distances, indices = tree.query(np.array([[y, x]]), k=actual_k)
        distances = distances.flatten()
        indices = indices.flatten()

        if newly_filled_coords:
            new_coords = np.array(newly_filled_coords)
            new_values = np.array(newly_filled_values)
            new_distances = np.sqrt(np.sum((new_coords - np.array([y, x])) ** 2, axis=1))
            all_distances = np.concatenate([distances, new_distances])
            all_values = np.concatenate([valid_values[indices], new_values])
            k_nearest_idx = np.argsort(all_distances)[:actual_k]
            distances = all_distances[k_nearest_idx]
            neighbor_values = all_values[k_nearest_idx]
        else:
            neighbor_values = valid_values[indices]

        epsilon = 1e-10
        weights = 1.0 / (distances ** power + epsilon)
        interpolated_value = np.sum(weights * neighbor_values) / np.sum(weights)

        filled_sst[y, x] = interpolated_value
        filled_count += 1
        newly_filled_coords.append([y, x])
        newly_filled_values.append(interpolated_value)

    return filled_sst, filled_count


def _knn_worker(args):
    frame_idx, sst_frame, missing_frame = args
    filled, count = knn_fill_single_frame(sst_frame, missing_frame)
    return frame_idx, filled, count


def stage3_knn_fill(sst_data, missing_masks, land_mask, num_workers):
    """Stage 3: 3D 因果渐进式 KNN 填充（替换原 2D 版本）"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from knn_fill_3d import progressive_knn_fill_3d

    print(f"\n  Stage 3: 3D因果KNN填充 (k={KNN_K}, window=7天, time_weight=0.9)")

    filled_data, total_filled = progressive_knn_fill_3d(
        sst_data, missing_masks, land_mask,
        k=KNN_K, radius=KNN_RADIUS, power=KNN_POWER,
        time_weight=0.9, space_weight=1.0,
        window_size=7, batch_size=500
    )

    return filled_data


# ============================================================
# 主流程
# ============================================================

def process_series(series_id, hour_offset, num_workers):
    """对一个年度序列的指定小时完整三阶段预处理"""
    start_total = time.time()

    output_dir = OUTPUT_BASE / f'h{hour_offset:02d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'jaxa_knn_filled_{series_id:02d}.h5'

    print(f"\n{'='*70}")
    print(f"处理 series_{series_id:02d}, hour={hour_offset:02d}")
    print(f"输出: {output_path}")
    print(f"{'='*70}")

    # Stage 1
    sst_data, missing_masks, fill_masks, lat, lon, timestamps = \
        stage1_temporal_fill(series_id, hour_offset, num_workers)

    if len(sst_data) == 0:
        print("  无有效帧，跳过")
        return

    # 计算陆地掩码
    land_mask = np.all(np.isnan(sst_data), axis=0).astype(np.uint8)

    # Stage 2 (仅平滑时间填充像素，保留原始观测真值)
    filtered_data = stage2_filter(sst_data, missing_masks, fill_masks)

    # Stage 3
    filled_data = stage3_knn_fill(filtered_data, missing_masks, land_mask, num_workers)

    # 计算 original_obs_mask
    original_obs_mask = ((fill_masks == 0) & (missing_masks == 0)).astype(np.uint8)
    original_missing_mask = missing_masks  # Stage 1 之后的缺失

    # 保存
    print(f"\n  保存: {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('sst_data', data=filled_data.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('land_mask', data=land_mask, compression='gzip', compression_opts=4)
        f.create_dataset('original_obs_mask', data=original_obs_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('temporal_fill_mask', data=fill_masks,
                         compression='gzip', compression_opts=4)
        f.create_dataset('original_missing_mask', data=original_missing_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('latitude', data=lat)
        f.create_dataset('longitude', data=lon)
        f.create_dataset('timestamps', data=timestamps)

        f.attrs['series_id'] = series_id
        f.attrs['hour_offset'] = hour_offset
        f.attrs['num_frames'] = len(filled_data)
        f.attrs['creation_date'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    elapsed = time.time() - start_total
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n  完成! 耗时: {elapsed:.0f}s, 文件: {file_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='JAXA SST 逐时预处理')
    parser.add_argument('--hour', type=int, required=True, help='目标小时 (0-23)')
    parser.add_argument('--series', type=str, default='0', help='序列ID (0-8) 或 "all"')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help='并行核心数')
    args = parser.parse_args()

    assert 0 <= args.hour <= 23, f"hour 必须在 0-23 之间，当前: {args.hour}"

    if args.series == 'all':
        series_ids = list(range(9))
    else:
        series_ids = [int(s) for s in args.series.split(',')]

    print("=" * 70)
    print(f"JAXA SST 逐时预处理 - hour={args.hour:02d}")
    print("=" * 70)
    print(f"  序列: {series_ids}")
    print(f"  并行核心: {args.workers}")
    print(f"  输出目录: {OUTPUT_BASE / f'h{args.hour:02d}'}")

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    for sid in series_ids:
        process_series(sid, args.hour, args.workers)

    print("\n全部完成!")


if __name__ == '__main__':
    main()
