#!/usr/bin/env python3
"""
3D Causal Progressive KNN Interpolation for SST Missing Value Imputation

在时空域进行 3D KNN 插值，替换原有的逐帧 2D KNN。
核心特点：
  1. 因果约束：只用过去帧 (t' <= t)，不用未来帧
  2. 双层 KDTree：历史帧树（静态） + 当前帧树（渐进更新）
  3. 批量查询：利用 cKDTree 向量化查询，避免逐像素循环
  4. 渐进填充：按缺失密度升序，边缘先填中心后填
"""

import numpy as np
from scipy.spatial import cKDTree
import time
from tqdm import tqdm


def compute_missing_density_2d(missing_coords, radius=20):
    """计算每个缺失点在 2D 空间中的缺失密度"""
    if len(missing_coords) == 0:
        return np.array([])
    tree = cKDTree(missing_coords)
    counts = tree.query_ball_point(missing_coords, r=radius, return_length=True)
    return np.array(counts)


def progressive_knn_fill_3d(sst_data, missing_masks, land_mask,
                             k=20, radius=20, power=2,
                             time_weight=0.9, space_weight=1.0,
                             window_size=7, batch_size=500):
    """
    3D 因果渐进式 KNN 填充

    外层按时间顺序 t=0..T-1（因果），内层按 2D 缺失密度升序（渐进）。
    对每个缺失像素，在 7 天时间窗口内搜索时空邻居做 IDW 插值。

    Args:
        sst_data: (T, H, W) float32, Stage 2 输出，缺失处为 NaN
        missing_masks: (T, H, W) uint8, 1=缺失 0=有效
        land_mask: (H, W) uint8, 1=陆地 0=海洋
        k: KNN 近邻数
        radius: 2D 缺失密度计算半径
        power: IDW 距离权重指数
        time_weight: 时间维度缩放系数
        space_weight: 空间维度缩放系数
        window_size: 时间回溯窗口（天数）
        batch_size: 批量查询大小

    Returns:
        filled_sst: (T, H, W) float32, 海洋区域全部填充
        total_filled: int, 填充的像素总数
    """
    T, H, W = sst_data.shape
    filled_sst = sst_data.copy()
    ocean_mask = (land_mask == 0)
    total_filled = 0

    print(f"  3D KNN 参数: k={k}, window={window_size}天, "
          f"time_weight={time_weight}, batch={batch_size}")
    print(f"  数据: {T}帧, 海洋像素/帧={ocean_mask.sum()}")

    for t in tqdm(range(T), desc="    3D KNN"):
        # --- 当前帧缺失像素 ---
        ocean_missing = (missing_masks[t] == 1) & ocean_mask
        missing_y, missing_x = np.where(ocean_missing)
        n_missing = len(missing_y)

        if n_missing == 0:
            continue

        # --- 2D 密度排序（渐进填充顺序） ---
        missing_coords_2d = np.column_stack([missing_y, missing_x])
        density = compute_missing_density_2d(missing_coords_2d, radius=radius)
        sort_idx = np.argsort(density)
        sorted_y = missing_y[sort_idx]
        sorted_x = missing_x[sort_idx]

        # --- Tier 1: 历史帧 KDTree（t-6..t-1，已完全填充） ---
        t_start = max(0, t - window_size + 1)
        past_coords_list = []
        past_values_list = []

        for t_past in range(t_start, t):
            # 过去帧已被填充，只取海洋非NaN像素
            valid = ~np.isnan(filled_sst[t_past]) & ocean_mask
            vy, vx = np.where(valid)
            if len(vy) == 0:
                continue
            t_offset = t - t_past  # 1..6
            past_coords_list.append(np.column_stack([
                np.full(len(vy), t_offset * time_weight),
                vy.astype(np.float64) * space_weight,
                vx.astype(np.float64) * space_weight
            ]))
            past_values_list.append(filled_sst[t_past, vy, vx])

        past_tree = None
        past_values = None
        if past_coords_list:
            past_coords = np.vstack(past_coords_list)
            past_values = np.concatenate(past_values_list)
            past_tree = cKDTree(past_coords)

        # --- Tier 2: 当前帧树（渐进更新） ---
        curr_tree = None
        curr_coords = None
        curr_values = None
        rebuild_needed = True

        filled_this_frame = 0

        for b_start in range(0, n_missing, batch_size):
            b_end = min(b_start + batch_size, n_missing)
            b_y = sorted_y[b_start:b_end]
            b_x = sorted_x[b_start:b_end]
            b_size = len(b_y)

            # 每个 batch 重建当前帧树（包含原始有效 + 已填充像素）
            if rebuild_needed:
                valid_curr = ~np.isnan(filled_sst[t]) & ocean_mask
                vy, vx = np.where(valid_curr)
                if len(vy) > 0:
                    curr_coords = np.column_stack([
                        np.zeros(len(vy)),
                        vy.astype(np.float64) * space_weight,
                        vx.astype(np.float64) * space_weight
                    ])
                    curr_values = filled_sst[t, vy, vx]
                    curr_tree = cKDTree(curr_coords)
                else:
                    curr_tree = None
                    curr_values = None
                rebuild_needed = False

            # 查询坐标（当前帧，t_offset=0）
            query_coords = np.column_stack([
                np.zeros(b_size),
                b_y.astype(np.float64) * space_weight,
                b_x.astype(np.float64) * space_weight
            ])

            # --- 查询 Tier 1（历史帧） ---
            if past_tree is not None and len(past_values) >= 1:
                k_past = min(k, len(past_values))
                d1, i1 = past_tree.query(query_coords, k=k_past)
                if k_past == 1:
                    d1 = d1.reshape(-1, 1)
                    i1 = i1.reshape(-1, 1)
                # 填充到 (b_size, k) 大小
                past_d = np.full((b_size, k), np.inf)
                past_v = np.zeros((b_size, k))
                past_d[:, :k_past] = d1
                past_v[:, :k_past] = past_values[i1]
            else:
                past_d = np.full((b_size, k), np.inf)
                past_v = np.zeros((b_size, k))

            # --- 查询 Tier 2（当前帧） ---
            if curr_tree is not None and curr_values is not None and len(curr_values) >= 1:
                k_curr = min(k, len(curr_values))
                d2, i2 = curr_tree.query(query_coords, k=k_curr)
                if k_curr == 1:
                    d2 = d2.reshape(-1, 1)
                    i2 = i2.reshape(-1, 1)
                curr_d = np.full((b_size, k), np.inf)
                curr_v = np.zeros((b_size, k))
                curr_d[:, :k_curr] = d2
                curr_v[:, :k_curr] = curr_values[i2]
            else:
                curr_d = np.full((b_size, k), np.inf)
                curr_v = np.zeros((b_size, k))

            # --- 合并两层结果，取 k 个最近的 ---
            all_d = np.concatenate([past_d, curr_d], axis=1)  # (b_size, 2k)
            all_v = np.concatenate([past_v, curr_v], axis=1)

            # 按距离排序取前 k
            sort_order = np.argsort(all_d, axis=1)[:, :k]
            row_idx = np.arange(b_size)[:, np.newaxis]
            best_d = all_d[row_idx, sort_order]  # (b_size, k)
            best_v = all_v[row_idx, sort_order]

            # --- IDW 插值（向量化） ---
            epsilon = 1e-10
            weights = 1.0 / (best_d ** power + epsilon)
            # inf 距离对应权重为 0
            valid_mask = np.isfinite(best_d)
            weights = weights * valid_mask
            w_sum = weights.sum(axis=1)

            has_neighbor = w_sum > 0
            interpolated = np.where(
                has_neighbor,
                np.sum(weights * best_v, axis=1) / np.where(has_neighbor, w_sum, 1.0),
                np.nan
            )

            # --- 写入填充值 ---
            fill_mask = has_neighbor
            filled_sst[t, b_y[fill_mask], b_x[fill_mask]] = interpolated[fill_mask]
            filled_this_frame += fill_mask.sum()

            # 标记需要重建当前帧树
            if fill_mask.any():
                rebuild_needed = True

        total_filled += filled_this_frame

    # --- 兜底：残余 NaN 用全局海洋均值填充 ---
    ocean_nan = np.isnan(filled_sst) & ocean_mask[np.newaxis, :, :]
    remaining = ocean_nan.sum()
    if remaining > 0:
        global_mean = np.nanmean(filled_sst[ocean_mask[np.newaxis, :, :].repeat(T, axis=0) & ~np.isnan(filled_sst)])
        filled_sst[ocean_nan] = global_mean
        print(f"    兜底填充: {remaining} 像素 (全局均值 {global_mean:.2f}K)")

    # 陆地保持 NaN
    land_3d = np.broadcast_to(land_mask[np.newaxis, :, :], (T, H, W))
    filled_sst[land_3d == 1] = np.nan

    print(f"    总填充: {total_filled:,} 像素")

    return filled_sst, total_filled


if __name__ == '__main__':
    # 快速测试：取 h=1 series_00 的 Stage 2 输出跑 3D KNN
    import h5py
    from pathlib import Path

    # 使用 weighted_aligned 数据（Stage 1 输出）模拟测试
    test_path = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_weighted_aligned/jaxa_weighted_series_00.h5')

    print("=" * 60)
    print("3D KNN 快速测试（前 10 帧）")
    print("=" * 60)

    with h5py.File(test_path, 'r') as f:
        sst = f['sst_data'][:10].astype(np.float32)
        masks = f['missing_mask'][:10].astype(np.uint8)
        land = np.all(np.isnan(sst), axis=0).astype(np.uint8)

    print(f"数据: {sst.shape}, 缺失率: {masks.mean()*100:.1f}%")

    start = time.time()
    filled, count = progressive_knn_fill_3d(
        sst, masks, land,
        k=20, window_size=7, batch_size=500
    )
    elapsed = time.time() - start

    ocean = (land == 0)
    remaining_nan = np.isnan(filled[:, ocean]).sum()
    print(f"\n耗时: {elapsed:.1f}s")
    print(f"填充: {count:,} 像素")
    print(f"残余 NaN: {remaining_nan}")
    print("测试完成!")
