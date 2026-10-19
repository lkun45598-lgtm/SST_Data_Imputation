#!/usr/bin/env python3
"""
为 Linear 和 Cubic 插值补充完整的评估指标
包括：MAE, RMSE, Bias, Gradient MAE, Gradient Ratio
使用与主结果表相同的评估协议
⚠️ 重要：应用与 FNO 相同的 Gaussian 平滑（sigma=1.0，只在 mask 区域）
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter

# Add project paths
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "inference"))

from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
from losses.gradient_masks import central_gradient_magnitude

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

W = 30
SERIES_ID = 8
SEED = 0
NUM_SAMPLES = 20  # 与主结果表一致
GAUSSIAN_SIGMA = 1.0  # 与 gen_fig6_stats.py 一致

LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]

def gradmag(field, valid_ref_mask):
    """严格中心差分；中心或四个轴向邻点无效时不生成梯度样本。"""
    return central_gradient_magnitude(field, valid_ref_mask)[0]

def gauss_filter(sst, land_mask, sigma, fill_region=None):
    """
    Gaussian 平滑，与 gen_fig6_stats.py 完全相同
    fill_region 给定时只在该区(填充/masked区)写回滤波值，观测保留原值。
    """
    valid = ~np.isnan(sst) & (land_mask == 0)
    if valid.sum() == 0:
        return sst
    mean_v = np.nanmean(sst)
    fill = sst.copy()
    fill[~valid] = mean_v
    out = gaussian_filter(fill, sigma=sigma)
    write = valid if fill_region is None else (valid & (fill_region == 1))
    return np.where(write, out, np.where(valid, sst, np.nan))

def linear_interp_2d(sst, mask, land, obs_mask=None):
    """2D linear interpolation
    obs_mask: 如果提供，只使用真实观测点（不使用KNN-filled的点）
    """
    H, W = sst.shape
    ocean = (land == 0)

    # 有效观测点：必须排除 KNN-filled 的云遮挡区域
    if obs_mask is not None:
        valid = (~mask) & ocean & (~np.isnan(sst)) & (obs_mask > 0)
    else:
        valid = (~mask) & ocean & (~np.isnan(sst))

    if valid.sum() < 3:
        return sst.copy()  # 不足3个点，无法插值

    # 构建坐标网格
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    # 观测点的坐标和值
    points = np.column_stack([y_coords[valid], x_coords[valid]])
    values = sst[valid]

    # 需要插值的点
    interp_mask = mask & ocean
    if interp_mask.sum() == 0:
        return sst.copy()

    interp_points = np.column_stack([y_coords[interp_mask], x_coords[interp_mask]])

    # 执行插值
    try:
        interp_values = griddata(points, values, interp_points, method='linear', fill_value=np.nan)

        # 填充结果
        result = sst.copy()
        result[interp_mask] = interp_values

        # 对于 NaN 值（插值范围外），使用最近邻
        still_nan = interp_mask & np.isnan(result)
        if still_nan.sum() > 0:
            nearest_values = griddata(points, values,
                                     np.column_stack([y_coords[still_nan], x_coords[still_nan]]),
                                     method='nearest')
            result[still_nan] = nearest_values

        return result
    except Exception as e:
        print(f"Linear interpolation failed: {e}")
        return sst.copy()

def cubic_interp_2d(sst, mask, land, obs_mask=None):
    """2D cubic interpolation
    obs_mask: 如果提供，只使用真实观测点
    """
    H, W = sst.shape
    ocean = (land == 0)

    if obs_mask is not None:
        valid = (~mask) & ocean & (~np.isnan(sst)) & (obs_mask > 0)
    else:
        valid = (~mask) & ocean & (~np.isnan(sst))

    if valid.sum() < 10:  # cubic 需要更多点
        return linear_interp_2d(sst, mask, land, obs_mask)  # fallback to linear

    y_coords, x_coords = np.mgrid[0:H, 0:W]
    points = np.column_stack([y_coords[valid], x_coords[valid]])
    values = sst[valid]

    interp_mask = mask & ocean
    if interp_mask.sum() == 0:
        return sst.copy()

    interp_points = np.column_stack([y_coords[interp_mask], x_coords[interp_mask]])

    try:
        # 尝试 cubic
        interp_values = griddata(points, values, interp_points, method='cubic', fill_value=np.nan)

        result = sst.copy()
        result[interp_mask] = interp_values

        # NaN fallback to linear
        still_nan = interp_mask & np.isnan(result)
        if still_nan.sum() > 0:
            linear_values = griddata(points, values,
                                    np.column_stack([y_coords[still_nan], x_coords[still_nan]]),
                                    method='linear', fill_value=np.nan)
            result[still_nan] = linear_values

        # 还有 NaN 就用 nearest
        still_nan = interp_mask & np.isnan(result)
        if still_nan.sum() > 0:
            nearest_values = griddata(points, values,
                                     np.column_stack([y_coords[still_nan], x_coords[still_nan]]),
                                     method='nearest')
            result[still_nan] = nearest_values

        return result
    except Exception as e:
        print(f"Cubic interpolation failed: {e}, fallback to linear")
        return linear_interp_2d(sst, mask, land)

def main():
    print("=" * 80)
    print("Linear 和 Cubic 插值完整指标评估")
    print("  Series 8, 3 missingness levels, N=20 per level")
    print("=" * 80)

    # 加载数据
    h5_path = REPO / f'experiments/hourly_data/h12/jaxa_knn_filled_{SERIES_ID:02d}.h5'
    with h5py.File(h5_path, 'r') as f:
        sst_all = f['sst_data'][:].astype(np.float32)
        obs_all = f['original_obs_mask'][:].astype(np.uint8)
        land = f['land_mask'][:].astype(np.uint8)

    ocean = (land == 0)

    # 候选帧
    cand = [i for i in range(W-1, sst_all.shape[0]) if (obs_all[i] * ocean).sum() > 4000]

    # 云模板库
    h5_paths = [str(REPO / f'experiments/hourly_data/h12/jaxa_knn_filled_{i:02d}.h5') for i in [0,3,6]]
    bank = build_cloud_bank(h5_paths, max_donors=400, seed=7)

    results = []
    rng = np.random.default_rng(SEED)

    methods = {
        'linear_2d': linear_interp_2d,
        'cubic_2d': cubic_interp_2d,
    }

    for method_name, interp_func in methods.items():
        print(f"\n{'='*80}")
        print(f"评估方法: {method_name}")
        print('='*80)

        for level_name, ratio in LEVELS:
            print(f"\n  处理 {level_name} (target ratio={ratio})...")

            # 选择样本
            idxs = rng.choice(cand, size=min(NUM_SAMPLES, len(cand)), replace=False)
            gen = RealCloudMaskGenerator(bank, seed=int(ratio * 1000))

            for idx in idxs:
                elig = (obs_all[idx] * ocean).astype(np.float32)
                if elig.sum() < 2000:
                    continue

                # 生成 mask
                mask = gen.generate(elig, target_ratio=ratio).astype(bool)
                ev = (mask & elig.astype(bool))
                if ev.sum() < 200:
                    continue

                # 真值
                gt = sst_all[idx]

                # 原始观测mask
                obs_mask_frame = obs_all[idx]

                # ⚠️ 关键修正：创建输入时只保留真实观测点
                # 不使用 KNN-filled 的云遮挡区域
                input_sst = gt.copy()
                input_sst[obs_mask_frame == 0] = np.nan  # 把原始云遮挡区域设为 NaN
                input_sst[mask] = np.nan  # 把人工遮挡区域也设为 NaN

                # 执行插值（只使用真实观测点）
                pred = interp_func(input_sst, mask, land, obs_mask=obs_mask_frame)

                # ⚠️ 关键：应用与 FNO 相同的 Gaussian 平滑（只在 mask 区域）
                pred = gauss_filter(pred, land, GAUSSIAN_SIGMA, fill_region=(mask > 0))

                # 计算指标
                e = pred[ev] - gt[ev]
                mae = float(np.abs(e).mean())
                rmse = float(np.sqrt((e**2).mean()))
                bias = float(e.mean())

                # 梯度指标
                valid_ref = obs_mask_frame.astype(bool) & (land == 0)
                gtg = gradmag(gt, valid_ref)
                pg = gradmag(pred, valid_ref)
                vv = ev & np.isfinite(pg) & np.isfinite(gtg)

                if vv.sum() > 0:
                    grad_mae = float(np.abs(pg[vv] - gtg[vv]).mean())
                    grad_ratio = float(pg[vv].mean() / (gtg[vv].mean() + 1e-9))
                else:
                    grad_mae = np.nan
                    grad_ratio = np.nan

                results.append({
                    'method': method_name,
                    'level': level_name,
                    'idx': int(idx),
                    'mae': mae,
                    'rmse': rmse,
                    'bias': bias,
                    'grad_mae': grad_mae,
                    'grad_ratio': grad_ratio,
                    'n_eval': int(ev.sum()),
                    'actual_ratio': float(mask.sum() / ocean.sum()),
                })

    # 保存结果
    records = np.array(results, dtype=object)
    out_path = CACHE / 'linear_cubic_full_metrics.npz'
    np.savez_compressed(out_path, records=records)

    print(f"\n{'='*80}")
    print(f"已保存: {out_path}")
    print(f"  总记录数: {len(records)}")

    # 汇总表
    print(f"\n{'='*80}")
    print("汇总统计 (格式：主结果表)")
    print('='*80)
    print(f"\n{'Method':<12} {'30%':>10} {'55%':>10} {'75%':>10} {'RMSE':>10} {'Bias':>10} {'GradMAE':>10} {'GradRatio':>10}")
    print("-" * 90)

    for method in ['linear_2d', 'cubic_2d']:
        method_recs = [r for r in results if r['method'] == method]

        mae_low = np.mean([r['mae'] for r in method_recs if r['level'] == 'low'])
        mae_mid = np.mean([r['mae'] for r in method_recs if r['level'] == 'mid'])
        mae_high = np.mean([r['mae'] for r in method_recs if r['level'] == 'high'])

        rmse_all = np.mean([r['rmse'] for r in method_recs])
        bias_all = np.mean([r['bias'] for r in method_recs])
        grad_mae_all = np.mean([r['grad_mae'] for r in method_recs if not np.isnan(r['grad_mae'])])
        grad_ratio_all = np.mean([r['grad_ratio'] for r in method_recs if not np.isnan(r['grad_ratio'])])

        print(f"{method:<12} {mae_low:>10.3f} {mae_mid:>10.3f} {mae_high:>10.3f} "
              f"{rmse_all:>10.3f} {bias_all:>+10.3f} {grad_mae_all:>10.3f} {grad_ratio_all:>10.2f}")

    print('='*80)

if __name__ == '__main__':
    main()
