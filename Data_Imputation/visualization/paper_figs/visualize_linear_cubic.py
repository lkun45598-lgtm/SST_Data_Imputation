#!/usr/bin/env python3
"""
可视化 Linear 和 Cubic 插值的重建结果
检查是否存在伪影、不连续性等问题
"""
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata

# Add project paths
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "inference"))

from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator

matplotlib.use('Agg')

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def linear_interp_2d(sst, mask, land):
    """2D linear interpolation"""
    H, W = sst.shape
    ocean = (land == 0)

    valid = (~mask) & ocean & (~np.isnan(sst))

    if valid.sum() < 3:
        return sst.copy()

    y_coords, x_coords = np.mgrid[0:H, 0:W]
    points = np.column_stack([y_coords[valid], x_coords[valid]])
    values = sst[valid]

    interp_mask = mask & ocean
    if interp_mask.sum() == 0:
        return sst.copy()

    interp_points = np.column_stack([y_coords[interp_mask], x_coords[interp_mask]])

    try:
        interp_values = griddata(points, values, interp_points, method='linear', fill_value=np.nan)

        result = sst.copy()
        result[interp_mask] = interp_values

        # NaN fallback to nearest
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

def cubic_interp_2d(sst, mask, land):
    """2D cubic interpolation"""
    H, W = sst.shape
    ocean = (land == 0)

    valid = (~mask) & ocean & (~np.isnan(sst))

    if valid.sum() < 10:
        return linear_interp_2d(sst, mask, land)

    y_coords, x_coords = np.mgrid[0:H, 0:W]
    points = np.column_stack([y_coords[valid], x_coords[valid]])
    values = sst[valid]

    interp_mask = mask & ocean
    if interp_mask.sum() == 0:
        return sst.copy()

    interp_points = np.column_stack([y_coords[interp_mask], x_coords[interp_mask]])

    try:
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

        # Still NaN -> nearest
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
    print("Linear 和 Cubic 插值可视化")
    print("=" * 80)

    # 加载数据
    h5_path = REPO / 'experiments/hourly_data/h12/jaxa_knn_filled_08.h5'
    with h5py.File(h5_path, 'r') as f:
        sst_all = f['sst_data'][:].astype(np.float32)
        obs_all = f['original_obs_mask'][:].astype(np.uint8)
        land = f['land_mask'][:].astype(np.uint8)

    ocean = (land == 0)

    # 云模板库
    h5_paths = [str(REPO / f'experiments/hourly_data/h12/jaxa_knn_filled_{i:02d}.h5') for i in [0,3,6]]
    bank = build_cloud_bank(h5_paths, max_donors=400, seed=7)

    # 选择几个有代表性的样本
    test_cases = [
        (100, 0.30, "low_30%"),
        (150, 0.55, "mid_55%"),
        (200, 0.75, "high_75%"),
    ]

    for idx, ratio, label in test_cases:
        print(f"\n处理样本 {idx} ({label})...")

        gt = sst_all[idx]
        elig = (obs_all[idx] * ocean).astype(np.float32)

        if elig.sum() < 2000:
            print(f"  跳过（观测不足）")
            continue

        # 生成 mask
        gen = RealCloudMaskGenerator(bank, seed=int(ratio * 1000))
        mask = gen.generate(elig, target_ratio=ratio).astype(bool)
        eval_mask = (mask & elig.astype(bool))

        if eval_mask.sum() < 200:
            print(f"  跳过（eval区域太小）")
            continue

        print(f"  Eval pixels: {eval_mask.sum()}, Mask ratio: {mask.sum()/ocean.sum()*100:.1f}%")

        # 创建输入
        input_sst = gt.copy()
        input_sst[mask] = np.nan

        # 执行插值
        pred_linear = linear_interp_2d(input_sst, mask, land)
        pred_cubic = cubic_interp_2d(input_sst, mask, land)

        # 计算误差
        err_linear = pred_linear - gt
        err_cubic = pred_cubic - gt

        mae_linear = np.abs(err_linear[eval_mask]).mean()
        mae_cubic = np.abs(err_cubic[eval_mask]).mean()

        print(f"  Linear MAE: {mae_linear:.3f} K")
        print(f"  Cubic MAE: {mae_cubic:.3f} K")

        # 可视化
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        # 设置海洋 mask（用于显示）
        plot_mask = (land == 1)

        # Row 1: Ground Truth
        im0 = axes[0, 0].imshow(np.ma.masked_where(plot_mask, gt), cmap='RdYlBu_r', vmin=296, vmax=304)
        axes[0, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # Mask overlay
        mask_overlay = np.ma.masked_where(~mask | plot_mask, np.ones_like(gt))
        axes[0, 1].imshow(np.ma.masked_where(plot_mask, gt), cmap='RdYlBu_r', vmin=296, vmax=304)
        axes[0, 1].imshow(mask_overlay, cmap='gray', alpha=0.6)
        axes[0, 1].set_title(f'Artificial Mask ({mask.sum()/ocean.sum()*100:.1f}%)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Input (with NaN)
        input_plot = input_sst.copy()
        input_plot = np.ma.masked_where(plot_mask | np.isnan(input_plot), input_plot)
        im02 = axes[0, 2].imshow(input_plot, cmap='RdYlBu_r', vmin=296, vmax=304)
        axes[0, 2].set_title('Input (masked)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Eval region
        eval_overlay = np.ma.masked_where(~eval_mask | plot_mask, np.ones_like(gt))
        axes[0, 3].imshow(np.ma.masked_where(plot_mask, gt), cmap='RdYlBu_r', vmin=296, vmax=304)
        axes[0, 3].imshow(eval_overlay, cmap='Greens', alpha=0.5)
        axes[0, 3].set_title('Evaluation Region', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')

        # Row 2: Linear
        im10 = axes[1, 0].imshow(np.ma.masked_where(plot_mask, pred_linear), cmap='RdYlBu_r', vmin=296, vmax=304)
        axes[1, 0].set_title(f'Linear Prediction\nMAE={mae_linear:.3f}K', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Linear error
        err_plot = np.ma.masked_where(~eval_mask | plot_mask, err_linear)
        im11 = axes[1, 1].imshow(err_plot, cmap='RdBu_r', vmin=-2, vmax=2)
        axes[1, 1].set_title('Linear Error (in eval region)', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04, label='K')

        # Linear zoomed region (检查伪影)
        y_center, x_center = np.where(eval_mask)
        if len(y_center) > 0:
            y_c, x_c = int(y_center.mean()), int(x_center.mean())
            zoom_size = 50
            y0, y1 = max(0, y_c-zoom_size), min(gt.shape[0], y_c+zoom_size)
            x0, x1 = max(0, x_c-zoom_size), min(gt.shape[1], x_c+zoom_size)

            zoom_pred = pred_linear[y0:y1, x0:x1]
            zoom_mask = plot_mask[y0:y1, x0:x1]
            im12 = axes[1, 2].imshow(np.ma.masked_where(zoom_mask, zoom_pred), cmap='RdYlBu_r', vmin=296, vmax=304)
            axes[1, 2].set_title('Linear Zoom (检查伪影)', fontsize=12)
            axes[1, 2].axis('off')
            plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

            # Error histogram
            axes[1, 3].hist(err_linear[eval_mask].flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 3].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[1, 3].set_title('Linear Error Distribution', fontsize=12)
            axes[1, 3].set_xlabel('Error (K)')
            axes[1, 3].set_ylabel('Frequency')
            axes[1, 3].grid(alpha=0.3)

        # Row 3: Cubic
        im20 = axes[2, 0].imshow(np.ma.masked_where(plot_mask, pred_cubic), cmap='RdYlBu_r', vmin=296, vmax=304)
        axes[2, 0].set_title(f'Cubic Prediction\nMAE={mae_cubic:.3f}K', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')
        plt.colorbar(im20, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # Cubic error
        err_plot = np.ma.masked_where(~eval_mask | plot_mask, err_cubic)
        im21 = axes[2, 1].imshow(err_plot, cmap='RdBu_r', vmin=-2, vmax=2)
        axes[2, 1].set_title('Cubic Error (in eval region)', fontsize=12)
        axes[2, 1].axis('off')
        plt.colorbar(im21, ax=axes[2, 1], fraction=0.046, pad=0.04, label='K')

        # Cubic zoomed
        if len(y_center) > 0:
            zoom_pred = pred_cubic[y0:y1, x0:x1]
            im22 = axes[2, 2].imshow(np.ma.masked_where(zoom_mask, zoom_pred), cmap='RdYlBu_r', vmin=296, vmax=304)
            axes[2, 2].set_title('Cubic Zoom (检查过冲)', fontsize=12)
            axes[2, 2].axis('off')
            plt.colorbar(im22, ax=axes[2, 2], fraction=0.046, pad=0.04)

            # Error histogram
            axes[2, 3].hist(err_cubic[eval_mask].flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[2, 3].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[2, 3].set_title('Cubic Error Distribution', fontsize=12)
            axes[2, 3].set_xlabel('Error (K)')
            axes[2, 3].set_ylabel('Frequency')
            axes[2, 3].grid(alpha=0.3)

        plt.tight_layout()

        out_path = OUT_DIR / f"linear_cubic_vis_{label}_idx{idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  已保存: {out_path}")

    print("\n" + "=" * 80)
    print("可视化完成")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
