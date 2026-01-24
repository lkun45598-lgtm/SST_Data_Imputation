#!/usr/bin/env python3
"""
JAXA SST填充模型评估 - 基于有label的区域（模拟训练时的情况）

评估方案:
1. 输入30天都用KNN完整数据（与训练一致）
2. 只在原始观测区域内人工挖空（这里有真实Ground Truth）
3. 挖空位置用均值填充
4. 比较模型预测 vs KNN值（在挖空区域，KNN值=原始观测值）
5. 计算 VRMSE、MAE、RMSE 等指标

作者: Claude Code
日期: 2026-01-23
"""

import os
import sys
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

KNN_FILLED_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/evaluation_results')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_8years/best_model.pth'

WINDOW_SIZE = 30
GPU_ID = 4
NUM_TEST_SAMPLES = 30
SERIES_ID = 0
MASK_RATIO = 0.2
MIN_MASK_SIZE = 10
MAX_MASK_SIZE = 50
SEED = 42

# 高斯滤波参数
APPLY_GAUSSIAN_FILTER = True
GAUSSIAN_SIGMA = 1.0


# ============================================================================
# Square Mask Generator (与训练时一致)
# ============================================================================

class SquareMaskGenerator:
    """方形挖空生成器"""

    def __init__(self, mask_ratio=0.2, min_size=10, max_size=50, seed=None):
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_mask, target_ratio=None):
        """在valid_mask区域内生成方形挖空"""
        if target_ratio is None:
            target_ratio = self.mask_ratio

        H, W = valid_mask.shape
        artificial_mask = np.zeros((H, W), dtype=np.float32)

        valid_count = valid_mask.sum()
        if valid_count == 0:
            return artificial_mask

        target_masked = int(valid_count * target_ratio)
        current_masked = 0

        valid_y, valid_x = np.where(valid_mask == 1)
        if len(valid_y) == 0:
            return artificial_mask

        y_min, y_max = valid_y.min(), valid_y.max()
        x_min, x_max = valid_x.min(), valid_x.max()

        max_attempts = 1000
        attempts = 0

        while current_masked < target_masked and attempts < max_attempts:
            size = self.rng.integers(self.min_size, self.max_size + 1)

            if y_max - size < y_min or x_max - size < x_min:
                attempts += 1
                continue

            y_start = self.rng.integers(y_min, max(y_min + 1, y_max - size + 1))
            x_start = self.rng.integers(x_min, max(x_min + 1, x_max - size + 1))

            y_end = min(y_start + size, H)
            x_end = min(x_start + size, W)

            region = valid_mask[y_start:y_end, x_start:x_end].copy()
            new_masked = region.sum() - (artificial_mask[y_start:y_end, x_start:x_end] * region).sum()

            if new_masked > 0:
                artificial_mask[y_start:y_end, x_start:x_end] = np.where(
                    region == 1, 1.0, artificial_mask[y_start:y_end, x_start:x_end]
                )
                current_masked = (artificial_mask * valid_mask).sum()

            attempts += 1

        return artificial_mask


# ============================================================================
# Gaussian Filter
# ============================================================================

def apply_gaussian_filter_sst(sst_data, land_mask, sigma=1.0):
    """
    对SST数据应用高斯滤波

    Args:
        sst_data: [H, W] SST数据 (Kelvin)
        land_mask: [H, W] 陆地掩码 (1=陆地, 0=海洋)
        sigma: 高斯滤波的标准差

    Returns:
        filtered_sst: [H, W] 滤波后的SST数据
    """
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


# ============================================================================
# Metrics
# ============================================================================

def calculate_metrics(pred, gt, mask):
    """计算评估指标（只在mask区域）"""
    valid = mask > 0
    if valid.sum() == 0:
        return {'vrmse': np.nan, 'mae': np.nan, 'rmse': np.nan, 'max_error': np.nan}

    pred_valid = pred[valid]
    gt_valid = gt[valid]

    # 转为摄氏度
    pred_celsius = pred_valid - 273.15
    gt_celsius = gt_valid - 273.15

    mae = np.abs(pred_celsius - gt_celsius).mean()
    rmse = np.sqrt(((pred_celsius - gt_celsius) ** 2).mean())
    max_error = np.abs(pred_celsius - gt_celsius).max()

    gt_std = np.std(gt_celsius)
    vrmse = rmse / gt_std if gt_std > 0 else np.nan

    return {
        'vrmse': vrmse,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'gt_std': gt_std,
        'num_pixels': int(valid.sum())
    }


# ============================================================================
# Model
# ============================================================================

def load_model(model_path: str, device: torch.device):
    """加载模型"""
    print(f"加载模型: {model_path}")

    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6,
        cbam_reduction_ratio=16
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_mean = checkpoint.get('norm_mean', 299.9221)
    norm_std = checkpoint.get('norm_std', 2.6919)

    print(f"  ✓ 模型加载成功 (Epoch {checkpoint.get('epoch', 'N/A')})")
    print(f"  归一化参数: mean={norm_mean:.4f}K, std={norm_std:.4f}K")

    return model, norm_mean, norm_std


def inference_like_training(model, sst_knn_30days, original_obs_mask, artificial_mask,
                            original_missing_mask_30days, land_mask, norm_mean, norm_std, device):
    """
    模拟训练时的推理（与jaxa_finetune_dataset.py一致）

    Args:
        sst_knn_30days: [30, H, W] 30天KNN完整数据
        original_obs_mask: [H, W] 第30天的原始观测区域 (1=观测, 0=缺失)
        artificial_mask: [H, W] 人工挖空掩码 (1=挖空, 0=保留)
        original_missing_mask_30days: [30, H, W] 30天的原始缺失掩码
        land_mask: [H, W] 陆地掩码

    Returns:
        pred_sst: [H, W] 模型预测 (Kelvin)
    """
    H, W = sst_knn_30days.shape[1], sst_knn_30days.shape[2]

    # 构建输入SST序列（与训练时一致）
    sst_input = sst_knn_30days.copy()

    # 第30天：在人工挖空位置用均值填充
    sst_input[-1] = np.where(artificial_mask > 0, norm_mean, sst_input[-1])

    # 构建mask序列
    # 前29天：使用原始缺失掩码
    # 第30天：使用人工挖空掩码（只有人工挖空，不包含原始缺失，因为我们用的是KNN完整数据）
    mask_seq = original_missing_mask_30days.copy().astype(np.float32)
    mask_seq[-1] = artificial_mask  # 第30天只有人工挖空

    # 归一化
    sst_norm = (sst_input - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)

    # 转为tensor
    sst_tensor = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)

    # 反归一化
    pred_kelvin = pred.squeeze().cpu().numpy() * norm_std + norm_mean

    # Output Composition：观测区域保留原值，挖空区域用预测
    filled_sst = sst_knn_30days[-1].copy()  # 第30天的KNN完整数据
    filled_sst = np.where(artificial_mask > 0, pred_kelvin, filled_sst)

    return filled_sst


# ============================================================================
# Data Loading
# ============================================================================

def load_knn_data(h5_path):
    """加载KNN填充后的数据"""
    with h5py.File(h5_path, 'r') as f:
        sst_data = f['sst_data'][:]
        original_obs_mask = f['original_obs_mask'][:]  # 原始观测区域
        original_missing_mask = f['original_missing_mask'][:]  # 原始缺失掩码
        land_mask = f['land_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]
    return {
        'sst_data': sst_data,
        'original_obs_mask': original_obs_mask,
        'original_missing_mask': original_missing_mask,
        'land_mask': land_mask,
        'lat': lat, 'lon': lon, 'timestamps': timestamps
    }


# ============================================================================
# Visualization
# ============================================================================

def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


def visualize_evaluation(gt_sst, pred_sst, artificial_mask, original_obs_mask, land_mask,
                         lon, lat, timestamp, metrics, save_path):
    """
    可视化评估结果 - 4连图布局

    1. Input SST (挖空区域显示为浅蓝色)
    2. Ground Truth SST (完整KNN数据)
    3. Model Prediction (FNO-CBAM)
    4. Absolute Error (仅挖空区域)
    """
    # 创建网格坐标
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 转换为摄氏度
    gt_celsius = gt_sst - 273.15
    pred_celsius = pred_sst - 273.15

    # 计算绝对误差
    abs_error = np.abs(pred_celsius - gt_celsius)

    # 海洋mask
    ocean_mask = (land_mask == 0)

    # 缺失率（挖空占观测区域的比例）
    mask_ratio = artificial_mask.sum() / (original_obs_mask.sum() + 1e-8) * 100

    # 设置colormap和范围
    valid_data = gt_celsius[ocean_mask & ~np.isnan(gt_celsius)]
    vmin_sst = np.percentile(valid_data, 2) if len(valid_data) > 0 else 20
    vmax_sst = np.percentile(valid_data, 98) if len(valid_data) > 0 else 32
    cmap_sst = 'RdYlBu_r'
    cmap_error = 'hot_r'
    land_color = '#D2B48C'

    # 创建图形 - 4连图布局
    fig = plt.figure(figsize=(28, 7))
    gs = gridspec.GridSpec(1, 6, figure=fig,
                          width_ratios=[1, 1, 1, 0.08, 1, 0.08],
                          wspace=0.15, hspace=0.1,
                          left=0.04, right=0.98, top=0.85, bottom=0.15)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_error = fig.add_subplot(gs[0, 4])

    # 准备数据
    land_display = np.ma.masked_where(land_mask == 0, land_mask)
    gt_masked = np.ma.masked_where(land_mask > 0, gt_celsius)
    pred_masked = np.ma.masked_where(land_mask > 0, pred_celsius)

    # 输入数据（挖空区域设为NaN）
    input_display = gt_celsius.copy()
    input_display[artificial_mask > 0] = np.nan
    input_masked = np.ma.masked_where(land_mask > 0, input_display)

    # ===== 1. Input SST（挖空区域为浅蓝色）=====
    ax = ax_input
    ax.set_facecolor('skyblue')  # 挖空区域背景色
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    ax.pcolormesh(lon_grid, lat_grid, input_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title(f'Input SST\n({mask_ratio:.1f}% Masked)', fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)

    # ===== 2. Ground Truth =====
    ax = ax_gt
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    ax.pcolormesh(lon_grid, lat_grid, gt_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Ground Truth SST\n(Complete)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 3. Model Prediction =====
    ax = ax_pred
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    im2 = ax.pcolormesh(lon_grid, lat_grid, pred_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    title_suffix = f'\n(+Gaussian σ={GAUSSIAN_SIGMA})' if APPLY_GAUSSIAN_FILTER else ''
    ax.set_title(f'FNO-CBAM Prediction{title_suffix}', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # SST Colorbar
    cax2_container = fig.add_subplot(gs[0, 3])
    cax2_container.axis('off')
    cax2 = inset_axes(cax2_container, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax2_container.transAxes)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('SST (°C)', fontsize=11)

    # ===== 4. Absolute Error (仅挖空区域) =====
    ax = ax_error
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    error_display = abs_error.copy()
    error_display[artificial_mask == 0] = np.nan
    error_masked = np.ma.masked_where((land_mask > 0) | (artificial_mask == 0), error_display)

    im3 = ax.pcolormesh(lon_grid, lat_grid, error_masked,
                        cmap=cmap_error, vmin=0, vmax=0.5, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Absolute Error\n(Masked Region Only)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # 统计信息
    stats_text = f"MAE: {metrics['mae']:.3f}°C\nRMSE: {metrics['rmse']:.3f}°C\nMax: {metrics['max_error']:.3f}°C"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Error Colorbar
    cax3_container = fig.add_subplot(gs[0, 5])
    cax3_container.axis('off')
    cax3 = inset_axes(cax3_container, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax3_container.transAxes)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('|Error| (°C)', fontsize=11)

    # 日期标题
    fig.text(0.5, 0.98, f'Date: {timestamp[:10]}', ha='center', va='top',
             fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("JAXA SST填充模型评估 - 模拟训练时的情况")
    print("=" * 70)
    print("输入: 30天KNN完整数据")
    print("挖空: 只在原始观测区域内")
    print("Ground Truth: KNN数据 (在观测区域 = 原始观测值)")
    if APPLY_GAUSSIAN_FILTER:
        print(f"后处理: 高斯滤波 (sigma={GAUSSIAN_SIGMA})")
    print("=" * 70)

    np.random.seed(SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    model, norm_mean, norm_std = load_model(MODEL_PATH, device)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载KNN数据
    knn_path = KNN_FILLED_DIR / f'jaxa_knn_filled_{SERIES_ID:02d}.h5'
    print(f"\n加载数据: {knn_path}")
    knn_data = load_knn_data(knn_path)

    sst_knn = knn_data['sst_data']  # [T, H, W] KNN完整数据
    original_obs_mask_all = knn_data['original_obs_mask']  # [T, H, W] 原始观测区域
    original_missing_mask_all = knn_data['original_missing_mask']  # [T, H, W] 原始缺失掩码
    land_mask = knn_data['land_mask']  # [H, W]
    lat = knn_data['lat']
    lon = knn_data['lon']
    timestamps = knn_data['timestamps']

    num_frames = sst_knn.shape[0]
    print(f"总帧数: {num_frames}")

    # 挖空生成器
    mask_generator = SquareMaskGenerator(
        mask_ratio=MASK_RATIO,
        min_size=MIN_MASK_SIZE,
        max_size=MAX_MASK_SIZE,
        seed=SEED
    )

    # 测试索引
    start_idx = WINDOW_SIZE - 1
    indices = list(range(start_idx, min(start_idx + NUM_TEST_SAMPLES, num_frames)))

    print(f"测试样本: {len(indices)} 个 (从第{start_idx}天开始)")
    print(f"人工挖空比例: {MASK_RATIO*100:.0f}% (相对于原始观测区域)")

    all_metrics = []
    vis_interval = max(1, len(indices) // 10)

    print(f"\n开始评估...")
    for i, idx in enumerate(tqdm(indices, desc="评估中")):
        # 获取30天KNN数据
        sst_knn_30days = np.zeros((WINDOW_SIZE, *sst_knn.shape[1:]), dtype=np.float32)
        original_missing_mask_30days = np.zeros((WINDOW_SIZE, *sst_knn.shape[1:]), dtype=np.float32)

        for t in range(WINDOW_SIZE):
            src_idx = idx - (WINDOW_SIZE - 1) + t
            if src_idx < 0:
                src_idx = 0
            sst_knn_30days[t] = sst_knn[src_idx]
            original_missing_mask_30days[t] = original_missing_mask_all[src_idx]

        # 第30天的原始观测区域
        original_obs_mask = original_obs_mask_all[idx]

        # 排除陆地
        valid_for_mask = original_obs_mask * (1 - land_mask)

        # 如果观测区域太少，跳过
        if valid_for_mask.sum() < 1000:
            continue

        # 在原始观测区域内生成人工挖空
        artificial_mask = mask_generator.generate(valid_for_mask.astype(np.float32))

        # Ground Truth = 第30天的KNN数据（在原始观测区域 = 原始观测值）
        gt_sst = sst_knn_30days[-1].copy()

        # 模型推理（模拟训练时的情况）
        pred_sst = inference_like_training(
            model, sst_knn_30days, original_obs_mask, artificial_mask,
            original_missing_mask_30days, land_mask, norm_mean, norm_std, device
        )

        # 高斯滤波后处理
        if APPLY_GAUSSIAN_FILTER:
            pred_sst = apply_gaussian_filter_sst(pred_sst, land_mask, sigma=GAUSSIAN_SIGMA)

        # 计算指标 - 只在人工挖空区域（原始观测区域内）
        eval_mask = artificial_mask * original_obs_mask * (1 - land_mask)
        metrics = calculate_metrics(pred_sst, gt_sst, eval_mask)
        metrics['timestamp'] = timestamps[idx]
        metrics['obs_ratio'] = original_obs_mask.sum() / (land_mask == 0).sum() * 100
        all_metrics.append(metrics)

        # 可视化
        if i % vis_interval == 0:
            vis_path = OUTPUT_DIR / f'eval_{timestamps[idx].replace(":", "").replace("-", "")}.png'
            visualize_evaluation(
                gt_sst, pred_sst, artificial_mask, original_obs_mask, land_mask,
                lon, lat, timestamps[idx], metrics, vis_path
            )

    # 汇总统计
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)

    vrmse_list = [m['vrmse'] for m in all_metrics if not np.isnan(m['vrmse'])]
    mae_list = [m['mae'] for m in all_metrics if not np.isnan(m['mae'])]
    rmse_list = [m['rmse'] for m in all_metrics if not np.isnan(m['rmse'])]
    max_list = [m['max_error'] for m in all_metrics if not np.isnan(m['max_error'])]

    print(f"\n样本数: {len(vrmse_list)}")
    print(f"\n{'指标':<15} {'均值':>12} {'标准差':>12} {'最小':>12} {'最大':>12}")
    print("-" * 60)
    print(f"{'VRMSE':<15} {np.mean(vrmse_list):>12.4f} {np.std(vrmse_list):>12.4f} {np.min(vrmse_list):>12.4f} {np.max(vrmse_list):>12.4f}")
    print(f"{'MAE (°C)':<15} {np.mean(mae_list):>12.4f} {np.std(mae_list):>12.4f} {np.min(mae_list):>12.4f} {np.max(mae_list):>12.4f}")
    print(f"{'RMSE (°C)':<15} {np.mean(rmse_list):>12.4f} {np.std(rmse_list):>12.4f} {np.min(rmse_list):>12.4f} {np.max(rmse_list):>12.4f}")
    print(f"{'Max Error (°C)':<15} {np.mean(max_list):>12.4f} {np.std(max_list):>12.4f} {np.min(max_list):>12.4f} {np.max(max_list):>12.4f}")

    print(f"\n核心指标:")
    print(f"  ★ 平均 VRMSE = {np.mean(vrmse_list):.4f}")
    print(f"  ★ 平均 MAE   = {np.mean(mae_list):.4f} °C")
    print(f"  ★ 平均 RMSE  = {np.mean(rmse_list):.4f} °C")

    if np.mean(vrmse_list) < 1:
        print(f"\n✓ VRMSE < 1，模型优于'猜均值'基线")
    else:
        print(f"\n✗ VRMSE >= 1，模型需要改进")

    print(f"\n可视化结果保存在: {OUTPUT_DIR}")
    print("=" * 70)

    # 保存结果
    import json
    results = {
        'config': {
            'model_path': MODEL_PATH,
            'series_id': SERIES_ID,
            'num_samples': len(vrmse_list),
            'mask_ratio': MASK_RATIO,
            'window_size': WINDOW_SIZE,
            'gaussian_filter': APPLY_GAUSSIAN_FILTER,
            'gaussian_sigma': GAUSSIAN_SIGMA if APPLY_GAUSSIAN_FILTER else None,
        },
        'summary': {
            'vrmse_mean': float(np.mean(vrmse_list)),
            'vrmse_std': float(np.std(vrmse_list)),
            'mae_mean': float(np.mean(mae_list)),
            'mae_std': float(np.std(mae_list)),
            'rmse_mean': float(np.mean(rmse_list)),
            'rmse_std': float(np.std(rmse_list)),
            'max_error_mean': float(np.mean(max_list)),
            'max_error_std': float(np.std(max_list)),
        },
        'per_sample': all_metrics
    }

    with open(OUTPUT_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"详细结果保存在: {OUTPUT_DIR / 'evaluation_results.json'}")


if __name__ == '__main__':
    main()
