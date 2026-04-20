#!/usr/bin/env python3
"""
H=1 推理 - 使用原始缺失JAXA数据
直接从原始nc文件读取，保留原始缺失模式，显示模型在真实缺失条件下的表现

输出路径: experiments/jaxa_finetune_h01/inference_original_vis/sample_XX.png
"""

import sys
import os
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

# ============================================================
# 路径配置
# ============================================================
MODEL_PATH = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h01/best_model.pth'
JAXA_ROOT  = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
KNN_DATA_DIR = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01'
OUTPUT_DIR = Path('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h01/inference_original_vis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 用KNN数据的坐标和统计信息
COORDS_FILE = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01/jaxa_knn_filled_08.h5'

# H=1 的日期序列 (与preprocessing对齐)
SERIES_DATES = {
    1: (datetime(2016, 7, 6), 317),   # series_1: 317天
}

# ============================================================
# 归一化参数
# ============================================================
NORM_MEAN = 300.0  # Kelvin
NORM_STD = 5.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_matplotlib():
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1.5, labelsize=14)
    plt.rc('lines', linewidth=1.5)
    plt.rcParams.update({
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.pad': 5, 'ytick.major.pad': 5,
    })


def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


def load_jaxa_frame(target_time):
    """加载单帧原始JAXA nc数据"""
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
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None, None, None


def get_coords_from_knn(knn_file):
    """从KNN数据获取坐标和陆地掩码"""
    import h5py
    with h5py.File(knn_file, 'r') as f:
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        land_mask = f['land_mask'][:].astype(np.float32)
    return lat, lon, land_mask


def temporal_weighted_fill_frame(target_time, start_time, hour_offset, lookback_days=2):
    """
    对单帧做时间加权填充（简化版）
    用向前看最多lookback_days天的同小时数据填充缺失
    """
    target_sst, lat, lon = load_jaxa_frame(target_time)
    if target_sst is None:
        return None, None, None, None

    filled_sst = target_sst.copy()
    missing_mask = np.isnan(target_sst)

    if not missing_mask.any():
        return filled_sst, target_sst.copy(), missing_mask, target_time

    # 收集历史帧
    history = {}
    for d in range(1, lookback_days + 1):
        hist_time = target_time - timedelta(days=d)
        if hist_time < start_time:
            break
        hist_sst, _, _ = load_jaxa_frame(hist_time)
        if hist_sst is not None:
            history[d] = hist_sst

    if not history:
        return filled_sst, target_sst.copy(), missing_mask, target_time

    # 加权填充
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

    return filled_sst, target_sst.copy(), missing_mask, target_time


def load_model(model_path, device):
    """加载模型"""
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80, modes2=64,
        width=64, depth=6,
        cbam_reduction_ratio=16
    )
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model


def inference_sequence(model, sst_seq, mask_seq, device):
    """
    模型推理
    Args:
        sst_seq: (30, H, W) 已归一化
        mask_seq: (30, H, W)
    Returns:
        pred: (1, H, W) 已归一化
    """
    sst_input = torch.from_numpy(sst_seq[np.newaxis, :, :, :]).float().to(device)
    mask_input = torch.from_numpy(mask_seq[np.newaxis, :, :, :]).float().to(device)

    with torch.no_grad():
        pred = model(sst_input, mask_input)

    return pred[0, 0].cpu().numpy()


def draw_four_panel_original(input_sst_norm, original_missing_mask, ground_truth_norm,
                             pred_norm, land_mask, lon_coords, lat_coords,
                             norm_mean, norm_std, sample_idx, save_path):
    """
    4连图：Input | Ground Truth | Prediction | Error
    用于原始缺失数据的可视化

    Args:
        input_sst_norm:        (H, W) 第30天输入（原始缺失处为NaN）
        original_missing_mask: (H, W) 1=原始缺失, 0=有观测
        ground_truth_norm:     (H, W) 填充后的完整真值
        pred_norm:             (H, W) 模型预测
        land_mask:             (H, W) 1=陆地
        lon_coords, lat_coords: 坐标数组
        norm_mean, norm_std:   归一化参数
        sample_idx:            样本序号
        save_path:             保存路径
    """
    to_celsius = lambda x: x * norm_std + norm_mean - 273.15

    input_celsius = to_celsius(np.nan_to_num(input_sst_norm, nan=norm_mean))
    gt_celsius = to_celsius(ground_truth_norm)
    pred_celsius = to_celsius(pred_norm)

    abs_error = np.abs(pred_celsius - gt_celsius)

    # 只在原始缺失的海洋区域计算误差
    ocean_mask = (land_mask == 0)
    missing_ocean = (original_missing_mask == 1) & ocean_mask

    mae = float(abs_error[missing_ocean].mean()) if missing_ocean.sum() > 0 else 0.0
    rmse = float(np.sqrt((abs_error[missing_ocean]**2).mean())) if missing_ocean.sum() > 0 else 0.0
    max_err = float(abs_error[missing_ocean].max()) if missing_ocean.sum() > 0 else 0.0

    # ========== 4连图 ==========
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, left=0.08, right=0.95, top=0.93, bottom=0.08,
                           wspace=0.3, hspace=0.35)

    # 颜色映射
    cmap_sst = plt.cm.RdYlBu_r
    cmap_err = plt.cm.YlOrRd

    lon_formatter = FuncFormatter(format_lon)
    lat_formatter = FuncFormatter(format_lat)

    # -------- Panel 1: Input --------
    ax1 = fig.add_subplot(gs[0, 0])
    vmin_sst, vmax_sst = -2, 30
    im1 = ax1.pcolormesh(lon_coords, lat_coords, input_celsius, cmap=cmap_sst,
                         vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax1.contourf(lon_coords, lat_coords, land_mask, levels=[0.5, 1.5], colors='lightgray')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Input (Day 30, Original Missing)', fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    cbar1 = plt.colorbar(im1, ax=ax1, label='°C')

    # -------- Panel 2: Ground Truth --------
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(lon_coords, lat_coords, gt_celsius, cmap=cmap_sst,
                         vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax2.contourf(lon_coords, lat_coords, land_mask, levels=[0.5, 1.5], colors='lightgray')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'Ground Truth (Filled)', fontsize=14, fontweight='bold')
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    cbar2 = plt.colorbar(im2, ax=ax2, label='°C')

    # -------- Panel 3: Prediction --------
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.pcolormesh(lon_coords, lat_coords, pred_celsius, cmap=cmap_sst,
                         vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax3.contourf(lon_coords, lat_coords, land_mask, levels=[0.5, 1.5], colors='lightgray')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title(f'Prediction', fontsize=14, fontweight='bold')
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    cbar3 = plt.colorbar(im3, ax=ax3, label='°C')

    # -------- Panel 4: Error (only in missing regions) --------
    ax4 = fig.add_subplot(gs[1, 1])
    error_masked = np.where(missing_ocean, abs_error, np.nan)
    im4 = ax4.pcolormesh(lon_coords, lat_coords, error_masked, cmap=cmap_err,
                         vmin=0, vmax=np.nanmax(error_masked) if not np.isnan(error_masked).all() else 1,
                         shading='auto')
    ax4.contourf(lon_coords, lat_coords, land_mask, levels=[0.5, 1.5], colors='lightgray')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_title(f'Absolute Error (Missing Regions)\nMAE={mae:.3f}°C, RMSE={rmse:.3f}°C, Max={max_err:.3f}°C',
                  fontsize=14, fontweight='bold')
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)
    cbar4 = plt.colorbar(im4, ax=ax4, label='°C')

    plt.suptitle(f'H=1 Original Jaxa Data Inference - Sample {sample_idx}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")
    print(f"  MAE={mae:.4f}°C, RMSE={rmse:.4f}°C, Max={max_err:.4f}°C")

    return mae, rmse, max_err


def main():
    print("="*70)
    print("H=1 推理 - 原始缺失JAXA数据")
    print("="*70)

    setup_matplotlib()

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH, DEVICE)

    # 获取坐标和陆地掩码
    print(f"获取坐标: {COORDS_FILE}")
    lat_coords, lon_coords, land_mask = get_coords_from_knn(COORDS_FILE)
    H, W = land_mask.shape
    print(f"  形状: {H}×{W}")

    # 选择样本日期（均匀分布）
    start_date, num_days = SERIES_DATES[1]
    sample_indices = [int(i * num_days / 4) for i in range(1, 5)]  # 25%, 50%, 75%, 100%

    all_results = []

    for sample_num, day_idx in enumerate(sample_indices, 1):
        print(f"\n处理样本 {sample_num}/4 (day={day_idx})...")

        target_date = start_date + timedelta(days=day_idx)
        target_date_h01 = target_date.replace(hour=1)

        # ===== 加载30天序列 =====
        sst_seq = []
        mask_seq = []
        original_missing_masks = []

        start_idx = max(0, day_idx - 29)

        for i in range(start_idx, day_idx + 1):
            frame_date = start_date + timedelta(days=i)
            frame_date_h01 = frame_date.replace(hour=1)

            # 加载原始数据（有缺失）
            raw_sst, _, _ = load_jaxa_frame(frame_date_h01)

            if raw_sst is None:
                print(f"    无法加载 {frame_date_h01}, 跳过")
                continue

            # 记录原始缺失位置
            original_missing = np.isnan(raw_sst).astype(np.float32)

            # 时间加权填充
            filled_sst, _, _, _ = temporal_weighted_fill_frame(
                frame_date_h01, start_date.replace(hour=1), 1, lookback_days=2
            )

            if filled_sst is None:
                continue

            # 处理NaN，转换为归一化值
            filled_sst = np.nan_to_num(filled_sst, nan=NORM_MEAN)
            sst_norm = (filled_sst - NORM_MEAN) / NORM_STD

            sst_seq.append(sst_norm)
            mask_seq.append(original_missing)
            original_missing_masks.append(original_missing)

        # Padding到30天
        while len(sst_seq) < 30:
            sst_seq.insert(0, sst_seq[0].copy())
            mask_seq.insert(0, mask_seq[0].copy())

        sst_seq = np.stack(sst_seq[-30:])  # (30, H, W)
        mask_seq = np.stack(mask_seq[-30:])  # (30, H, W)

        # ===== 推理 =====
        pred_norm = inference_sequence(model, sst_seq, mask_seq, DEVICE)

        # ===== 获取目标日期的完整真值（KNN填充数据） =====
        # 从KNN数据中读取该日期的数据作为Ground Truth
        import h5py
        knn_file = Path(KNN_DATA_DIR) / 'jaxa_knn_filled_01.h5'  # series_1

        if knn_file.exists():
            with h5py.File(knn_file, 'r') as f:
                # 找到对应的时间索引
                num_frames = f['sst_data'].shape[0]
                # 使用最后一帧（或根据日期计算索引）
                frame_idx = min(day_idx, num_frames - 1)

                gt_sst = f['sst_data'][frame_idx].astype(np.float32)
                gt_norm = (gt_sst - NORM_MEAN) / NORM_STD
                original_obs_mask = f['original_obs_mask'][frame_idx].astype(np.float32)
        else:
            # 使用最后一帧的原始数据作为真值（填充版）
            gt_sst, _, _ = load_jaxa_frame(target_date_h01)
            gt_sst = temporal_weighted_fill_frame(target_date_h01, start_date.replace(hour=1), 1)[0]
            gt_sst = np.nan_to_num(gt_sst, nan=NORM_MEAN)
            gt_norm = (gt_sst - NORM_MEAN) / NORM_STD
            original_obs_mask = np.ones_like(gt_sst)

        # ===== 可视化 =====
        save_path = OUTPUT_DIR / f'sample_original_{sample_num:02d}.png'
        mae, rmse, max_err = draw_four_panel_original(
            sst_seq[-1], mask_seq[-1], gt_norm, pred_norm, land_mask,
            lon_coords, lat_coords, NORM_MEAN, NORM_STD, sample_num, save_path
        )

        all_results.append({'sample': sample_num, 'mae': mae, 'rmse': rmse, 'max_err': max_err})

    # ===== 汇总 =====
    print("\n" + "="*70)
    print("推理完成！")
    print("="*70)
    print("\n误差汇总：")
    for res in all_results:
        print(f"  样本{res['sample']}: MAE={res['mae']:.4f}°C, RMSE={res['rmse']:.4f}°C, Max={res['max_err']:.4f}°C")

    if all_results:
        avg_mae = np.mean([r['mae'] for r in all_results])
        avg_rmse = np.mean([r['rmse'] for r in all_results])
        print(f"\n平均: MAE={avg_mae:.4f}°C, RMSE={avg_rmse:.4f}°C")

    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
