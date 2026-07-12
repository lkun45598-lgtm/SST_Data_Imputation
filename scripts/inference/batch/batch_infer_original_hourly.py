#!/usr/bin/env python3
"""
批量推理 + 可视化（原始云层缺失模式）：h01 ~ h22

4连横图风格：
  Filtered JAXA SST | KNN Filled SST | FNO-CBAM Filled SST | Model-KNN Difference

对每个小时：
  - 加载 experiments/jaxa_finetune_h{hh}/best_model.pth
  - 从 experiments/hourly_data/h{hh}/ series 8 取4个均匀样本
  - mask_ratio=0.0：使用原始云层缺失（不人工挖空）
  - 生成4连图 → experiments/jaxa_finetune_h{hh}/inference_original_missing_vis/
  - 汇总各小时差异统计

用法:
  python batch_infer_original_hourly.py [--hours 1-22] [--gpu 6] [--n_samples 4] [--series 8]
"""

import sys
import os
import argparse
import torch
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy.ndimage import gaussian_filter

# ── 项目路径 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3] / 'Data_Imputation'
sys.path.insert(0, str(PROJECT_ROOT))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.jaxa_inference_dataset import JAXAFinetuneDataset

EXP_ROOT = PROJECT_ROOT / 'experiments'

GAUSSIAN_SIGMA = 1.0  # 与 postprocessing/gaussian_filter.py 一致

# ── 高斯滤波（与 GitHub 版本一致）────────────────────────────────────────────
def apply_gaussian_filter(sst_celsius, land_mask, sigma=GAUSSIAN_SIGMA, fill_region=None):
    """NaN填均值 → gaussian_filter → 恢复NaN。
    fill_region 给定时只在该区(云/填充区)写回滤波值，观测保留原值(真值不被滤波)。"""
    sst = sst_celsius.copy()
    mask_valid = ~np.isnan(sst) & (land_mask == 0)
    if mask_valid.sum() == 0:
        return sst
    sst_tmp = sst.copy()
    sst_tmp[~mask_valid] = np.nanmean(sst)
    filtered = gaussian_filter(sst_tmp, sigma=sigma)
    write = mask_valid if fill_region is None else (mask_valid & (fill_region == 1))
    return np.where(write, filtered, np.where(mask_valid, sst, np.nan))


# ── Matplotlib 全局设置 ───────────────────────────────────────────────────────
def setup_matplotlib():
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1.5, labelsize=14)
    plt.rc('lines', linewidth=1.5)
    plt.rcParams.update({
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.pad': 5, 'ytick.major.pad': 5,
    })

def fmt_lon(x, pos): return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"
def fmt_lat(y, pos): return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


# ── 4连横图（原始缺失模式，复刻 fill_jaxa.py 风格）────────────────────────────
def create_four_panel(original_sst_celsius, knn_celsius, model_filled_celsius,
                      original_missing_mask, land_mask,
                      lon_coords, lat_coords, timestamp_str, save_path):
    """
    1. Filtered JAXA SST  (缺失区域=skyblue)
    2. KNN Filled SST     (粗糙填充参考)
    3. FNO-CBAM Filled SST (模型重建，output composition)
    4. Model - KNN Diff   (只在原始缺失区域)
    """
    fig = plt.figure(figsize=(28, 7))
    gs = gridspec.GridSpec(1, 6, figure=fig,
                           width_ratios=[1, 1, 1, 0.08, 1, 0.08],
                           wspace=0.15, hspace=0.1,
                           left=0.04, right=0.98, top=0.85, bottom=0.15)

    ax_orig  = fig.add_subplot(gs[0, 0])
    ax_knn   = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    ax_diff  = fig.add_subplot(gs[0, 4])

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    cmap_sst   = 'RdYlBu_r'
    cmap_diff  = 'RdBu_r'
    land_color = '#D2B48C'
    land_disp  = np.ma.masked_where(land_mask == 0, land_mask.astype(float))

    # SST 色阶（三图共用，取全局 0.5~99.5 百分位）
    ocean = land_mask == 0
    all_vals = np.concatenate([
        original_sst_celsius[ocean & (original_missing_mask == 0)],
        knn_celsius[ocean],
        model_filled_celsius[ocean],
    ])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) > 0:
        vmin = np.floor(np.percentile(all_vals, 0.5))
        vmax = np.ceil(np.percentile(all_vals, 99.5))
        if vmax - vmin < 5:
            mid = (vmin + vmax) / 2
            vmin, vmax = mid - 3, mid + 3
    else:
        vmin, vmax = 20, 32

    ocean_px  = ocean.sum()
    miss_px   = ((original_missing_mask > 0) & ocean).sum()
    miss_rate = miss_px / ocean_px * 100 if ocean_px > 0 else 0

    def _base_ax(ax, yticks=True):
        ax.pcolormesh(lon_grid, lat_grid, land_disp,
                      cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.locator_params(axis='x', nbins=4)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_box_aspect(1)
        if yticks:
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
            ax.locator_params(axis='y', nbins=5)
            ax.set_ylabel('Latitude', fontsize=12)
        else:
            ax.set_yticks([])

    # ===== Panel 1: Filtered JAXA SST =====
    ax_orig.set_facecolor('skyblue')
    _base_ax(ax_orig, yticks=True)
    orig_disp = original_sst_celsius.copy()
    orig_disp[original_missing_mask > 0] = np.nan
    ax_orig.pcolormesh(lon_grid, lat_grid,
                       np.ma.masked_where((land_mask > 0) | np.isnan(orig_disp), orig_disp),
                       cmap=cmap_sst, vmin=vmin, vmax=vmax, shading='auto')
    ax_orig.set_title(f'Filtered JAXA SST\n(Missing: {miss_rate:.1f}%)',
                      fontsize=13, fontweight='bold', pad=10)

    # ===== Panel 2: KNN Filled SST =====
    ax_knn.set_facecolor('lightgray')
    _base_ax(ax_knn, yticks=False)
    ax_knn.pcolormesh(lon_grid, lat_grid,
                      np.ma.masked_where(land_mask > 0, knn_celsius),
                      cmap=cmap_sst, vmin=vmin, vmax=vmax, shading='auto')
    ax_knn.set_title('KNN Filled SST\n(Coarse Fill)',
                     fontsize=13, fontweight='bold', pad=10)

    # ===== Panel 3: FNO-CBAM Filled SST =====
    ax_model.set_facecolor('lightgray')
    _base_ax(ax_model, yticks=False)
    im3 = ax_model.pcolormesh(lon_grid, lat_grid,
                               np.ma.masked_where(land_mask > 0, model_filled_celsius),
                               cmap=cmap_sst, vmin=vmin, vmax=vmax, shading='auto')
    ax_model.set_title(f'FNO-CBAM Filled SST\n(+Gaussian σ={GAUSSIAN_SIGMA})',
                       fontsize=13, fontweight='bold', pad=10)

    # SST colorbar
    cax3c = fig.add_subplot(gs[0, 3]); cax3c.axis('off')
    cax3 = inset_axes(cax3c, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax3c.transAxes)
    plt.colorbar(im3, cax=cax3).set_label('SST (°C)', fontsize=11)

    # ===== Panel 4: Model - KNN Difference =====
    ax_diff.set_facecolor('white')
    _base_ax(ax_diff, yticks=False)
    diff = model_filled_celsius - knn_celsius
    diff_disp = diff.copy()
    diff_disp[(original_missing_mask == 0) | (land_mask > 0)] = np.nan
    im4 = ax_diff.pcolormesh(lon_grid, lat_grid,
                              np.ma.masked_where(np.isnan(diff_disp), diff_disp),
                              cmap=cmap_diff, vmin=-2.0, vmax=2.0, shading='auto')
    ax_diff.set_title('Model - KNN Difference\n(Missing Regions Only)',
                      fontsize=13, fontweight='bold', pad=10)

    valid_diff = diff_disp[~np.isnan(diff_disp)]
    mean_diff = float(np.mean(valid_diff)) if len(valid_diff) > 0 else 0.0
    std_diff  = float(np.std(valid_diff))  if len(valid_diff) > 0 else 0.0
    ax_diff.text(0.02, 0.98,
                 f'Mean: {mean_diff:.3f}°C\nStd: {std_diff:.3f}°C',
                 transform=ax_diff.transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Diff colorbar
    cax4c = fig.add_subplot(gs[0, 5]); cax4c.axis('off')
    cax4 = inset_axes(cax4c, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax4c.transAxes)
    plt.colorbar(im4, cax=cax4).set_label('Diff (°C)', fontsize=11)

    # 顶部日期标题
    fig.text(0.5, 0.98, f'Date: {timestamp_str}',
             ha='center', va='top', fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return mean_diff, std_diff, miss_rate


# ── 单个小时推理 ───────────────────────────────────────────────────────────────
def run_hour(hour: int, device: torch.device, series: int, n_samples: int):
    hh = f'{hour:02d}'
    model_path = EXP_ROOT / f'jaxa_finetune_h{hh}' / 'best_model.pth'
    data_dir   = EXP_ROOT / 'hourly_data' / f'h{hh}'
    coords_f   = data_dir / f'jaxa_knn_filled_{series:02d}.h5'
    out_dir    = EXP_ROOT / f'jaxa_finetune_h{hh}' / 'inference_original_missing_vis'

    if not model_path.exists():
        print(f'  [H={hh}] 模型不存在，跳过')
        return None
    if not coords_f.exists():
        print(f'  [H={hh}] series {series:02d} 数据不存在，跳过')
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    norm_mean = float(ckpt.get('norm_mean', 299.9221))
    norm_std  = float(ckpt.get('norm_std',  2.6919))
    model.eval()

    # 加载坐标
    with h5py.File(coords_f, 'r') as f:
        lat_coords = f['latitude'][:]
        lon_coords = f['longitude'][:]
        timestamps_raw = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else str(ts)
                      for ts in timestamps_raw]

    # 加载数据集（mask_ratio=0 → 使用原始云层缺失）
    dataset = JAXAFinetuneDataset(
        data_dir=str(data_dir),
        series_ids=[series],
        window_size=30,
        mask_ratio=0.0,       # 不人工挖空
        min_mask_size=0,
        max_mask_size=0,
        normalize=True,
        mean=norm_mean,
        std=norm_std,
        cache_size=50,
        seed=42,
    )

    if len(dataset) == 0:
        print(f'  [H={hh}] 数据集为空，跳过')
        return None

    # 均匀选取 n_samples 个样本
    step = max(1, len(dataset) // (n_samples + 1))
    indices = [step * (i + 1) for i in range(n_samples)]
    indices = [min(i, len(dataset) - 1) for i in indices]

    results = []
    with torch.no_grad():
        for idx, s_idx in enumerate(indices):
            batch = dataset[s_idx]

            sst_seq  = batch['input_sst_seq']       # (30, H, W), 归一化
            mask_seq = batch['mask_seq']             # (30, H, W), 1=original missing
            gt_norm  = batch['ground_truth_sst']     # (H, W), KNN filled, 归一化
            orig_obs = batch['original_obs_mask']    # (H, W), 1=observed
            land_mask= batch['land_mask']            # (H, W), 1=land

            orig_missing = (1 - orig_obs).astype(np.float32)  # 1=cloud missing

            # 模型推理
            sst_t  = torch.from_numpy(sst_seq[np.newaxis]).float().to(device)
            mask_t = torch.from_numpy(mask_seq[np.newaxis]).float().to(device)
            pred_norm = model(sst_t, mask_t)[0, 0].cpu().numpy()

            # 反归一化 → Celsius
            to_c = lambda x: x * norm_std + norm_mean - 273.15
            input_c = to_c(sst_seq[-1])       # 第30天输入（缺失区已填0）
            knn_c   = to_c(gt_norm)            # KNN完整参考
            pred_c  = to_c(pred_norm)

            # output composition: 有观测区留原值，缺失区用模型
            composed_c = np.where(orig_missing > 0, pred_c, input_c)
            # 高斯滤波后处理（仅平滑云/填充区，观测区保留真值不动）
            composed_c = apply_gaussian_filter(composed_c, land_mask, sigma=GAUSSIAN_SIGMA,
                                               fill_region=(orig_missing > 0))

            # 时间戳（取第30天对应时间）
            ts_idx = min(s_idx + 29, len(timestamps) - 1)
            ts_str = timestamps[ts_idx][:10] if len(timestamps) > 0 else f'sample_{idx}'

            save_path = out_dir / f'sample_{idx+1:02d}_{ts_str.replace("-", "")}.png'
            mean_d, std_d, miss_r = create_four_panel(
                original_sst_celsius=input_c,
                knn_celsius=knn_c,
                model_filled_celsius=composed_c,
                original_missing_mask=orig_missing,
                land_mask=land_mask,
                lon_coords=lon_coords,
                lat_coords=lat_coords,
                timestamp_str=ts_str,
                save_path=save_path,
            )
            results.append({'miss_rate': miss_r, 'mean_diff': mean_d, 'std_diff': std_d})
            print(f'  [H={hh}] Sample {idx+1}: missing={miss_r:.1f}%, '
                  f'diff_mean={mean_d:+.3f}°C, diff_std={std_d:.3f}°C → {save_path.name}')

    avg_miss = float(np.mean([r['miss_rate']  for r in results]))
    avg_std  = float(np.mean([r['std_diff']   for r in results]))
    print(f'  [H={hh}] 完成 | 平均缺失率={avg_miss:.1f}%, 平均Diff Std={avg_std:.3f}°C')
    return {'hour': hour, 'avg_missing': avg_miss, 'avg_diff_std': avg_std}


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', default='1-22',
                        help='小时范围，如 1-22 或 1,3,5')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=4,
                        help='每小时可视化的样本数')
    parser.add_argument('--series', type=int, default=8,
                        help='使用的数据序列 (默认8=验证集)')
    args = parser.parse_args()

    if '-' in args.hours:
        lo, hi = args.hours.split('-')
        hours = list(range(int(lo), int(hi) + 1))
    else:
        hours = [int(x) for x in args.hours.split(',')]

    setup_matplotlib()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print(f'批量原始缺失推理  |  GPU: {device}  |  hours: {hours}  |  series: {args.series}')
    print('=' * 65)

    summary = []
    for h in hours:
        print(f'\n--- H={h:02d} ---')
        r = run_hour(h, device, series=args.series, n_samples=args.n_samples)
        if r:
            summary.append(r)

    # 汇总
    print('\n' + '=' * 65)
    print(f'{"小时":>6}  {"平均缺失率":>12}  {"Diff Std (°C)":>14}')
    print('-' * 40)
    for r in summary:
        print(f'  H={r["hour"]:02d}    {r["avg_missing"]:>10.1f}%  {r["avg_diff_std"]:>14.3f}')
    if summary:
        print('-' * 40)
        print(f'  平均    '
              f'{np.mean([r["avg_missing"] for r in summary]):>10.1f}%  '
              f'{np.mean([r["avg_diff_std"] for r in summary]):>14.3f}')
    print('=' * 65)


if __name__ == '__main__':
    main()
