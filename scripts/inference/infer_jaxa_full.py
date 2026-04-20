#!/usr/bin/env python3
"""
全量 JAXA SST 推理脚本  h=01 ~ h=23

对每个小时的所有 series (00~08)，对每一帧做推理后保存 NC 文件。

推理方式（方案A）：
  - sst_seq  : 30天 KNN填充序列（原始值，不替换）
  - mask_seq : 前29天用 original_missing_mask；第30天强制置0（与可视化脚本一致）
  - output composition : 有观测区保留KNN值，云区用模型预测
  - Gaussian 滤波 σ=1.0

输出目录结构：
  /data/sst_data/SST_Data_Imputation/YYYYMM/DD/YYYYMMDDHHMMSS.nc

NC 文件变量：
  lat, lon, time
  sst_original      : 原始滤波后JAXA SST（云区为NaN）
  sst_knn_filled    : KNN粗糙填充SST（完整）
  sst_model_filled  : FNO-CBAM重建（output composition + Gaussian σ=1.0）
  original_missing_mask : 云层掩码（1=云/缺失，0=有观测）

用法：
  python infer_jaxa_full.py [--hours 1-23] [--gpu 6] [--batch_size 4]
"""

import sys
import os
import argparse
import torch
import numpy as np
import h5py
import netCDF4 as nc4
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ── 路径 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent / 'Data_Imputation'
sys.path.insert(0, str(PROJECT_ROOT))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

EXP_ROOT    = PROJECT_ROOT / 'experiments'
HOURLY_ROOT = EXP_ROOT / 'hourly_data'
OUTPUT_ROOT = Path('/data/sst_data/SST_Data_Imputation')
WINDOW_SIZE = 30
GAUSSIAN_SIGMA = 1.0
N_SERIES = 9   # series 00 ~ 08


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def parse_timestamp(ts_str: str):
    """'2017-07-06T01:00:00' → ('201707', '06', '20170706010000')"""
    dt = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
    return dt.strftime("%Y%m"), dt.strftime("%d"), dt.strftime("%Y%m%d%H%M%S")


def apply_gaussian(sst_celsius, land_mask, sigma=GAUSSIAN_SIGMA):
    """NaN填均值 → gaussian_filter → 恢复NaN（与postprocessing/gaussian_filter.py一致）"""
    sst = sst_celsius.copy()
    valid = ~np.isnan(sst) & (land_mask == 0)
    if valid.sum() == 0:
        return sst
    tmp = sst.copy()
    tmp[~valid] = np.nanmean(sst)
    filtered = gaussian_filter(tmp, sigma=sigma)
    return np.where(valid, filtered, np.nan)


TIME_UNITS    = 'seconds since 1981-01-01 00:00:00'
TIME_CALENDAR = 'standard'
TIME_EPOCH    = datetime(1981, 1, 1)


def save_nc_file(output_path: Path, lat, lon, timestamp_str, sst_model):
    """保存单帧NC文件（与 h=00 格式一致：仅 sea_surface_temperature）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dt = datetime.strptime(timestamp_str[:19], "%Y-%m-%dT%H:%M:%S")
    seconds = int((dt - TIME_EPOCH).total_seconds())

    with nc4.Dataset(str(output_path), 'w', format='NETCDF4') as ds:
        ds.createDimension('time', 1)
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))

        v_lat  = ds.createVariable('lat',  'f4', ('lat',))
        v_lon  = ds.createVariable('lon',  'f4', ('lon',))
        v_time = ds.createVariable('time', 'i8', ('time',))

        v_lat[:]  = lat
        v_lon[:]  = lon
        v_time[0] = seconds

        v_lat.units      = 'degrees_north';  v_lat.long_name  = 'latitude'
        v_lon.units      = 'degrees_east';   v_lon.long_name  = 'longitude'
        v_time.units     = TIME_UNITS
        v_time.calendar  = TIME_CALENDAR
        v_time.long_name = 'reference time of sst file'

        v_sst = ds.createVariable('sea_surface_temperature', 'f4',
                                  ('time', 'lat', 'lon'),
                                  fill_value=np.float32(np.nan))
        v_sst[0] = sst_model.astype(np.float32)
        v_sst.units     = 'kelvin'
        v_sst.long_name = 'sea surface skin temperature (FNO filled)'

        ds.title       = 'JAXA SST FNO Filled'
        ds.institution = 'FNO-CBAM Model'
        ds.source      = 'Temporal weighted data + FNO inference + Gaussian filter'
        ds.history     = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def load_model(hour: int, device):
    hh = f'{hour:02d}'
    model_path = EXP_ROOT / f'jaxa_finetune_h{hh}' / 'best_model.pth'
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16
    ).to(device)
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    norm_mean = float(ckpt.get('norm_mean', 299.9221))
    norm_std  = float(ckpt.get('norm_std',  2.6919))
    return model, norm_mean, norm_std


# ── 单小时推理 ────────────────────────────────────────────────────────────────
def run_hour(hour: int, device, batch_size: int):
    hh = f'{hour:02d}'
    print(f'\n{"="*60}')
    print(f'H={hh}  开始推理')
    print(f'{"="*60}')

    model_path = EXP_ROOT / f'jaxa_finetune_h{hh}' / 'best_model.pth'
    if not model_path.exists():
        print(f'  模型不存在，跳过: {model_path}')
        return

    model, norm_mean, norm_std = load_model(hour, device)
    print(f'  模型加载完成 | norm_mean={norm_mean:.4f}K  norm_std={norm_std:.4f}K')

    total_saved = total_skipped = total_failed = 0

    for series_id in range(N_SERIES):
        h5_path = HOURLY_ROOT / f'h{hh}' / f'jaxa_knn_filled_{series_id:02d}.h5'
        if not h5_path.exists():
            print(f'  [series {series_id:02d}] 文件不存在，跳过')
            continue

        # 整个 series 读入内存
        with h5py.File(str(h5_path), 'r') as f:
            sst_all  = f['sst_data'][:]                  # (T, H, W) float32, Kelvin
            miss_all = f['original_missing_mask'][:].astype(np.float32)  # (T, H, W)
            obs_all  = f['original_obs_mask'][:].astype(np.float32)      # (T, H, W)
            land     = f['land_mask'][:].astype(np.float32)              # (H, W)
            lat      = f['latitude'][:]
            lon      = f['longitude'][:]
            ts_raw   = f['timestamps'][:]
            timestamps = [t.decode() if isinstance(t, bytes) else str(t) for t in ts_raw]

        T = sst_all.shape[0]
        print(f'  [series {series_id:02d}] {timestamps[0][:10]} ~ {timestamps[-1][:10]}, {T} 帧')

        # 替换NaN（陆地/无效）→ 0（归一化空间的背景值）
        sst_norm_all = (np.nan_to_num(sst_all, nan=norm_mean) - norm_mean) / norm_std

        # 按 batch 推理
        saved = skipped = failed = 0

        # 收集需要推理的帧索引
        pending = []
        for t in range(T):
            yyyymm, dd, fname = parse_timestamp(timestamps[t])
            out_path = OUTPUT_ROOT / yyyymm / dd / f'{fname}.nc'
            if out_path.exists():
                skipped += 1
            else:
                pending.append(t)

        if skipped > 0:
            print(f'    跳过已存在: {skipped} 帧')

        # 分批推理
        for b_start in range(0, len(pending), batch_size):
            b_indices = pending[b_start: b_start + batch_size]
            B = len(b_indices)

            sst_batch  = np.zeros((B, WINDOW_SIZE, 451, 351), dtype=np.float32)
            mask_batch = np.zeros((B, WINDOW_SIZE, 451, 351), dtype=np.float32)

            for bi, t in enumerate(b_indices):
                start = max(0, t - WINDOW_SIZE + 1)
                win_sst  = sst_norm_all[start: t + 1]   # (<=30, H, W)
                win_miss = miss_all[start: t + 1]        # (<=30, H, W)

                # 不足30天时用第一帧填充
                pad = WINDOW_SIZE - win_sst.shape[0]
                if pad > 0:
                    win_sst  = np.concatenate([np.tile(win_sst[:1],  (pad, 1, 1)), win_sst],  axis=0)
                    win_miss = np.concatenate([np.tile(win_miss[:1], (pad, 1, 1)), win_miss], axis=0)

                # 方案A：第30天的mask置0（与可视化推理保持一致）
                win_miss = win_miss.copy()
                win_miss[-1] = 0.0

                sst_batch[bi]  = win_sst
                mask_batch[bi] = win_miss

            # GPU 推理
            with torch.no_grad():
                sst_t  = torch.from_numpy(sst_batch).to(device)
                mask_t = torch.from_numpy(mask_batch).to(device)
                pred   = model(sst_t, mask_t)             # (B, 1, H, W)
                pred_np = pred[:, 0].cpu().numpy()        # (B, H, W)

            # 逐帧保存
            for bi, t in enumerate(b_indices):
                try:
                    # 反归一化 → Kelvin
                    pred_k = pred_np[bi] * norm_std + norm_mean   # (H, W)
                    knn_k  = sst_all[t]                            # (H, W) KNN填充，含NaN(陆地)

                    # 原始SST：观测区保留，云区为NaN
                    orig_miss = miss_all[t]   # 1=云/缺失
                    sst_orig  = np.where(orig_miss == 0, knn_k, np.nan)

                    # Output composition：有观测区用KNN，云区用模型
                    sst_model = np.where(orig_miss == 1, pred_k, knn_k)
                    sst_model = np.where(land == 1, np.nan, sst_model)  # 陆地置NaN

                    # 高斯滤波
                    sst_model = apply_gaussian(sst_model, land)

                    # 保存NC（仅保存模型填充+滤波后的海温）
                    yyyymm, dd, fname = parse_timestamp(timestamps[t])
                    out_path = OUTPUT_ROOT / yyyymm / dd / f'{fname}.nc'
                    save_nc_file(out_path, lat, lon, timestamps[t], sst_model)
                    saved += 1

                except Exception as e:
                    print(f'    保存失败 t={t} {timestamps[t]}: {e}')
                    failed += 1

            # 进度报告（每100帧）
            done = b_start + B
            if done % 100 == 0 or done == len(pending):
                print(f'    进度: {done}/{len(pending)} 帧完成', flush=True)

        total_saved   += saved
        total_skipped += skipped
        total_failed  += failed
        print(f'  [series {series_id:02d}] 完成: 新增={saved}, 跳过={skipped}, 失败={failed}')

    print(f'\nH={hh} 汇总: 新增={total_saved}, 跳过={total_skipped}, 失败={total_failed}')
    return total_saved


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', default='1-23')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    if '-' in args.hours:
        lo, hi = args.hours.split('-')
        hours = list(range(int(lo), int(hi) + 1))
    else:
        hours = [int(x) for x in args.hours.split(',')]

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f'全量 JAXA 推理')
    print(f'  GPU: {device}  |  batch_size: {args.batch_size}')
    print(f'  hours: {hours}')
    print(f'  output: {OUTPUT_ROOT}')
    print(f'  Gaussian sigma: {GAUSSIAN_SIGMA}')

    t0 = datetime.now()
    total = 0
    for h in hours:
        n = run_hour(h, device, args.batch_size)
        if n:
            total += n

    elapsed = (datetime.now() - t0).total_seconds() / 3600
    print(f'\n全部完成 | 总新增={total} | 耗时={elapsed:.1f}h')


if __name__ == '__main__':
    main()
