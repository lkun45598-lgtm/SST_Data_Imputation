#!/usr/bin/env python3
"""
全量 JAXA SST 推理脚本  h=01 ~ h=23

对每个小时的所有 series (00~08)，对每一帧做推理后保存 NC 文件。

推理方式（方案B，与训练/fill_jaxa.py 一致）：
  - 历史 29 帧 : KNN+时间填充后的代理场（信息增强，原样喂入）
  - 第 30 天   : 只保留真实观测(original_obs_mask==1)，其余(非观测=时间填充+KNN+云)
                 填 norm_mean（归一化后=0）；mask[-1] = 非观测海洋区(=待重建)
  - output composition : 真实观测区保留观测值，所有非观测区用模型预测（重建全部原始缺失）
  - Gaussian 滤波 σ=1.0（只平滑重建区，真实观测不动）

  说明：这与训练完全一致——训练时 loss 只在“填成均值且 mask=1 的洞”上计算，
  故模型唯一被验证过的重建通路就是“mask=1 + 均值填充 → 据历史重建”。旧“方案A”
  （day30 喂满代理、mask 全 0、只写 KNN 区）处于未训练区间且只重建 29.9%，已废弃。

输出目录结构：
  /data/sst_data/SST_Data_Imputation/YYYYMM/DD/YYYYMMDDHHMMSS.nc

NC 文件变量（与 h=00 格式一致）：
  lat, lon, time, sea_surface_temperature  （仅保存模型填充+滤波后的海温）

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
PROJECT_ROOT = Path(__file__).resolve().parents[3] / 'Data_Imputation'
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


def apply_gaussian(sst_celsius, land_mask, fill_region=None, sigma=GAUSSIAN_SIGMA):
    """NaN填均值 → gaussian_filter → 恢复NaN。

    fill_region: 允许被平滑的区域(=模型填充的云区, 1=可平滑)。只在该区写回滤波值，
                 原始观测区保持原值(真值不被滤波)。None 时退回旧行为(整场平滑)。
    高斯卷积输入仍用整场(含观测)，使云区能借真值邻居平滑，但结果不覆盖观测。
    """
    sst = sst_celsius.copy()
    valid = ~np.isnan(sst) & (land_mask == 0)
    if valid.sum() == 0:
        return sst
    tmp = sst.copy()
    tmp[~valid] = np.nanmean(sst)
    filtered = gaussian_filter(tmp, sigma=sigma)
    if fill_region is None:
        write = valid
    else:
        write = valid & (fill_region == 1)
    # 写回区用滤波值；其余(观测/陆地)保留原值(观测=原合成值, 陆地已为NaN)
    return np.where(write, filtered, sst)


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


def load_model(hour: int, device, suffix: str = ''):
    hh = f'{hour:02d}'
    model_path = EXP_ROOT / f'jaxa_finetune_h{hh}{suffix}' / 'best_model.pth'
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
def run_hour(hour: int, device, batch_size: int, suffix: str = ''):
    hh = f'{hour:02d}'
    print(f'\n{"="*60}')
    print(f'H={hh}  开始推理')
    print(f'{"="*60}')

    model_path = EXP_ROOT / f'jaxa_finetune_h{hh}{suffix}' / 'best_model.pth'
    if not model_path.exists():
        print(f'  模型不存在，跳过: {model_path}')
        return

    model, norm_mean, norm_std = load_model(hour, device, suffix)
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

                # 方案B（与训练/fill_jaxa.py 一致）：第30天只留真实观测，其余填均值
                # (归一化空间=0)，mask[-1]=非观测海洋区(待重建)。win_sst 可能是切片视图，
                # 必须先 copy 再改，否则会污染 sst_norm_all。
                obs_t    = obs_all[t]
                nonobs_t = ((obs_t == 0) & (land == 0)).astype(np.float32)
                win_sst  = win_sst.copy()
                win_miss = win_miss.copy()
                win_sst[-1]  = np.where(obs_t == 1, win_sst[-1], 0.0)
                win_miss[-1] = nonobs_t

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

                    # Output composition（方案B）：真实观测区保留观测值(knn_k 在观测处即真值),
                    # 所有非观测区(时间填充+KNN+云)用模型预测 -> 重建全部原始缺失
                    not_obs   = (obs_all[t] == 0) & (land == 0)
                    sst_model = np.where(not_obs, pred_k, knn_k)
                    sst_model = np.where(land == 1, np.nan, sst_model)  # 陆地置NaN

                    # 高斯滤波:只平滑重建区(非观测),真实观测原样保留 -> 连续无接缝
                    sst_model = apply_gaussian(sst_model, land, fill_region=not_obs)

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
    parser.add_argument('--realmask', action='store_true',
                        help='使用真实云形状重训的模型 jaxa_finetune_h{HH}_realmask（推荐，与验证口径一致）')
    args = parser.parse_args()
    suffix = '_realmask' if args.realmask else ''

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
        n = run_hour(h, device, args.batch_size, suffix)
        if n:
            total += n

    elapsed = (datetime.now() - t0).total_seconds() / 3600
    print(f'\n全部完成 | 总新增={total} | 耗时={elapsed:.1f}h')


if __name__ == '__main__':
    main()
