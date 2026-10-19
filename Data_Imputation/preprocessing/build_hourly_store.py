#!/usr/bin/env python3
"""
把 JAXA 抽取版逐小时 NC (451x351) 排成一条连续的逐小时时间轴，写入单个 H5。

与填充方法无关: 这里只保存原始观测，不做任何填充。

存储:
  sst           (T, H, W) int16  = round((K - 273.15) / 0.01)，缺失/无效 = -32768
  present       (T,)  uint8      该小时是否有 NC 文件
  valid_count   (T,)  uint32     该帧有效像素数
  slot_hours    (T,)  int64      自 1970-01-01 00:00 起的小时数 (与文件名时间一致)
  land_mask     (H,W) uint8      整个时段从未有过有效观测的像素 (=1)
  lat, lon

时间轴上缺文件的小时保留为全缺失帧并标 present=0，滑窗取样时丢弃含缺帧的窗口。
构建过程校验: (1) 文件名时间 == NC 内 time 变量; (2) int16 量化无损; (3) 无 NaN 之外的异常值。
新输出目录; 目标文件已存在时拒绝运行。
"""

import os
import re
import sys
import json
import time
import argparse
import numpy as np
import h5py
import netCDF4 as nc
from multiprocessing import Pool
from datetime import datetime, timedelta

SRC = '/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3'
OFFSET, SCALE, FILL = 273.15, 0.01, -32768
PAT = re.compile(r'^(\d{4})(\d{2})(\d{2})(\d{2})0000\.nc$')
T1981 = datetime(1981, 1, 1)
T1970 = datetime(1970, 1, 1)
BLOCK = 64  # 每个任务处理的时间槽数


def scan(src):
    files = {}
    for m in sorted(os.listdir(src)):
        if not re.fullmatch(r'\d{6}', m) or not os.path.isdir(f'{src}/{m}'):
            continue
        for d in sorted(os.listdir(f'{src}/{m}')):
            if not os.path.isdir(f'{src}/{m}/{d}'):
                continue
            for f in os.listdir(f'{src}/{m}/{d}'):
                g = PAT.match(f)
                if g:
                    t = datetime(*[int(x) for x in g.groups()])
                    files[t] = f'{src}/{m}/{d}/{f}'
    return files


def read_block(args):
    """读取一段连续时间槽。返回 int16 块和统计。"""
    s0, s1, t_start, paths = args  # paths: {slot: path}
    block = None
    present = np.zeros(s1 - s0, dtype=np.uint8)
    st = dict(max_err=0.0, out_of_range=0, time_mismatch=[], bad_files=[], max_time_offset_s=0.0)
    for slot in range(s0, s1):
        p = paths.get(slot)
        if p is None:
            continue
        try:
            ds = nc.Dataset(p)
            ds.set_auto_mask(False)
            v = np.asarray(ds['sea_surface_temperature'][0], dtype=np.float32)
            t_file = T1981 + timedelta(seconds=int(ds['time'][0]))
            ds.close()
        except Exception as e:  # 坏文件按缺失处理并记录
            st['bad_files'].append((slot, p, str(e)))
            continue
        if block is None:
            block = np.full((s1 - s0,) + v.shape, FILL, dtype=np.int16)
        finite = np.isfinite(v)
        oor = finite & ((v < 250.0) | (v > 330.0))
        st['out_of_range'] += int(oor.sum())
        ok = finite & ~oor
        q = np.rint((v[ok] - OFFSET) / SCALE)
        if q.size:
            st['max_err'] = max(st['max_err'], float(np.abs(q * SCALE + OFFSET - v[ok]).max()))
        out = np.full(v.shape, FILL, dtype=np.int16)
        out[ok] = q.astype(np.int16)
        block[slot - s0] = out
        present[slot - s0] = 1
        t_name = t_start + timedelta(hours=slot)
        dt_s = abs((t_file - t_name).total_seconds())
        st['max_time_offset_s'] = max(st['max_time_offset_s'], dt_s)
        if dt_s > 600:  # NC 内 time 比文件名常有几十秒偏移，超过 10 分钟才算错位
            st['time_mismatch'].append((slot, str(t_name), str(t_file)))
    return s0, s1, block, present, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--max-slots', type=int, default=0, help='只处理前N个时间槽 (测试用)')
    a = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = a.out or f'{repo}/experiments/hourly_store_v1'
    out_path = f'{out_dir}/sst_hourly.h5'
    if os.path.exists(out_path):
        sys.exit(f'拒绝运行: {out_path} 已存在')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    files = scan(SRC)
    times = sorted(files)
    t_start, t_end = times[0], times[-1]
    T = int((t_end - t_start).total_seconds() // 3600) + 1
    if a.max_slots:
        T = min(T, a.max_slots)
    slot_of = {int((t - t_start).total_seconds() // 3600): p for t, p in files.items()}
    print(f'文件 {len(files)}  时间轴 {t_start} → {t_end}  共 {T} 个小时槽  扫描耗时 {time.time()-t0:.1f}s', flush=True)

    with nc.Dataset(files[t_start]) as ds0:
        lat = np.asarray(ds0['lat'][:], dtype=np.float32)
        lon = np.asarray(ds0['lon'][:], dtype=np.float32)
    H, W = len(lat), len(lon)

    tmp_path = out_path + '.tmp'
    jobs = []
    for s0 in range(0, T, BLOCK):
        s1 = min(s0 + BLOCK, T)
        jobs.append((s0, s1, t_start, {s: slot_of[s] for s in range(s0, s1) if s in slot_of}))

    count_pix = np.zeros((H, W), dtype=np.uint32)
    agg = dict(max_err=0.0, out_of_range=0, time_mismatch=[], bad_files=[], max_time_offset_s=0.0)
    present_all = np.zeros(T, dtype=np.uint8)
    valid_count = np.zeros(T, dtype=np.uint32)

    with h5py.File(tmp_path, 'w') as f, Pool(a.workers) as pool:
        dset = f.create_dataset('sst', shape=(T, H, W), dtype=np.int16, chunks=(1, H, W), fillvalue=FILL)
        done = 0
        for s0, s1, block, present, st in pool.imap(read_block, jobs):
            if block is None:
                block = np.full((s1 - s0, H, W), FILL, dtype=np.int16)
            dset[s0:s1] = block
            valid = block != FILL
            count_pix += valid.sum(axis=0, dtype=np.uint32)
            valid_count[s0:s1] = valid.reshape(s1 - s0, -1).sum(axis=1)
            present_all[s0:s1] = present
            agg['max_err'] = max(agg['max_err'], st['max_err'])
            agg['out_of_range'] += st['out_of_range']
            agg['max_time_offset_s'] = max(agg['max_time_offset_s'], st['max_time_offset_s'])
            agg['time_mismatch'] += st['time_mismatch']
            agg['bad_files'] += st['bad_files']
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f'  {done}/{len(jobs)} blocks  {time.time()-t0:.0f}s', flush=True)

        land = (count_pix == 0).astype(np.uint8)
        f.create_dataset('present', data=present_all)
        f.create_dataset('valid_count', data=valid_count)
        f.create_dataset('slot_hours', data=int((t_start - T1970).total_seconds() // 3600) + np.arange(T, dtype=np.int64))
        f.create_dataset('land_mask', data=land)
        f.create_dataset('lat', data=lat)
        f.create_dataset('lon', data=lon)
        f.attrs.update(offset=OFFSET, scale=SCALE, fill_value=FILL, start_time=str(t_start),
                       description='K = sst*scale + offset; fill_value=无观测')

    os.replace(tmp_path, out_path)

    frac = count_pix[land == 0] / max(int(present_all.sum()), 1)
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    hist = np.histogram(frac, bins=bins)[0].tolist()
    stats = {
        'n_slots': T, 'n_present': int(present_all.sum()), 'n_missing_slots': int(T - present_all.sum()),
        'land_fraction': float(land.mean()), 'ocean_pixels': int((land == 0).sum()),
        'ocean_pixel_valid_fraction_hist': {'bins': bins, 'counts': hist},
        'quantization_max_abs_error_K': agg['max_err'],
        'out_of_range_pixels': agg['out_of_range'],
        'max_time_offset_s': agg['max_time_offset_s'],
        'time_mismatch_count': len(agg['time_mismatch']), 'time_mismatch_head': agg['time_mismatch'][:10],
        'bad_files': agg['bad_files'][:20], 'n_bad_files': len(agg['bad_files']),
        'missing_slot_times': [str(t_start + timedelta(hours=int(i))) for i in np.where(present_all == 0)[0]],
        'elapsed_s': time.time() - t0,
    }
    with open(f'{out_dir}/build_stats.json', 'w') as g:
        json.dump(stats, g, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != 'missing_slot_times'}, indent=2))


if __name__ == '__main__':
    main()
