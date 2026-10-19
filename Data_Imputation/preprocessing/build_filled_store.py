#!/usr/bin/env python3
"""
为逐小时原始库 (hourly_store_v1) 的每个时间槽生成"时间背景 + 去云边校正"的填充底图。

每个槽 t 只用 t 之前 LOOKBACK 个小时的帧 (因果) 和 t 自己的观测:
  1) 背景 B: 过去帧各自对齐整体偏移 (与当前帧共同观测像素差的中位数) 后按 1/lag 加权平均
  2) 过去 LOOKBACK 小时从未被观测的像素, 用调和延拓补全
  3) bc: 当前观测区向内腐蚀 ERODE 像素 (去掉云边污染) 后取 δ = 观测 - B, 经 tanh 软限幅
     (CLIP K) 后调和延拓进缺口, 填充 = B + δ; 观测像素保持原值
参数与 experiments/fill_diag_v1 的实验结论一致 (tw + bc erode2 + tanh 1K)。

输出 (新目录, 目标已存在则拒绝运行):
  sst_filled  (T,H,W) int16  同 hourly_store 的量化 (0.01 K); 无效/陆地 = -32768
  hist_count  (T,H,W) uint8  过去 LOOKBACK 小时内该像素的观测次数 C (0..LOOKBACK)
  valid       (T,)    uint8  该槽是否生成了填充 (本帧缺失或历史帧过少则为 0)

注意: 这是对真实数据的填充。训练/评估里的人工挖洞样本必须在挖洞之后重新计算,
不能直接用这里的结果 (否则被挖像素的真值会经由历史帧泄漏)。
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import h5py
from multiprocessing import Pool
from scipy import sparse
from scipy.ndimage import binary_erosion, label as cclabel, sum as ndsum
from scipy.sparse.linalg import spsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.hourly_store import HourlyStore
from preprocessing.fill_temporal import _offsets, _weighted_background, _complete
from preprocessing.fill_methods import _harmonic_solve

LOOKBACK = 48
ERODE = 2
CLIP = 1.0
MIN_KNOWN = 100      # 当前帧观测像素少于此数: 不做 bc, 只输出背景
MIN_BG_PIXELS = 2000  # 背景有效像素少于此数: 视为无历史, 该槽无效
FILL = -32768
BLOCK = 64

_store = None


def _get_store(path):
    global _store
    if _store is None:
        _store = HourlyStore(path)
    return _store


def _source(known, erode):
    """腐蚀后的观测区 (去云边)"""
    src = binary_erosion(known, iterations=erode) & known if erode > 0 else known.copy()
    return src if src.sum() >= MIN_KNOWN else known


def fill_one(st, t):
    """返回 (filled float32, count uint8, valid bool); 不满足条件时 filled 全 NaN"""
    ocean = st.ocean
    nan_frame = np.full(st.shape, np.nan, dtype=np.float32)

    def get(s):
        return st.frame(s) if 0 <= s < st.T and st.present[s] else nan_frame

    cur = get(t)
    known = np.isfinite(cur) & ocean
    past = {l: get(t - l) for l in range(1, LOOKBACK + 1)}
    count = np.zeros(st.shape, dtype=np.uint8)
    for fr in past.values():
        count += (np.isfinite(fr) & ocean).astype(np.uint8)
    if not st.present[t] or known.sum() < MIN_KNOWN:
        return None, count, False

    field = np.where(known, cur, np.nan)
    bg = _weighted_background(past, _offsets(field, known, past), lambda l: 1.0 / l)
    if np.isfinite(bg[ocean]).sum() < MIN_BG_PIXELS:
        return None, count, False
    bg_full = _complete(bg, ocean)

    src = _source(known, ERODE)
    delta = np.where(src, field - bg_full, 0.0)
    delta = CLIP * np.tanh(delta / CLIP)
    ext = _harmonic_solve(delta, src, ocean) * (~src)
    out = np.where(known, cur, bg_full + ext).astype(np.float32)
    out[~ocean] = np.nan
    return out, count, True


def work(args):
    path, s0, s1 = args
    st = _get_store(path)
    n = s1 - s0
    q = np.full((n, *st.shape), FILL, dtype=np.int16)
    cnt = np.zeros((n, *st.shape), dtype=np.uint8)
    valid = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        f, c, ok = fill_one(st, s0 + i)
        cnt[i] = c
        if ok:
            valid[i] = 1
            qq = np.round((f - st.offset) / st.scale)
            qq[~np.isfinite(f)] = FILL
            q[i] = qq.astype(np.int16)
    return s0, q, cnt, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ROOT}/experiments/filled_store_v1')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--max-slots', type=int, default=0, help='只处理 [start, start+N) 个槽 (测试用)')
    a = ap.parse_args()

    st = HourlyStore()
    out_path = f'{a.out}/filled_hourly.h5'
    if os.path.exists(out_path):
        sys.exit(f'拒绝运行: {out_path} 已存在')
    os.makedirs(a.out, exist_ok=True)

    T = st.T
    end = min(T, a.start + a.max_slots) if a.max_slots else T
    tasks = [(st.path, s, min(s + BLOCK, end)) for s in range(a.start, end, BLOCK)]
    t0 = time.time()
    with h5py.File(out_path, 'w') as f:
        f.attrs.update(dict(description='K = sst_filled*scale + offset; fill_value=无效', fill_value=FILL,
                            offset=st.offset, scale=st.scale, lookback=LOOKBACK, erode=ERODE, clip=CLIP,
                            source=st.path, start_slot=a.start, end_slot=end))
        dq = f.create_dataset('sst_filled', (T, *st.shape), dtype='int16', chunks=(1, *st.shape),
                              compression='lzf', fillvalue=FILL)
        dc = f.create_dataset('hist_count', (T, *st.shape), dtype='uint8', chunks=(1, *st.shape),
                              compression='lzf', fillvalue=0)
        dv = f.create_dataset('valid', (T,), dtype='uint8', fillvalue=0)
        f.create_dataset('land_mask', data=st.land.astype(np.uint8))
        done = 0
        with Pool(a.workers) as pool:
            for s0, q, cnt, valid in pool.imap_unordered(work, tasks):
                dq[s0:s0 + len(q)] = q
                dc[s0:s0 + len(q)] = cnt
                dv[s0:s0 + len(q)] = valid
                done += len(q)
                if (done // BLOCK) % 20 == 0:
                    print(f'{done}/{end - a.start} slots, {time.time() - t0:.0f}s', flush=True)
        n_valid = int(dv[:].sum())
    stats = dict(slots=int(end - a.start), valid=n_valid, seconds=round(time.time() - t0, 1))
    json.dump(stats, open(f'{a.out}/build_stats.json', 'w'), indent=1)
    print(stats)


if __name__ == '__main__':
    main()
