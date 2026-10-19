#!/usr/bin/env python3
"""
给 build_filled_store.py 生成的库补上"仅背景"槽: 有文件但当前帧观测太少 (<MIN_KNOWN) 而被跳过的槽。
输出过去 LOOKBACK 小时的时间背景 B (不做 bc，当前观测不足以校正)。整体偏移对齐在共同观测像素不足时取 0。
过去 LOOKBACK 小时背景有效像素仍不足 (如库开头) 的槽保持 valid=0。
结果写入同一文件: sst_filled / valid 置位，新增 bg_only (T,) uint8 标记这些槽。训练取样应排除 bg_only=1。
"""
import os, sys, time, argparse
import numpy as np
import h5py
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datasets.hourly_store import HourlyStore
from preprocessing.fill_temporal import _offsets, _weighted_background, _complete
from preprocessing.build_filled_store import LOOKBACK, MIN_BG_PIXELS, FILL

_st = None


def one(t):
    global _st
    if _st is None:
        _st = HourlyStore()
    st = _st
    nan = np.full(st.shape, np.nan, np.float32)
    get = lambda s: st.frame(s) if 0 <= s < st.T and st.present[s] else nan
    cur = get(t); known = np.isfinite(cur) & st.ocean
    past = {l: get(t - l) for l in range(1, LOOKBACK + 1)}
    offs = _offsets(np.where(known, cur, np.nan), known, past)
    offs = {l: offs.get(l, 0.0) for l in past}
    bg = _weighted_background(past, offs, lambda l: 1.0 / l)
    if np.isfinite(bg[st.ocean]).sum() < MIN_BG_PIXELS:
        return t, None
    full = _complete(bg, st.ocean).astype(np.float32)
    out = np.where(known, cur, full)            # 少量当前观测原样保留
    q = np.round((out - st.offset) / st.scale)
    q[~st.ocean] = FILL
    return t, q.astype(np.int16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--file', default=f'{ROOT}/experiments/filled_store_v1/filled_hourly.h5')
    ap.add_argument('--workers', type=int, default=64); a = ap.parse_args()
    st = HourlyStore(); t0 = time.time()
    with h5py.File(a.file, 'r+') as f:
        if 'bg_only' in f: sys.exit('bg_only 已存在，拒绝重复运行')
        valid = f['valid'][:]
        todo = np.flatnonzero((st.present) & (valid == 0))
        print('待补槽', len(todo), flush=True)
        bg = f.create_dataset('bg_only', (len(valid),), dtype='uint8', fillvalue=0)
        done = ok = 0
        with Pool(a.workers) as p:
            for t, q in p.imap_unordered(one, todo, chunksize=8):
                done += 1
                if q is not None:
                    f['sst_filled'][t] = q; f['valid'][t] = 1; bg[t] = 1; ok += 1
        print('补充成功', ok, '/', len(todo), '耗时 %.0fs' % (time.time() - t0))
        print('valid 总数', int(f['valid'][:].sum()), '| bg_only', int(bg[:].sum()))


if __name__ == '__main__':
    main()
