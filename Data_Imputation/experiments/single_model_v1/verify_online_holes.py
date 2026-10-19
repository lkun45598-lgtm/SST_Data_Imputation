#!/usr/bin/env python3
"""
自检 datasets/online_holes.py: 在线换洞的重算结果必须能复现文件里存好的样本。
  [1] 用原样本的洞重算目标帧: filled0 / B / C / known / truth 与文件里的逐位比较 (验证 s0 重放正确)
  [2] 用同一 s0 重算历史帧 k=1 (不涉及洞): 与文件里 sst[28] 逐位比较 (验证"历史填充与洞无关")
  [3] 换新洞: 洞比例落在目标区间、与原洞不同、样本能产出；并测单样本耗时
用法: python experiments/single_model_v1/verify_online_holes.py --data DIR/samples_full.h5 --store DIR/sst_hourly.h5 --n 12
"""
import os, sys, time, argparse
from datetime import datetime
import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from datasets.hourly_store import HourlyStore
from datasets.artificial_holes import fill_from, LOOKBACK
from datasets import online_holes as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--store', required=True)
    ap.add_argument('--n', type=int, default=12); ap.add_argument('--seed', type=int, default=0, help='生成原样本时的 --seed')
    a = ap.parse_args()
    st = HourlyStore(a.store); oc = st.ocean
    src = O.source_slots(st, datetime(2015, 7, 7), datetime(2023, 7, 5))
    print(f'源云事件候选槽 {len(src)}', flush=True)
    f = h5py.File(a.data, 'r')
    ok = np.where(f['ok'][:] > 0)[0]; big = f['big'][:] > 0; tt = f['t'][:]
    rng = np.random.default_rng(1)
    pick = list(rng.choice([i for i in ok if big[i]], a.n // 2, replace=False)) + list(rng.choice([i for i in ok if not big[i]], a.n - a.n // 2, replace=False))
    bank = O.build_bank(st, st.slot_range(datetime(2015, 7, 7), datetime(2023, 4, 1)))
    bad = 0; times = []
    for j in pick:
        j = int(j); t = int(tt[j])
        hole0 = np.unpackbits(f['hole'][j], axis=-1)[:, :st.shape[1]].astype(bool)
        s0 = O.replay_source_slot(st, t, src, a.seed)
        r = O.resample_target(st, t, s0, hole=hole0)
        res = {}
        res['filled0'] = int((O.quantize(r['filled0']) != f['sst'][j, -1]).sum())
        res['B'] = int((O.quantize(r['B']) != f['B'][j]).sum())
        res['C'] = int((r['C'] != f['C'][j]).sum())
        miss0 = np.unpackbits(f['miss'][j, -1], axis=-1)[:, :st.shape[1]].astype(bool)
        res['known'] = int((((~r['known0']) & oc) != miss0).sum())
        res['truth'] = int((O.quantize(r['truth']) != f['truth'][j]).sum())
        # [2] 历史帧 k=1: 与洞无关
        k = 1; ev = {l: ((~np.isfinite(st.frame(s0 - l))) & oc) if st.present[s0 - l] else np.zeros(st.shape, bool) for l in range(1, LOOKBACK + k + 1)}
        def get(l):
            x = st.frame(t - l).copy() if st.present[t - l] else np.full(st.shape, np.nan, np.float32)
            x[ev[l]] = np.nan; return x
        c1 = get(1); known1 = np.isfinite(c1) & oc
        r1 = O.fill_from(st, c1, {l: get(k + l) for l in range(1, LOOKBACK + 1)}, known1)
        res['hist_k1'] = int((O.quantize(r1[0]) != f['sst'][j, -2]).sum())
        flag = any(v for v in res.values()); bad += flag
        # [3] 新洞
        lo, hi = O.ratio_range(st, t); ratio = float(np.random.default_rng(j).uniform(lo, hi))
        t0 = time.time(); n = O.resample_target(st, t, s0, bank, np.random.default_rng(j + 999), ratio); times.append(time.time() - t0)
        fin = np.isfinite(st.frame(t)) & oc
        share = n['hole'].sum() / fin.sum() if n else float('nan')
        diff = (n['hole'] != hole0).mean() if n else float('nan')
        print(f't={t} big={bool(big[j])} 不一致像素数 {res} {"<-- 不一致" if flag else "ok"} | 新洞占可用观测 {share:.2f} (目标 {ratio:.2f}) 与原洞不同像素 {diff:.3f} 耗时 {times[-1]:.1f}s', flush=True)
    print(f'\n逐位一致: {len(pick) - bad}/{len(pick)}；新洞单样本平均耗时 {np.mean(times):.1f}s')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
