#!/usr/bin/env python3
"""
预生成单模型训练样本 (全量): 训练期每个小时槽作一次目标帧，时间一致人工洞 + 挖洞后重算的底图/C。

目标帧选择: 训练期 (默认 2015-07-07 ~ 2023-07-04) 内、本帧及前 80 小时都有文件的槽。
  覆盖率 >= 0.7 的帧 -> 大洞 (挖 40-60% 可用观测，同时作为 GAN 样本, big=1)
  其余帧            -> 小洞 (挖 20-30% 可用观测, big=0)；洞 < 2000 像素的帧跳过 (ok=0)

每个样本 (时间顺序，下标 0 = t-29 小时 ... 下标 29 = 目标帧 t):
  sst   (N,30,H,W) int16 gzip  填充后 SST ((K-273.15)*100，陆地 = -32768)
  miss  (N,30,H,Wp) uint8      np.packbits(沿 W)，1 = 该帧该像素挖洞后仍无观测 (海洋)
  C     (N,H,W)    uint8       目标帧挖洞后，过去 48h 观测次数
  truth (N,H,W)    int16       目标帧真值 (云/陆地 = -32768)
  B     (N,H,W)    int16       目标帧底图
  hole  (N,H,Wp)   uint8       packbits，人工洞
  t, ratio, big, ok            元数据；ok=1 表示该下标已写入 (用于断点续跑)

用法: python preprocessing/build_single_samples.py --out DIR/samples_full.h5 --procs 130 [--resume]
      冒烟: --limit 40 --procs 4
"""
import os, sys, time, argparse
import numpy as np
import h5py
from datetime import datetime
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
from datasets.artificial_holes import HourlyStore, build_bank_from_store, make_sample

WINDOW = 30
FILL = -32768
SMALL = (0.20, 0.30)
BIG = (0.40, 0.60)
BIG_COVERAGE = 0.7

_g = {}


def _q(a):
    out = np.full(a.shape, FILL, np.int16)
    m = np.isfinite(a)
    out[m] = np.round((a[m] - 273.15) * 100).astype(np.int16)
    return out


def _targets(st, lo, hi):
    tr = st.slot_range(lo, hi)
    return [int(s) for s in tr if s >= 100 and st.present[s - 80:s + 1].all()], tr


def _init(lo, hi, path, min_hole):
    _g['min_hole'] = min_hole
    st = HourlyStore(path) if path else HourlyStore()
    _, tr = _targets(st, lo, hi)
    _g['st'] = st
    _g['bank'] = build_bank_from_store(st, tr, n=300)
    _g['src'] = [s for s in tr if s >= 100 and st.present[s - 80:s + 1].all() and 0.2 <= st.coverage[s] <= 0.6]


def _work(args):
    idx, t, seed = args
    st = _g['st']; rng = np.random.default_rng(seed)
    big = bool(st.coverage[t] >= BIG_COVERAGE)
    lo, hi = BIG if big else SMALL
    ratio = float(rng.uniform(lo, hi))
    s = make_sample(st, t, _g['bank'], rng, ratio, _g['src'], window=WINDOW, min_hole=_g['min_hole'])
    if s is None:
        return idx, None
    oc = st.ocean
    sst = np.stack([s['filled'][k] for k in range(WINDOW - 1, -1, -1)])
    miss = np.stack([(~s['known'][k]) & oc for k in range(WINDOW - 1, -1, -1)])
    return idx, dict(
        sst=_q(sst), miss=np.packbits(miss, axis=-1), C=s['C'], truth=_q(s['truth']), B=_q(s['B']),
        hole=np.packbits(s['hole'], axis=-1), t=t, ratio=float(s['hole'].sum() / oc.sum()), big=int(big))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--store', default=None, help='sst_hourly.h5 路径 (缺省用 datasets.hourly_store 的默认位置)')
    ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--start', default='2015-07-07')
    ap.add_argument('--end', default='2023-07-05')
    ap.add_argument('--min-hole', type=int, default=2000, help='人工洞最少像素数 (低覆盖率帧可放宽)')
    ap.add_argument('--only-skipped-from', default=None, help='只处理该样本文件里 ok=0 的目标帧')
    ap.add_argument('--limit', type=int, default=0, help='只做前 N 个目标 (冒烟)')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    exists = os.path.exists(a.out)
    if exists and not a.resume:
        sys.exit('目标已存在，拒绝覆盖 (续跑请加 --resume): ' + a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    lo = datetime.strptime(a.start, '%Y-%m-%d'); hi = datetime.strptime(a.end, '%Y-%m-%d')
    st = HourlyStore(a.store) if a.store else HourlyStore(); H, W = st.shape; Wp = (W + 7) // 8
    targets, _ = _targets(st, lo, hi)
    if a.only_skipped_from:
        with h5py.File(a.only_skipped_from, 'r') as g0:
            skipped = set(g0['t'][:][g0['ok'][:] == 0].tolist())
        targets = [t for t in targets if t in skipped]
    if a.limit:
        targets = targets[::max(len(targets) // a.limit, 1)][:a.limit]
    n = len(targets)
    print(f'目标帧 {n} (覆盖率>=0.7 的大洞帧 {sum(st.coverage[t] >= BIG_COVERAGE for t in targets)})', flush=True)

    f = h5py.File(a.out, 'a' if exists else 'w')
    if not exists:
        f.create_dataset('sst', (n, WINDOW, H, W), 'i2', chunks=(1, 1, H, W), compression='gzip', compression_opts=1, shuffle=True)
        f.create_dataset('miss', (n, WINDOW, H, Wp), 'u1', chunks=(1, WINDOW, H, Wp), compression='gzip', compression_opts=1)
        f.create_dataset('C', (n, H, W), 'u1', chunks=(1, H, W), compression='gzip', compression_opts=1)
        f.create_dataset('truth', (n, H, W), 'i2', chunks=(1, H, W), compression='gzip', compression_opts=1, shuffle=True)
        f.create_dataset('B', (n, H, W), 'i2', chunks=(1, H, W), compression='gzip', compression_opts=1, shuffle=True)
        f.create_dataset('hole', (n, H, Wp), 'u1', chunks=(1, H, Wp), compression='gzip', compression_opts=1)
        f.create_dataset('t', data=np.array(targets, np.int64))
        f.create_dataset('ratio', (n,), 'f4'); f.create_dataset('big', (n,), 'u1'); f.create_dataset('ok', (n,), 'u1')
        f.create_dataset('done', (n,), 'u1')           # 已处理 (含被跳过的)
        f.attrs['temp_offset_K'] = 273.15; f.attrs['window'] = WINDOW
    assert np.array_equal(f['t'][:], np.array(targets, np.int64)), '续跑时目标帧列表与已有文件不一致'
    done = f['done'][:] > 0
    jobs = [(i, targets[i], a.seed * 1000003 + targets[i]) for i in range(n) if not done[i]]
    print(f'待处理 {len(jobs)} / {n}', flush=True)
    t0 = time.time(); cnt = 0; skipped = 0
    d = {k: f[k] for k in ('sst', 'miss', 'C', 'truth', 'B', 'hole')}
    with Pool(a.procs, initializer=_init, initargs=(lo, hi, a.store, a.min_hole)) as pool:
        for idx, r in pool.imap_unordered(_work, jobs, chunksize=2):
            cnt += 1
            if r is None:
                skipped += 1
            else:
                for k in d:
                    d[k][idx] = r[k]
                f['ratio'][idx] = r['ratio']; f['big'][idx] = r['big']; f['ok'][idx] = 1
            f['done'][idx] = 1
            if cnt % 200 == 0 or cnt == len(jobs):
                f.flush()
                el = time.time() - t0
                print(f'{cnt}/{len(jobs)} 跳过={skipped} {el:.0f}s 预计剩余 {el / cnt * (len(jobs) - cnt) / 60:.0f} min', flush=True)
    f.close()
    print('完成', a.out)


if __name__ == '__main__':
    main()
