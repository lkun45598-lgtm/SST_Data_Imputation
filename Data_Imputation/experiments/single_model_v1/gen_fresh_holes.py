#!/usr/bin/env python3
"""
记忆测试用: 从训练目标帧里抽 n 个，用新的随机种子重新挖洞 (其余流程与 build_single_samples 完全一致:
同样的云库、源云事件池、覆盖率->大/小洞比例、挖洞后重算 30 帧底图/C)，写成与 samples_full.h5 同结构的小文件。
原样本的洞来自 seed=0 (种子 = seed*1000003 + t)，这里默认 seed=777，因此洞和历史缺失都和原来不同。

选帧: 训练期 (t < cut-80，与 train_single_gan 一致) 且原样本 ok=1，大洞/小洞各取一半 (不足则取全部)。
用法: python experiments/single_model_v1/gen_fresh_holes.py --orig DIR/samples_full.h5 --store DIR/sst_hourly.h5 \\
        --out DIR/fresh_holes_train.h5 --n 600 --procs 48
"""
import os, sys, time, argparse
from datetime import datetime
import numpy as np
import h5py
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from preprocessing import build_single_samples as B
from datasets.artificial_holes import HourlyStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', required=True); ap.add_argument('--store', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=600); ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--seed', type=int, default=777, help='必须不同于生成原样本时的 seed (0)')
    ap.add_argument('--pick-seed', type=int, default=5); ap.add_argument('--val-start', default='2023-04-01')
    ap.add_argument('--start', default='2015-07-07'); ap.add_argument('--end', default='2023-07-05')
    ap.add_argument('--min-hole', type=int, default=2000)
    a = ap.parse_args()
    assert a.seed != 0, '种子与原样本相同，洞会完全一样'
    if os.path.exists(a.out): sys.exit('拒绝覆盖: ' + a.out)

    meta = np.load(os.path.join(os.path.dirname(os.path.abspath(a.orig)), 'meta.npz'))
    from datetime import timedelta
    times = np.array([datetime(1970, 1, 1) + timedelta(hours=int(h)) for h in meta['hours']])
    cut = int(np.argmax(times >= datetime.strptime(a.val_start, '%Y-%m-%d')))
    with h5py.File(a.orig, 'r') as f:
        ok = f['ok'][:] > 0; big = f['big'][:] > 0; t = f['t'][:]
    tr = ok & (t < cut - 80)
    rng = np.random.default_rng(a.pick_seed); pick = []
    for flag in (True, False):
        pool_t = t[tr & (big == flag)]
        pick += rng.choice(pool_t, min(a.n // 2, len(pool_t)), replace=False).tolist()
    pick = sorted(pick); n = len(pick)
    print(f'训练帧 {int(tr.sum())}，抽取 {n} (大洞 {sum(big[np.isin(t, pick)])})，cut_slot {cut}', flush=True)

    lo = datetime.strptime(a.start, '%Y-%m-%d'); hi = datetime.strptime(a.end, '%Y-%m-%d')
    st = HourlyStore(a.store); H, W = st.shape; Wp = (W + 7) // 8
    f = h5py.File(a.out, 'w')
    W_ = B.WINDOW
    f.create_dataset('sst', (n, W_, H, W), 'i2', chunks=(1, 1, H, W), compression='gzip', compression_opts=1, shuffle=True)
    f.create_dataset('miss', (n, W_, H, Wp), 'u1', chunks=(1, W_, H, Wp), compression='gzip', compression_opts=1)
    f.create_dataset('C', (n, H, W), 'u1', chunks=(1, H, W), compression='gzip', compression_opts=1)
    f.create_dataset('truth', (n, H, W), 'i2', chunks=(1, H, W), compression='gzip', compression_opts=1, shuffle=True)
    f.create_dataset('B', (n, H, W), 'i2', chunks=(1, H, W), compression='gzip', compression_opts=1, shuffle=True)
    f.create_dataset('hole', (n, H, Wp), 'u1', chunks=(1, H, Wp), compression='gzip', compression_opts=1)
    f.create_dataset('t', data=np.array(pick, np.int64))
    f.create_dataset('ratio', (n,), 'f4'); f.create_dataset('big', (n,), 'u1'); f.create_dataset('ok', (n,), 'u1')
    f.attrs['temp_offset_K'] = 273.15; f.attrs['window'] = W_; f.attrs['seed'] = a.seed
    d = {k: f[k] for k in ('sst', 'miss', 'C', 'truth', 'B', 'hole')}
    jobs = [(i, pick[i], a.seed * 1000003 + pick[i]) for i in range(n)]
    t0 = time.time(); cnt = 0; skipped = 0
    with Pool(a.procs, initializer=B._init, initargs=(lo, hi, a.store, a.min_hole)) as pool:
        for idx, r in pool.imap_unordered(B._work, jobs, chunksize=1):
            cnt += 1
            if r is None: skipped += 1
            else:
                for k in d: d[k][idx] = r[k]
                f['ratio'][idx] = r['ratio']; f['big'][idx] = r['big']; f['ok'][idx] = 1
            if cnt % 20 == 0 or cnt == n:
                f.flush(); el = time.time() - t0
                print(f'{cnt}/{n} 跳过={skipped} {el:.0f}s 预计剩余 {el / cnt * (n - cnt) / 60:.1f} min', flush=True)
    f.close(); print('完成', a.out)


if __name__ == '__main__':
    main()
