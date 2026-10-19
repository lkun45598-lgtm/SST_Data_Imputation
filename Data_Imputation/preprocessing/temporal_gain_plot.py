#!/usr/bin/env python3
"""
证明"聚合时间信息有用": 单帧纯空间重建 vs 聚合过去 L 小时 (tw_bc_s1, L=3/12/48)。

验证期，覆盖率>=0.8 且前 48 小时齐全的帧，按真实云形状挖洞 (挖洞先于填充)，
在缺口像素上算 RMSE。a: 按目标小时(UTC); b: 总体; c: 随缺失比例。
输出到新目录 experiments/temporal_gain_v1/。
"""

import os
import sys
import csv
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.hourly_store import HourlyStore
from inference.real_cloud_mask import RealCloudMaskGenerator
from preprocessing.fill_methods import harmonic_fill
from preprocessing.fill_temporal import _make, _weighted_background
from preprocessing.fill_compare import build_cloud_bank, VAL_START, VAL_END

WINDOWS = [3, 12, 48]
RATIOS = [0.3, 0.6, 0.85]
METHODS = {'single': None}
for L in WINDOWS:
    METHODS[f'w{L}'] = _make(lambda f, o: _weighted_background(f, o, lambda l: 1.0 / l),
                             True, 'past', max_lag=L, smooth=1.0)

_G = {}


def _init(bank):
    _G['store'] = HourlyStore()
    _G['bank'] = bank


def work(args):
    s, hour = args
    st, ocean = _G['store'], _G['store'].ocean
    K = st.frame(s)
    known0 = np.isfinite(K) & ocean
    past = {l: st.frame(s - l) for l in range(1, max(WINDOWS) + 1)}
    out = []
    for ratio in RATIOS:
        gen = RealCloudMaskGenerator(_G['bank'], seed=s * 10 + int(ratio * 100))
        art = gen.generate(known0, target_ratio=ratio) > 0
        known = known0 & ~art
        if known.sum() < 1000:  # 挖洞几乎挖光，任何方法都无从填起
            out.append((s, hour, ratio, 'SKIPPED', 0.0, 0.0, 0))
            continue
        f = np.where(known, K, np.nan)
        for name, fn in METHODS.items():
            u = harmonic_fill(f, known, ocean) if fn is None else fn(f, known, ocean, {'past': past})
            e = (u - K)[art]
            out.append((s, hour, ratio, name, float((e ** 2).sum()), float(np.abs(e).sum()), int(art.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ROOT}/experiments/temporal_gain_v1')
    ap.add_argument('--per-hour', type=int, default=12)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    if os.path.exists(f'{a.out}/rows.csv'):
        sys.exit(f'拒绝运行: {a.out}/rows.csv 已存在')
    os.makedirs(a.out, exist_ok=True)

    store = HourlyStore()
    rng = np.random.default_rng(0)
    cand = [s for s in store.slot_range(VAL_START, VAL_END)
            if store.present[s] and store.coverage[s] >= 0.8 and store.present[s - 48:s].all()]
    jobs = []
    for h in range(24):
        c = [s for s in cand if store.times[s].hour == h]
        if len(c) > a.per_hour:
            c = list(rng.choice(c, a.per_hour, replace=False))
        jobs += [(int(s), h) for s in c]
    print(f'样本 {len(jobs)} 帧 (每小时最多 {a.per_hour})', flush=True)
    bank = build_cloud_bank(store, seed=0)

    with Pool(a.workers, initializer=_init, initargs=(bank,)) as p:
        rows = [r for part in p.imap_unordered(work, jobs) for r in part]
    with open(f'{a.out}/rows.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slot', 'hour', 'ratio', 'method', 'sq_sum', 'abs_sum', 'n_pix'])
        w.writerows(rows)

    n_skip = sum(r[3] == 'SKIPPED' for r in rows)
    print(f'跳过退化样本 {n_skip} 个 (挖洞后已知像素<1000)')
    R = np.array([(r[1], r[2], list(METHODS).index(r[3]), r[4], r[5], r[6]) for r in rows if r[3] != 'SKIPPED'])
    names = list(METHODS)
    label = {'single': 'Single frame (spatial only)', 'w3': 'Past 3 h', 'w12': 'Past 12 h', 'w48': 'Past 48 h'}
    style = {'single': dict(color='#404040', ls='--', marker='o'), 'w3': dict(color='#0072B2', marker='o'),
             'w12': dict(color='#D55E00', marker='s'), 'w48': dict(color='#009E73', marker='^')}

    def rmse(mask, m):
        sel = mask & (R[:, 2] == m)
        return np.sqrt(R[sel, 3].sum() / R[sel, 5].sum())

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    mid = np.isclose(R[:, 1], 0.6)
    for m, n in enumerate(names):
        y = [rmse(mid & (R[:, 0] == h), m) for h in range(24)]
        ax[0].plot(range(24), y, label=label[n], lw=2, ms=4, markevery=3, **style[n])
    ax[0].set(xlabel='Target hour (UTC)', ylabel='RMSE in gaps (K)', xticks=range(0, 25, 6), xlim=(0, 23))
    ax[0].yaxis.grid(True, color='#e0e0e0'); ax[0].legend(frameon=False, fontsize=9)
    ax[1].bar(range(len(names)), [rmse(np.ones(len(R), bool), m) for m in range(len(names))],
              color=[style[n]['color'] for n in names])
    ax[1].set(xticks=range(len(names)), ylabel='RMSE in gaps (K)', title='All hours, all ratios')
    ax[1].set_xticklabels([label[n].replace(' (spatial only)', '') for n in names], rotation=20, ha='right')
    for m, n in enumerate(names):
        ax[2].plot(RATIOS, [rmse(np.isclose(R[:, 1], r), m) for r in RATIOS], label=label[n], lw=2, **style[n])
    ax[2].set(xlabel='Artificial missing ratio', ylabel='RMSE in gaps (K)', xticks=RATIOS)
    ax[2].yaxis.grid(True, color='#e0e0e0')
    for k, t in enumerate('abc'):
        ax[k].text(-0.02, 1.08, t, transform=ax[k].transAxes, fontweight='bold', fontsize=14)
    fig.savefig(f'{a.out}/temporal_gain.png', dpi=200)
    fig.savefig(f'{a.out}/temporal_gain.pdf')

    print('\n总体 RMSE / MAE (K):')
    for m, n in enumerate(names):
        sel = R[:, 2] == m
        print(f'  {label[n]:28s} RMSE {rmse(np.ones(len(R), bool), m):.3f}  MAE {R[sel,4].sum()/R[sel,5].sum():.3f}')
    for r in RATIOS:
        print(f'  ratio {r}:', {n: round(rmse(np.isclose(R[:, 1], r), m), 3) for m, n in enumerate(names)})


if __name__ == '__main__':
    main()
