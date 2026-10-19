#!/usr/bin/env python3
"""
任务 3: 缺口填充方法对比 (只用验证期数据，不碰测试期)。

两类样本:
  A. 有真值: 验证期里观测覆盖率 >= 0.8 的帧，按真实云形状人工挖洞 (挖洞先于填充)，
     在缺口像素上与真值比较 MAE/RMSE，并比较接缝指标和缺口内部梯度保持程度。
  B. 无真值: 验证期里观测覆盖率 35%-50% 的真实缺口帧，只看接缝指标和填充图。

输出到新目录 (默认 experiments/fill_compare_v2/) : metrics.csv, summary.md, 每个样本的对比图。
"""

import os
import sys
import csv
import time
import json
import argparse
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.hourly_store import HourlyStore
from inference.real_cloud_mask import RealCloudMaskGenerator
from preprocessing.fill_methods import METHODS
from preprocessing.fill_temporal import TEMPORAL_METHODS, build_context
from preprocessing.seam_metrics import seam_stats, seam_reference, gradient_ratio, highpass_stats

NEED_PAST = 72  # 目标帧之前至少连续 72 小时有文件
ALL_METHODS = {**METHODS, **TEMPORAL_METHODS}
VAL_START, VAL_END = datetime(2023, 7, 5), datetime(2024, 7, 4)
TRAIN_END = datetime(2023, 7, 5)
SEASONS = {'winter': (12, 1, 2), 'spring': (3, 4, 5), 'summer': (6, 7, 8), 'autumn': (9, 10, 11)}


def build_cloud_bank(store, n=300, seed=0):
    rng = np.random.default_rng(seed)
    slots = store.slot_range(datetime(2015, 7, 1), TRAIN_END)
    slots = [s for s in slots if store.present[s] and 0.2 <= store.coverage[s] <= 0.95]
    bank = []
    for s in rng.permutation(slots)[:n]:
        bank.append((~np.isfinite(store.frame(int(s)))) & store.ocean)
    return np.array(bank, dtype=bool)


def pick_truth_frames(store, rng, n_extra=2):
    val = [s for s in store.slot_range(VAL_START, VAL_END) if store.present[s] and store.coverage[s] >= 0.8 and s >= NEED_PAST and store.present[s - NEED_PAST:s].all()]
    chosen = []
    for months in SEASONS.values():
        c = [s for s in val if store.times[s].month in months]
        if c:
            chosen.append(int(rng.choice(c)))
    rest = [s for s in val if s not in chosen]
    chosen += [int(s) for s in rng.choice(rest, n_extra, replace=False)]
    return sorted(chosen)


def pick_real_gap_frames(store, rng, n=2):
    val = [s for s in store.slot_range(VAL_START, VAL_END) if store.present[s] and 0.35 <= store.coverage[s] <= 0.5 and s >= NEED_PAST and store.present[s - NEED_PAST:s].all()]
    return sorted(int(s) for s in rng.choice(val, n, replace=False))


def run_methods(field, known, ocean, names, ctx):
    out, secs = {}, {}
    for name in names:
        fn = ALL_METHODS[name]
        t = time.time()
        f = np.where(known, field, np.nan)
        out[name] = fn(f, known, ocean, ctx) if name in TEMPORAL_METHODS else fn(f, known, ocean)
        secs[name] = time.time() - t
    return out, secs


SST_CMAP = 'RdYlBu_r'                          # 论文图配色: 蓝-浅黄白-橙-深红
LAND_COLOR, GAP_COLOR = '#dccba5', '#b8b8b8'    # 陆地米黄 / 缺失灰，两者分开着色
OCEAN = None                                    # main() 里赋值


def _sst_cmap():
    cm = plt.get_cmap(SST_CMAP).copy()
    cm.set_bad(GAP_COLOR)
    return cm


def _land_layer(sl=None):
    rgba = np.zeros(OCEAN.shape + (4,))
    rgba[~OCEAN] = matplotlib.colors.to_rgba(LAND_COLOR)
    return rgba if sl is None else rgba[sl]


def _show(ax, arr, cmap, vmin, vmax, land):
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.imshow(land, interpolation='nearest')
    ax.axis('off')
    return im


def save_fill_figure(path, items, ref, ncols, suptitle):
    lo, hi = np.nanpercentile(ref - 273.15, [1, 99])
    cm = _sst_cmap()
    land = _land_layer()
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, arr) in zip(axes, items):
        im = _show(ax, arr - 273.15, cm, lo, hi, land)
        ax.set_title(title, fontsize=9)
    for ax in axes[len(items):]:
        ax.axis('off')
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label='SST (°C)')
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_zoom_figure(path, items, hole, ncols, suptitle, half=45):
    """围绕最大缺口的边界附近放大，上: 温度，下: 梯度幅值"""
    from scipy.ndimage import label, binary_erosion
    lab, n = label(hole)
    if n == 0:
        return
    sizes = np.bincount(lab.ravel())[1:]
    comp = lab == (1 + int(np.argmax(sizes)))
    edge = comp & ~binary_erosion(comp)
    ys, xs = np.where(edge)
    k = len(ys) // 2
    cy = int(np.clip(ys[k], half, hole.shape[0] - half))
    cx = int(np.clip(xs[k], half, hole.shape[1] - half))
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    crops = [(t, a[sl] - 273.15) for t, a in items]
    # 梯度保留 NaN (缺口/陆地不画)，色标只用填充结果和真值定，避免"输入"格的缺口边缘撑爆
    grads = [(t, np.hypot(*np.gradient(a))) for t, a in crops]
    scale_from = [g for (t, _), (_, g) in zip(crops, grads) if not t.startswith(('observed', 'input'))]
    cm, land = _sst_cmap(), _land_layer(sl)
    gcm = plt.get_cmap('magma').copy()
    gcm.set_bad(GAP_COLOR)
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(3.2 * ncols, 6.0 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    lo, hi = np.nanpercentile(np.concatenate([a[~np.isnan(a)] for _, a in crops]), [1, 99])
    g_hi = np.percentile(np.concatenate([g[np.isfinite(g)] for g in scale_from]), 99)
    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            if i >= len(items):
                axes[2 * r, c].axis('off')
                axes[2 * r + 1, c].axis('off')
                continue
            _show(axes[2 * r, c], crops[i][1], cm, lo, hi, land)
            axes[2 * r, c].set_title(crops[i][0], fontsize=9)
            _show(axes[2 * r + 1, c], grads[i][1], gcm, 0, g_hi, land)
            axes[2 * r + 1, c].set_title('|grad| (K/pixel)', fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ROOT}/experiments/fill_compare_v2')
    ap.add_argument('--methods', default='harmonic,idw,tw,tw_bc,lvo_bc,sameh_bc,rec_bc')
    ap.add_argument('--ratios', default='0.3,0.6')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(f'{a.out}/metrics.csv'):
        sys.exit(f'拒绝运行: {a.out}/metrics.csv 已存在')
    os.makedirs(a.out, exist_ok=True)

    names = a.methods.split(',')
    rng = np.random.default_rng(a.seed)
    store = HourlyStore()
    ocean = store.ocean
    global OCEAN
    OCEAN = ocean
    bank = build_cloud_bank(store, seed=a.seed)
    print(f'云形状库 {len(bank)} 个', flush=True)
    truth_slots = pick_truth_frames(store, rng)
    real_slots = pick_real_gap_frames(store, rng)
    print('有真值帧', [(s, str(store.times[s]), round(float(store.coverage[s]), 3)) for s in truth_slots], flush=True)
    print('真实缺口帧', [(s, str(store.times[s]), round(float(store.coverage[s]), 3)) for s in real_slots], flush=True)

    rows = []
    # A. 有真值
    for s in truth_slots:
        K = store.frame(s)
        known0 = np.isfinite(K) & ocean
        ctx = build_context(store, s)
        for ratio in [float(r) for r in a.ratios.split(',')]:
            gen = RealCloudMaskGenerator(bank, seed=s * 10 + int(ratio * 10))
            art = gen.generate(known0, target_ratio=ratio) > 0
            known, hole = known0 & ~art, art
            fills, secs = run_methods(K, known, ocean, names, ctx)
            ref = seam_stats(K, known, hole)
            tag = f'truth_{store.times[s]:%Y%m%d%H}_r{int(ratio*100)}'
            for name, u in fills.items():
                err = (u - K)[hole]
                ss = seam_stats(u, known, hole)
                hp = highpass_stats(u, K, hole, ocean)
                rows.append(dict(kind='truth', case=tag, ratio=ratio, method=name,
                                 hp_corr=hp['hp_corr'], hp_ratio=hp['hp_ratio'], mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                                 bias=float(err.mean()), grad_ratio=gradient_ratio(np.nan_to_num(u), np.nan_to_num(K), hole),
                                 slope_jump=ss['slope_jump'], value_jump=ss['value_jump'],
                                 ref_slope_jump=ref['slope_jump'], ref_value_jump=ref['value_jump'],
                                 n_quads=ss['n'], seconds=secs[name]))
            shown = np.where(known, K, np.nan)
            items = [('truth', np.where(ocean, K, np.nan)), (f'input (hole {hole.sum()/known0.sum():.0%})', shown)]
            items += [(n, u) for n, u in fills.items()]
            title = f'{store.times[s]} coverage {store.coverage[s]:.2f}, artificial hole ratio {ratio}'
            save_fill_figure(f'{a.out}/{tag}_fills.png', items, K, 3, title)
            save_zoom_figure(f'{a.out}/{tag}_zoom.png', items, hole, 3, title + ' (zoom at hole edge)')
            print(f'  done {tag}', flush=True)

    # B. 真实缺口，无真值
    for s in real_slots:
        K = store.frame(s)
        known = np.isfinite(K) & ocean
        hole = ocean & ~known
        ctx = build_context(store, s)
        fills, secs = run_methods(K, known, ocean, names, ctx)
        ref = seam_reference(K, known, hole)
        tag = f'real_{store.times[s]:%Y%m%d%H}'
        for name, u in fills.items():
            ss = seam_stats(u, known, hole)
            rows.append(dict(kind='real', case=tag, ratio=float(hole.sum() / ocean.sum()), method=name,
                             hp_corr=np.nan, hp_ratio=np.nan, mae=np.nan, rmse=np.nan, bias=np.nan, grad_ratio=np.nan,
                             slope_jump=ss['slope_jump'], value_jump=ss['value_jump'],
                             ref_slope_jump=ref['slope_jump'], ref_value_jump=ref['value_jump'],
                             n_quads=ss['n'], seconds=secs[name]))
        items = [('observed (gray = gap)', np.where(known, K, np.nan))] + [(n, u) for n, u in fills.items()]
        title = f'{store.times[s]} real gap, coverage {store.coverage[s]:.2f}'
        save_fill_figure(f'{a.out}/{tag}_fills.png', items, K, 4, title)
        save_zoom_figure(f'{a.out}/{tag}_zoom.png', items, hole, 4, title + ' (zoom at gap edge)')
        print(f'  done {tag}', flush=True)

    with open(f'{a.out}/metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 汇总
    def mean(kind, method, key):
        v = [r[key] for r in rows if r['kind'] == kind and r['method'] == method and np.isfinite(r[key])]
        return float(np.mean(v)) if v else np.nan

    lines = ['# 填充方法对比 (验证期样本，仅作方法选择)', '',
             f'有真值样本 {len([r for r in rows if r["kind"]=="truth"])//len(names)} 个 (真实云形状挖洞, 缺失比例 {a.ratios}); '
             f'真实缺口样本 {len(real_slots)} 个。', '',
             '## A. 有真值 (缺口像素上，均值)', '',
             '| 方法 | MAE (K) | RMSE (K) | 梯度保持 | 高频相关 | 高频幅度比 | slope_jump | value_jump | 耗时 (s) |',
             '|---|---|---|---|---|---|---|---|---|']
    ref_sj = float(np.mean([r['ref_slope_jump'] for r in rows if r['kind'] == 'truth' and r['method'] == names[0]]))
    ref_vj = float(np.mean([r['ref_value_jump'] for r in rows if r['kind'] == 'truth' and r['method'] == names[0]]))
    for m in names:
        lines.append(f'| {m} | {mean("truth", m, "mae"):.3f} | {mean("truth", m, "rmse"):.3f} | '
                     f'{mean("truth", m, "grad_ratio"):.2f} | {mean("truth", m, "hp_corr"):.2f} | {mean("truth", m, "hp_ratio"):.2f} | {mean("truth", m, "slope_jump"):.4f} | '
                     f'{mean("truth", m, "value_jump"):.4f} | {mean("truth", m, "seconds"):.2f} |')
    lines += ['', f'真值在同一位置的接缝指标 (自然变化量): slope_jump {ref_sj:.4f}, value_jump {ref_vj:.4f} K/像素。',
              '', '## B. 真实缺口 (无真值，接缝指标均值)', '',
              '| 方法 | slope_jump | value_jump |', '|---|---|---|']
    for m in names:
        lines.append(f'| {m} | {mean("real", m, "slope_jump"):.4f} | {mean("real", m, "value_jump"):.4f} |')
    ref_r = [r for r in rows if r['kind'] == 'real' and r['method'] == names[0]]
    lines += ['', f'观测像素在界线附近的参考: slope_jump {np.mean([r["ref_slope_jump"] for r in ref_r]):.4f}, '
                  f'value_jump {np.mean([r["ref_value_jump"] for r in ref_r]):.4f}。']
    with open(f'{a.out}/summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
