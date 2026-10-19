#!/usr/bin/env python3
"""
记忆测试: 同一批训练目标帧上，比较模型在 "训练时见过的固定洞" 与 "新种子重新挖的洞" 上的误差 (配对)。
  orig  = samples_full.h5 里的原样本 (训练时反复见到的洞和历史缺失)
  fresh = gen_fresh_holes.py 生成的同帧新洞 (新种子，洞形状/比例/历史缺失都重抽，底图和 C 都重算)
判读:
  fresh 的 RMSE ≈ orig  -> 没有记住具体的洞，训练/验证差距来自别处 (帧内容/时间分布)
  fresh 的 RMSE 明显升高 -> 记住了固定的洞 (或其输入模式)
注意: 目标帧 t 本身仍会作为别的样本的历史输入被模型见过，所以本测试只度量"洞/历史缺失模式"的记忆，
      不度量"帧内容"的记忆。判读时要同时看底图 B 的 fresh-orig 差异 (难度对照)。
用法: python experiments/single_model_v1/eval_memorization.py --orig DIR/samples_full.h5 --fresh DIR/fresh_holes_train.h5 \\
        --out OUT --ckpt ep10=...pth --ckpt ep20=...pth --gpu 7
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import Subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from datasets.single_model_dataset import SingleModelDataset
import eval_gan_effect as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', required=True); ap.add_argument('--fresh', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', action='append', required=True); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=4); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda', a.gpu)
    meta = os.path.join(os.path.dirname(os.path.abspath(a.orig)), 'meta.npz')
    od = SingleModelDataset(a.orig, meta=meta); fd = SingleModelDataset(a.fresh, meta=meta)
    to = od.t_all[od.idx]; tf = fd.t_all[fd.idx]
    common = np.intersect1d(to, tf)
    po = {int(t): i for i, t in enumerate(to)}; pf = {int(t): i for i, t in enumerate(tf)}
    so = [po[int(t)] for t in common]; sf = [pf[int(t)] for t in common]
    print(f'orig ok {len(to)}, fresh ok {len(tf)}, 配对帧 {len(common)}', flush=True)

    models, names = {}, []
    for s in a.ckpt:
        n, p = s.split('=', 1); m, ck = E.load_model(p, dev)
        models[n] = (m, not ck['args'].get('no_c', False), not ck['args'].get('no_film', False)); names.append(n)
    models['base'] = (None, False, False); allm = list(models)

    ra, ba, _ = E.run_set('orig', Subset(od, so), models, a, dev, set())
    rb, bb, _ = E.run_set('fresh', Subset(fd, sf), models, a, dev, set())
    assert (ba == bb).all(), '配对帧的大/小洞标记不一致'
    # 洞规模对照 (新旧洞的难度是否可比)
    res = {}
    for n in allm:
        res[n + '_orig'] = ra[n]; res[n + '_fresh'] = rb[n]
    stat = ['## 洞规模对照 (洞内评估像素数的均值)\n',
            f'- orig: {np.mean(ra[allm[0]]["n_all"]):.0f}  fresh: {np.mean(rb[allm[0]]["n_all"]):.0f}  (配对帧 {len(common)}，其中大洞 {int(ba.sum())})']
    rng = np.random.default_rng(0); text = ['# 记忆测试 (同帧: 原固定洞 vs 新洞)\n'] + stat
    for n in allm:
        text.append(E.summarize({k: res[k] for k in (n + '_orig', n + '_fresh')}, ba, [n + '_orig', n + '_fresh'], n + '_orig', a.boot, rng, n))
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(text)); print('\n'.join(text))
    np.savez_compressed(os.path.join(a.out, 'per_sample.npz'), t=common, big=ba,
                        **{f'{k}__{m}': v for k, r in res.items() for m, v in r.items() if m not in ('psd_curve',)})


if __name__ == '__main__':
    main()
