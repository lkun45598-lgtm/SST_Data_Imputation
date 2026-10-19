#!/usr/bin/env python3
"""只重画样例图 (不重算指标): 无黑色洞边界线，每个样本单独一张高分辨率图。洞的位置由第三行 (只显示洞内误差) 表示。
用法: python experiments/single_model_v1/plot_crops_only.py --data DIR/samples_full.h5 --out OUT --ckpt A=..pth --ckpt B=..pth --idx 21 719 1260 1969 --gpu 7"""
import os, sys, argparse
from datetime import datetime
import numpy as np, torch
from torch.utils.data import Subset
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from datasets.single_model_dataset import SingleModelDataset
import eval_gan_effect as E
ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--ckpt', action='append', required=True); ap.add_argument('--idx', type=int, nargs='+', required=True)
ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--val-start', default='2023-04-01'); a = ap.parse_args()
a.bs, a.workers = 1, 2; os.makedirs(a.out, exist_ok=True); dev = torch.device('cuda', a.gpu)
probe = SingleModelDataset(a.data); cut = int(np.argmax(probe.times >= datetime.strptime(a.val_start, '%Y-%m-%d')))
va = SingleModelDataset(a.data, t_range=(cut, 10 ** 9))
models, names = {}, []
for s in a.ckpt:
    n, p = s.split('=', 1); m, ck = E.load_model(p, dev); models[n] = (m, not ck['args'].get('no_c', False), not ck['args'].get('no_film', False)); names.append(n)
models['base'] = (None, False, False)
_, _, crops = E.run_set('crops', Subset(va, a.idx), models, a, dev, set(range(len(a.idx))))
for pos, i in enumerate(a.idx):
    E.plot_crops({i: crops[pos]}, names, os.path.join(a.out, f'sample_{i}.png'), contour=False, dpi=130, full=True)
print('完成', a.out)
