#!/usr/bin/env python3
"""OnlineHoleDataset 自检: (1) 用文件里的原数组走 _assemble，必须与父类输出逐位相同; (2) p_online=0 与父类相同;
(3) 在线换洞: 历史帧不变、洞/目标帧/C 变、键与形状与父类一致、同一帧两次取样洞不同、耗时。"""
import os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from datasets.single_model_dataset import SingleModelDataset
from datasets.single_model_online import OnlineHoleDataset
data, store = sys.argv[1], sys.argv[2]; cut = 67800
par = SingleModelDataset(data, t_range=(0, cut - 80))
on = OnlineHoleDataset(data, store, seed=0, p_online=1.0, bank_slot_lt=cut - 80, t_range=(0, cut - 80))
on0 = OnlineHoleDataset(data, store, seed=0, p_online=0.0, bank_slot_lt=cut - 80, t_range=(0, cut - 80))
rng = np.random.default_rng(3); idx = rng.choice(len(par), 10, replace=False).tolist(); bad = 0
def same(a, b): return all(torch.equal(a[k], b[k]) for k in a) and set(a) == set(b)
for i in idx:
    p = par[i]; j = int(par.idx[i]); t = int(par.t_all[j])
    a1 = on._assemble(j, t, *on.stored_args(j)); a2 = on0[i]
    ok1, ok2 = same(p, a1), same(p, a2); bad += (not ok1) + (not ok2)
    t0 = time.time(); n1 = on[i]; dt = time.time() - t0; n2 = on[i]
    assert set(n1) == set(p) and all(n1[k].shape == p[k].shape and n1[k].dtype == p[k].dtype for k in p), '键/形状/类型与父类不一致'
    hist_same = torch.equal(n1['input_sst_seq'][:-1], p['input_sst_seq'][:-1]) and torch.equal(n1['mask_seq'][:-1], p['mask_seq'][:-1])
    fin = all(torch.isfinite(v.float()).all() for v in n1.values())
    chg = (n1['hole'] != p['hole']).float().mean().item(); twice = (n1['hole'] != n2['hole']).float().mean().item()
    sub = bool(((n1['loss_mask'] > 0) & (n1['hole'] > 0)).sum() == (n1['loss_mask'] > 0).sum())
    print(f'idx={i} big={bool(p["big"])} 组装复现={ok1} p=0 复现={ok2} 历史不变={hist_same} 有限={fin} 换洞像素占比={chg:.3f} 两次取样洞不同={twice:.3f} loss_mask⊂洞={sub} {dt:.1f}s')
    bad += (not hist_same) + (not fin) + (twice == 0) + (not sub)
print('全部通过' if bad == 0 else f'失败项 {bad}'); sys.exit(1 if bad else 0)
