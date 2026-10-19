#!/usr/bin/env python3
"""
同一批人工洞上对比若干检查点 (以及纯底图 B 参照) 的精度与细节指标，回答两个问题:
  1. GAN 有没有让细节变好 (梯度/高频谱/残差统计/接缝)，代价是多少 RMSE;
  2. val 变差是不是过拟合 (同一批指标在训练子集上再算一遍，按大洞/小洞分层对比)。

评估区域 = loss_mask (洞 ∩ 有真值 ∩ 海洋)，与训练里 evaluate() 同口径；单位 K。
输出的 comp 与训练一致: 有观测处用输入，缺失处用预测 (compose)。

指标 (均在评估区域内):
  rmse/mae/bias   汇总像素 (pooled)，rmse 与训练日志的 val rmse 做一致性自检
  rmse_le2/_mid/_hi  按 C (过去 48h 观测次数) 分层: <=2, 3-24, >24
  grad_ratio      预测梯度幅值 / 真值梯度幅值 (相邻像素差，两像素都在区域内)，<1 = 偏平滑
  grad_err        |dp-dg| 之和 / |dg| 之和，越小越好
  grad_corr       预测与真值梯度差的逐样本 Pearson 相关，越大越好
  res_rms_ratio   (预测-B) 的均方根 / (真值-B) 的均方根，GAN 的判别对象正是这个残差
  res_corr        (预测-B) 与 (真值-B) 的逐样本相关
  seam_pred/gt    洞边界上 洞内像元与相邻观测像元的 |差| 均值 (预测 / 真值)，excess = pred-gt
  hf_log_psd      洞心 128x128 块的高频段 log10 功率比 (baselines/spectral_metrics.py)，0 最好，<0 偏平滑

对 "第一个检查点" 做配对 bootstrap (按样本重采样) 给差值的 95% 区间。
用法:
  python experiments/single_model_v1/eval_gan_effect.py --data DIR/samples_full.h5 --out OUT \\
      --ckpt ep10=runs/gan50/epoch10.pth --ckpt ep20=runs/gan50/epoch20.pth --gpu 7
"""
import os, sys, json, argparse, time
from datetime import datetime
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'baselines'))
from datasets.single_model_dataset import SingleModelDataset, STD
from models.fno_cbam_single import FNO_CBAM_SST_Single
from training.train_single_gan import MODEL_KW, compose
from spectral_metrics import psd_ratio_at_mask

NB = 64                                  # 128x128 块的径向频率 bin 数
KEYS_STRATA = ['all', 'le2', 'mid', 'hi']


def load_model(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ta = ck['args']
    m = FNO_CBAM_SST_Single(**MODEL_KW, use_cbam=not ta.get('no_cbam', False),
                            use_c=not ta.get('no_c', False), use_film=not ta.get('no_film', False))
    m.load_state_dict(ck['model_state_dict'], strict=True)
    return m.to(dev).eval(), ck


def predict(name, model, b, use_c, use_film):
    if name == 'base':                              # 参照: 洞内直接用底图 B
        pred = b['base_sst'].unsqueeze(1)
    else:
        pred = model(b['input_sst_seq'], b['mask_seq'],
                     b['c_seq'] if use_c else None, b['hour_feat'] if use_film else None)
    return compose(pred, b)[:, 0]                   # [B,H,W] 归一化空间


def _d(t):
    return t[:, :, 1:] - t[:, :, :-1], t[:, 1:, :] - t[:, :-1, :]


def _pm(m):
    return m[:, :, 1:] & m[:, :, :-1], m[:, 1:, :] & m[:, :-1, :]


def batch_metrics(comp, b):
    """返回 {key: [B] float64 ndarray}，全是可相加的充分统计量，汇总在 summarize 里做。"""
    x = comp.double() * STD; g = b['ground_truth_sst'].double() * STD; bs = b['base_sst'].double() * STD
    m = b['loss_mask'] > 0
    mf = m.double(); out = {}
    err = (x - g) * mf
    c = torch.round(b['c_seq'][:, 0] * 48)
    for name, sm in (('all', m), ('le2', m & (c <= 2)), ('mid', m & (c > 2) & (c <= 24)), ('hi', m & (c > 24))):
        s = sm.double()
        out['se_' + name] = ((x - g) ** 2 * s).sum((1, 2)); out['n_' + name] = s.sum((1, 2))
        if name == 'all':
            out['ae_all'] = ((x - g).abs() * s).sum((1, 2)); out['bias_all'] = ((x - g) * s).sum((1, 2))
    # 梯度 (相邻两像素都在区域内)
    z = lambda: torch.zeros(len(x), dtype=torch.float64, device=x.device)
    gn, sp, sg, spp, sgg, spg, ap, ag, ae = z(), z(), z(), z(), z(), z(), z(), z(), z()
    for dp, dg, pm in zip(_d(x), _d(g), _pm(m)):
        w = pm.double()
        gn += w.sum((1, 2)); sp += (dp * w).sum((1, 2)); sg += (dg * w).sum((1, 2))
        spp += (dp * dp * w).sum((1, 2)); sgg += (dg * dg * w).sum((1, 2)); spg += (dp * dg * w).sum((1, 2))
        ap += (dp.abs() * w).sum((1, 2)); ag += (dg.abs() * w).sum((1, 2)); ae += ((dp - dg).abs() * w).sum((1, 2))
    out.update(g_n=gn, g_sp=sp, g_sg=sg, g_spp=spp, g_sgg=sgg, g_spg=spg, g_ap=ap, g_ag=ag, g_ae=ae)
    # 相对 B 的残差
    rp, rg = (x - bs) * mf, (g - bs) * mf
    out.update(r_n=mf.sum((1, 2)), r_sp=rp.sum((1, 2)), r_sg=rg.sum((1, 2)), r_spp=(rp * rp).sum((1, 2)),
               r_sgg=(rg * rg).sum((1, 2)), r_spg=(rp * rg).sum((1, 2)))
    # 接缝: 洞内像元 与 相邻观测像元(目标帧保留的真实观测，非陆地)
    obs = (b['original_obs_mask'] > 0) & (b['land_mask'] <= 0)
    sn, sjp, sjg = z(), z(), z()
    for axis, dp, dg in ((0, _d(x)[0], _d(g)[0]), (1, _d(x)[1], _d(g)[1])):
        ml, mr = _split(m, axis); ol, orr = _split(obs, axis)
        pm = (ml & orr) | (ol & mr); w = pm.double()
        sn += w.sum((1, 2)); sjp += (dp.abs() * w).sum((1, 2)); sjg += (dg.abs() * w).sum((1, 2))
    out.update(s_n=sn, s_jp=sjp, s_jg=sjg)
    return {k: v.cpu().numpy() for k, v in out.items()}


def _split(t, axis):
    """返回 (前一格, 后一格) 两个切片；axis 0 = 水平(W 方向)，1 = 垂直(H 方向)"""
    return (t[:, :, :-1], t[:, :, 1:]) if axis == 0 else (t[:, :-1, :], t[:, 1:, :])


def psd_one(pred_k, gt_k, valid, hole):
    ratio, hf = psd_ratio_at_mask(pred_k, gt_k, valid, hole)
    if ratio is None: return np.full(NB, np.nan), np.nan
    return np.log10(np.maximum(ratio, 1e-6))[:NB], hf


# ---------------------------------------------------------------- 汇总
def _corr(n, sp, sg, spp, sgg, spg):
    den = np.sqrt(np.maximum(n * spp - sp ** 2, 0) * np.maximum(n * sgg - sg ** 2, 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (n * spg - sp * sg) / den
    return np.where((n >= 50) & (den > 0), r, np.nan)


def _sq(a, i, k): return a[k][i].sum()


METRICS = {
    'rmse': lambda a, i: np.sqrt(_sq(a, i, 'se_all') / _sq(a, i, 'n_all')),
    'mae': lambda a, i: _sq(a, i, 'ae_all') / _sq(a, i, 'n_all'),
    'bias': lambda a, i: _sq(a, i, 'bias_all') / _sq(a, i, 'n_all'),
    'rmse_le2': lambda a, i: np.sqrt(_sq(a, i, 'se_le2') / max(_sq(a, i, 'n_le2'), 1)),
    'rmse_mid': lambda a, i: np.sqrt(_sq(a, i, 'se_mid') / max(_sq(a, i, 'n_mid'), 1)),
    'rmse_hi': lambda a, i: np.sqrt(_sq(a, i, 'se_hi') / max(_sq(a, i, 'n_hi'), 1)),
    'grad_ratio': lambda a, i: _sq(a, i, 'g_ap') / _sq(a, i, 'g_ag'),
    'grad_err': lambda a, i: _sq(a, i, 'g_ae') / _sq(a, i, 'g_ag'),
    'grad_corr': lambda a, i: np.nanmean(_corr(*(a[k][i] for k in ('g_n', 'g_sp', 'g_sg', 'g_spp', 'g_sgg', 'g_spg')))),
    'res_rms_ratio': lambda a, i: np.sqrt(_sq(a, i, 'r_spp') / _sq(a, i, 'r_sgg')),
    'res_corr': lambda a, i: np.nanmean(_corr(*(a[k][i] for k in ('r_n', 'r_sp', 'r_sg', 'r_spp', 'r_sgg', 'r_spg')))),
    'seam_pred': lambda a, i: _sq(a, i, 's_jp') / _sq(a, i, 's_n'),
    'seam_gt': lambda a, i: _sq(a, i, 's_jg') / _sq(a, i, 's_n'),
    'seam_excess': lambda a, i: (_sq(a, i, 's_jp') - _sq(a, i, 's_jg')) / _sq(a, i, 's_n'),
    'hf_log_psd': lambda a, i: np.nanmean(a['hf'][i]),
}
GOOD = {'rmse': 'low', 'mae': 'low', 'rmse_le2': 'low', 'rmse_mid': 'low', 'rmse_hi': 'low', 'grad_err': 'low',
        'grad_corr': 'high', 'res_corr': 'high'}          # 其余是"越接近参照越好"，不标方向


def summarize(res, big, names, ref, nboot, rng, label):
    lines = []
    for sname, sel in (('全部', np.ones(len(big), bool)), ('大洞', big), ('小洞', ~big)):
        idx = np.where(sel)[0]
        if len(idx) < 20: continue
        lines.append(f'\n### {label} / {sname} (n={len(idx)})\n')
        lines.append('| 指标 | ' + ' | '.join(names) + ' |'); lines.append('|---|' + '---|' * len(names))
        vals = {n: {k: float(f(res[n], idx)) for k, f in METRICS.items()} for n in names}
        for k in METRICS:
            lines.append(f'| {k} | ' + ' | '.join(f'{vals[n][k]:.4f}' for n in names) + ' |')
        # 配对 bootstrap: 各模型 - 参照
        boots = rng.integers(0, len(idx), (nboot, len(idx)))
        others = [n for n in names if n != ref]
        if others and ref in res:
            lines.append(f'\n差值 (模型 − {ref}) 及 95% bootstrap 区间; * = 区间不含 0\n')
            lines.append('| 指标 | ' + ' | '.join(others) + ' |'); lines.append('|---|' + '---|' * len(others))
            for k, f in METRICS.items():
                cells = []
                for n in others:
                    d = np.array([f(res[n], idx[b]) - f(res[ref], idx[b]) for b in boots])
                    lo, hi = np.nanpercentile(d, [2.5, 97.5]); d0 = vals[n][k] - vals[ref][k]
                    cells.append(f'{d0:+.4f} [{lo:+.4f},{hi:+.4f}]' + ('*' if lo > 0 or hi < 0 else ''))
                lines.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------- 主流程
def run_set(label, ds, models, args, dev, save_pos):
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers)
    acc = {n: {} for n in models}; bigs = []; hfs = {n: [] for n in models}; curves = {n: [] for n in models}
    crops = {}; pos = 0; t0 = time.time()
    for it, b in enumerate(loader):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        bigs.append(b['big'].cpu().numpy() > 0)
        gt_k = (b['ground_truth_sst'] * STD).cpu().numpy()
        tv = ((b['loss_mask'] + b['original_obs_mask']) > 0).cpu().numpy()
        valid = tv & (b['land_mask'].cpu().numpy() <= 0); hole = b['hole'].cpu().numpy()
        for name, (model, uc, uf) in models.items():
            with torch.no_grad():
                comp = predict(name, model, b, uc, uf)
            for k, v in batch_metrics(comp, b).items(): acc[name].setdefault(k, []).append(v)
            comp_k = (comp * STD).cpu().numpy()
            for j in range(len(comp_k)):
                cv, hf = psd_one(comp_k[j], gt_k[j], valid[j], hole[j])
                curves[name].append(cv); hfs[name].append(hf)
                if pos + j in save_pos:
                    crops.setdefault(pos + j, {'gt': np.where(valid[j], gt_k[j], np.nan), 'hole': hole[j],
                                               'base': (b['base_sst'][j] * STD).cpu().numpy(), 'land': b['land_mask'][j].cpu().numpy() > 0, 'm': {}})['m'][name] = comp_k[j]
        pos += len(comp_k)
        if it % 20 == 0: print(f'  [{label}] {pos}/{len(ds)}  {time.time() - t0:.0f}s', flush=True)
    res = {n: {k: np.concatenate(v) for k, v in acc[n].items()} for n in models}
    for n in models: res[n]['hf'] = np.array(hfs[n]); res[n]['psd_curve'] = np.array(curves[n])
    return res, np.concatenate(bigs), crops


def plot_psd(sets, names, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 4), squeeze=False)
    for ax, (label, res, big) in zip(axs[0], sets):
        for n in names:
            cv = res[n]['psd_curve'][big]
            if len(cv): ax.plot(np.arange(NB), np.nanmean(cv, 0), label=n)
        ax.axhline(0, c='k', lw=.8); ax.set_title(f'{label} big holes: log10(PSD_pred/PSD_gt)'); ax.set_xlabel('radial wavenumber bin (high = fine scale)')
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_crops(crops, names, path, contour=True, dpi=100, full=False):
    """每个样本 3 行: 绝对场 / 预测-底图 / 预测-真值 (只在洞内)。第 0 列 = 模型输入。
    配色约定: 陆地 = 浅褐色; 真值/输入里没有观测的地方 (真实云) = 白色; 底图和模型列在真实云区也显示预测值 (只是没有真值可比)。"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = sorted(crops); cols = ['truth', 'base (B)'] + names; nc = len(cols) + 1
    fig, axs = plt.subplots(3 * len(ks), nc, figsize=(2.7 * nc, 2.5 * 3 * len(ks) + 0.4), squeeze=False)
    LAND = (0.80, 0.74, 0.62, 1.0)
    def render(f, land, cmap, lo, hi):
        rgba = plt.get_cmap(cmap)(np.clip((f - lo) / (hi - lo), 0, 1)); rgba[np.isnan(f)] = (1, 1, 1, 1); rgba[land] = LAND
        return rgba
    for r, k in enumerate(ks):
        c = crops[k]; hole, land, gt, base = c['hole'] > 0, c['land'], c['gt'], c['base']
        if full: y0, y1, x0, x1 = 0, hole.shape[0], 0, hole.shape[1]
        else:
            ys, xs = np.where(hole); y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, hole.shape[0]); x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, hole.shape[1])
        sl = (slice(y0, y1), slice(x0, x1)); lo, hi = np.nanpercentile(gt[sl], [2, 98])
        inp = render(gt, land, 'RdYlBu_r', lo, hi); inp[hole & ~land] = (0.5, 0.5, 0.5, 1)
        a = axs[3 * r, 0]; a.imshow(inp[sl])
        if r == 0: a.set_title('input (gray = hole)', fontsize=9)
        a.set_xlabel(f'hole = {100 * hole[np.isfinite(gt)].mean():.0f}% of valid pixels', fontsize=7)
        a = axs[3 * r + 1, 0]; img = np.ones(hole.shape + (4,)); img[hole] = (0.4, 0.4, 0.4, 1); img[land] = LAND
        a.imshow(img[sl]); a.set_xlabel('hole mask', fontsize=7); axs[3 * r + 2, 0].axis('off')
        fields = [gt, base] + [c['m'][n] for n in names]
        for j, (cn, f) in enumerate(zip(cols, fields), start=1):
            f = np.where(land, np.nan, f)
            a = axs[3 * r, j]; a.imshow(render(f, land, 'RdYlBu_r', lo, hi)[sl])
            if r == 0: a.set_title(cn, fontsize=9)
            a = axs[3 * r + 1, j]; a.imshow(render(f - np.where(land, np.nan, base), land, 'RdBu_r', -1.5, 1.5)[sl])
            a = axs[3 * r + 2, j]
            if j == 1: a.axis('off')
            else:
                m = hole & np.isfinite(gt); err = np.where(m, f - gt, np.nan)
                a.imshow(render(err, land, 'RdBu_r', -1.5, 1.5)[sl]); a.set_xlabel(f'RMSE in hole {np.sqrt(np.nanmean(err[m] ** 2)):.3f} K', fontsize=7)
        axs[3 * r, 0].set_ylabel(f'sample {k}\nabsolute', fontsize=8); axs[3 * r + 1, 0].set_ylabel('minus base (B)', fontsize=8)
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.text(0.01, 0.003, 'brown = land | white = no observation (real cloud) in truth/input; base and models show predictions there | gray = artificial hole', fontsize=8)
    fig.tight_layout(rect=(0, 0.012, 1, 1)); fig.savefig(path, dpi=dpi); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', action='append', required=True, help='name=path，第一个作为 bootstrap 参照')
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--val-start', default='2023-04-01')
    ap.add_argument('--n-train', type=int, default=1970, help='训练子集样本数 (0 = 不评估)')
    ap.add_argument('--limit', type=int, default=0, help='调试: 每个集合只取前 N 个样本')
    ap.add_argument('--boot', type=int, default=500); ap.add_argument('--n-crops', type=int, default=4)
    ap.add_argument('--no-base', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda', a.gpu)

    probe = SingleModelDataset(a.data)
    cut = int(np.argmax(probe.times >= datetime.strptime(a.val_start, '%Y-%m-%d')))
    va = SingleModelDataset(a.data, t_range=(cut, 10 ** 9)); tr = SingleModelDataset(a.data, t_range=(0, cut - 80))
    print(f'train {len(tr)} val {len(va)} cut_slot {cut}', flush=True)

    models, names, cks = {}, [], {}
    for s in a.ckpt:
        n, p = s.split('=', 1); m, ck = load_model(p, dev)
        models[n] = (m, not ck['args'].get('no_c', False), not ck['args'].get('no_film', False)); names.append(n); cks[n] = ck
        print(f'{n}: epoch {ck["epoch"]}  训练日志 val rmse {ck["val"]["rmse"]:.4f}', flush=True)
    if not a.no_base: models['base'] = (None, False, False)
    all_names = list(models)

    vsel = list(range(min(len(va), a.limit or len(va))))
    va_sub = Subset(va, vsel)
    with h5py.File(a.data, 'r') as f: vbig = np.where(f['big'][:][va.idx[vsel]] > 0)[0]
    pick = set(vbig[np.linspace(0, len(vbig) - 1, a.n_crops).astype(int)].tolist()) if len(vbig) else set()

    sets = []; text = [f'# 评估 {datetime.now():%F %T}\n', '检查点: ' + ', '.join(f'{n}(epoch {c["epoch"]+1})' for n, c in cks.items())]
    res_v, big_v, crops = run_set('val', va_sub, models, a, dev, pick)
    # 自检: 与训练日志里同一检查点的 val rmse 对齐 (训练日志的 1976 含 DDP 补齐的重复样本，允许千分之几偏差)
    if not a.limit:
        text.append('\n## 自检 (本脚本 val rmse vs 训练日志)\n')
        for n in names:
            mine = float(METRICS['rmse'](res_v[n], np.arange(len(big_v)))); ref = cks[n]['val']['rmse']
            text.append(f'- {n}: 本脚本 {mine:.4f} / 训练日志 {ref:.4f} / 差 {mine - ref:+.4f}')
    sets.append(('val', res_v, big_v))
    rng = np.random.default_rng(0)
    text.append(summarize(res_v, big_v, all_names, names[0], a.boot, rng, 'val'))
    if a.n_train > 0:
        tsel = np.sort(np.random.default_rng(1).choice(len(tr), min(a.n_train, len(tr)), replace=False))
        if a.limit: tsel = tsel[:a.limit]
        res_t, big_t, _ = run_set('train', Subset(tr, tsel.tolist()), models, a, dev, set())
        sets.append(('train', res_t, big_t))
        text.append(summarize(res_t, big_t, all_names, names[0], a.boot, rng, 'train(子集)'))
    np.savez_compressed(os.path.join(a.out, 'per_sample.npz'), **{f'{l}__{n}__{k}': v for l, r, _ in sets for n in all_names for k, v in r[n].items()},
                        **{f'{l}__big': b for l, _, b in sets})
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(text))
    print('\n'.join(text))
    try:
        plot_psd(sets, all_names, os.path.join(a.out, 'psd_ratio.png'))
        if crops: plot_crops(crops, names, os.path.join(a.out, 'crops.png'))
    except Exception as e:
        print('作图失败 (不影响数值):', repr(e))


if __name__ == '__main__':
    main()
