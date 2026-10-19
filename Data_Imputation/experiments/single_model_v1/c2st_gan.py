#!/usr/bin/env python3
"""
C2ST: 用冻结的生成器输出，独立训练一个全新的判别器，看它能否区分"真残差 (真值-B)"与"预测残差 (预测-B)"。
回答: GAN 训练是否真的让输出更难被区分 (B_gan 的可区分度是否显著低于 A_noGAN)，以及现有 GAN 设计 (D 结构、输入形式) 能否抓到差异。

设计要点 (防止自己骗自己):
  - 数据: 验证期大洞样本 (满足训练 GAN 的筛选: big 且有效判别块 >= min_blocks)，按时间排序，前 70% 训练 D、后 30% 测试，中间留 48h 间隔。
  - D 的输入与训练时完全一致: [残差/res_std * V, hole * V]，PatchDiscriminator(in_ch=2, n_down=3)，LSGAN，只在感受野全落在 V 内的位置计分。
  - 变体 plain = 与训练一致；hp = 残差先去掉低频 (σ=3 的归一化高斯卷积，只在真值有效处)，检验"大尺度残差主导、D 抓不到小尺度差异"的猜测。
  - 假样本来源: A (λ=0)、B (GAN)、base (正对照: 预测残差=0，D 必须能轻松识破，否则检验本身有问题)。
  - 指标 (测试集): 逐样本平均分的 AUC (真 vs 假)、配对准确率 (同一样本里真分>假分的比例)，bootstrap 按样本重采样；
    另报训练集上的同口径指标，看 D 是否过拟合；补报正确掩码下的 patch 级准确率 (训练日志里的 d_acc 被无有效块的卡稀释，不可信)。
用法: python experiments/single_model_v1/c2st_gan.py --data DIR/samples_full.h5 --out OUT --ckpt A=..pth --ckpt B=..pth --gpu 7
"""
import os, sys, json, argparse, time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from datasets.single_model_dataset import SingleModelDataset
import eval_gan_effect as E
from training.train_single_gan import gan_region
from models.patchgan_discriminator import PatchDiscriminator, full_valid, d_loss_lsgan


def blur(x, k=19, sigma=3.0):
    g = torch.exp(-0.5 * (torch.arange(k, device=x.device) - k // 2) ** 2 / sigma ** 2); g = (g / g.sum()).float()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='reflect'), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode='reflect'), g.view(1, 1, k, 1))


def highpass(r, tv):
    """r, tv: [1,1,H,W]。只在真值有效处做归一化卷积，去掉低频"""
    low = blur(r * tv) / (blur(tv) + 1e-6)
    return (r - low) * tv


def collect(args, dev, models):
    probe = SingleModelDataset(args.data); cut = int(np.argmax(probe.times >= datetime.strptime(args.val_start, '%Y-%m-%d')))
    tr = SingleModelDataset(args.data, t_range=(0, cut - 80))
    rs = []
    for i in range(min(len(tr), 200)):                      # 与 train_single_gan 相同的 res_std 估计
        s = tr[i]; m = s['loss_mask'] > 0
        if m.any(): rs.append(((s['ground_truth_sst'] - s['base_sst'])[m]).numpy())
    res_std = float(np.concatenate(rs).std()); print(f'res_std {res_std:.4f}', flush=True)
    va = SingleModelDataset(args.data, t_range=(cut, 10 ** 9), big_only=True)
    Dg = PatchDiscriminator(in_ch=2, n_down=3)
    items = []; t0 = time.time(); n_scan = min(len(va), args.scan or len(va))
    for i in range(n_scan):
        s = va[i]; b = {k: v[None].to(dev) for k, v in s.items()}
        V, hole = gan_region(b, args.dil, args.erode)
        w = full_valid(Dg.patch_fraction(V.float().cpu())); nb = float(w.sum())
        if nb < args.min_blocks: continue
        tv = ((b['loss_mask'] + b['original_obs_mask']) > 0).float().unsqueeze(1)
        base = b['base_sst'].unsqueeze(1); gt = b['ground_truth_sst'].unsqueeze(1)
        it = dict(t=int(s['t']), V=V.half(), hole=hole.half(), tv=tv.half())
        rr = (gt - base) / res_std; it['real'] = rr.half(); it['real_hp'] = highpass(rr, tv).half()
        for name, (m, uc, uf) in models.items():
            with torch.no_grad(): comp = E.predict(name, m, b, uc, uf).unsqueeze(1)
            rf = (comp - base) / res_std; it[name] = rf.half(); it[name + '_hp'] = highpass(rf, tv).half()
        it['base'] = torch.zeros_like(rr).half(); it['base_hp'] = torch.zeros_like(rr).half()   # 正对照: 预测残差=0
        items.append(it)
        if len(items) % 50 == 0: print(f'  收集 {len(items)} 个可用样本 (扫描 {i + 1}/{n_scan}) {time.time() - t0:.0f}s', flush=True)
    items.sort(key=lambda x: x['t']); n = len(items); k = int(n * 0.7)
    gap = 48; train = [x for x in items[:k]]; test = [x for x in items[k:] if x['t'] - train[-1]['t'] >= gap] if train else []
    print(f'可用样本 {n}，D 训练 {len(train)}，测试 {len(test)}', flush=True)
    return train, test


def make_x(it, src, hp):
    r = it[src + ('_hp' if hp else '')].float(); V = it['V'].float()
    return torch.cat([r * V, it['hole'].float() * V], 1)


def train_d(train, src, hp, args, dev, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    D = PatchDiscriminator(in_ch=2, n_down=3).to(dev); opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    for step in range(args.steps):
        sel = rng.choice(len(train), args.bs, replace=False); its = [train[i] for i in sel]
        xr = torch.cat([make_x(i, 'real', hp) for i in its]); xf = torch.cat([make_x(i, src, hp) for i in its])
        V = torch.cat([i['V'].float() for i in its]); w = full_valid(D.patch_fraction(V))
        loss = d_loss_lsgan(D(xr), w, D(xf), w); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 300 == 0: print(f'    step {step} loss {loss.item():.4f}', flush=True)
    return D.eval()


@torch.no_grad()
def score(D, items, src, hp):
    out = []
    for it in items:
        V = it['V'].float(); w = full_valid(D.patch_fraction(V))
        sr, sf = D(make_x(it, 'real', hp)), D(make_x(it, src, hp)); n = float(w.sum())
        out.append((float((sr * w).sum() / n), float((sf * w).sum() / n), float(((sr > .5).float() * w).sum() / n), float(((sf < .5).float() * w).sum() / n)))
    return np.array(out)


def auc(a, b):
    """P(a > b)，平局算 0.5"""
    return float(((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()))


def report(sc, rng, nb=1000):
    r, f = sc[:, 0], sc[:, 1]; n = len(r)
    stat = lambda idx: (auc(r[idx], f[idx]), float((r[idx] > f[idx]).mean()), float(0.5 * (sc[idx, 2].mean() + sc[idx, 3].mean())))
    base = stat(np.arange(n)); bs = np.array([stat(rng.integers(0, n, n)) for _ in range(nb)])
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(auc=base[0], auc_ci=[ci[0][0], ci[1][0]], paired_acc=base[1], paired_ci=[ci[0][1], ci[1][1]], patch_acc=base[2], mean_real=float(r.mean()), mean_fake=float(f.mean()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--val-start', default='2023-04-01')
    ap.add_argument('--dil', type=int, default=4); ap.add_argument('--erode', type=int, default=2); ap.add_argument('--min-blocks', type=int, default=100)
    ap.add_argument('--steps', type=int, default=1200); ap.add_argument('--bs', type=int, default=6); ap.add_argument('--lr-d', type=float, default=2e-4)
    ap.add_argument('--scan', type=int, default=0, help='调试: 只扫描前 N 个验证大洞样本'); ap.add_argument('--seeds', type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = torch.device('cuda', a.gpu)
    models = {}
    for s in a.ckpt:
        n, p = s.split('=', 1); m, ck = E.load_model(p, dev); models[n] = (m, not ck['args'].get('no_c', False), not ck['args'].get('no_film', False))
    train, test = collect(a, dev, models)
    for m, _, _ in models.values(): m.cpu()
    torch.cuda.empty_cache()
    srcs = list(models) + ['base']; results = {}; rng = np.random.default_rng(0); lines = ['# C2ST: 独立判别器能否区分 真残差 vs 预测残差', f'D 训练样本 {len(train)}，测试样本 {len(test)}，每个 D 训练 {a.steps} 步 (bs {a.bs}, lr {a.lr_d})，种子数 {a.seeds}\n',
                                                                       '| 假样本来源 | D 输入 | 测试 AUC [95%CI] | 配对准确率 [95%CI] | patch 准确率 | 训练集 AUC |', '|---|---|---|---|---|---|']
    for hp in (False, True):
        for src in srcs:
            for sd in range(a.seeds):
                t0 = time.time(); D = train_d(train, src, hp, a, dev, 100 + sd)
                rt, rr = report(score(D, test, src, hp), rng), report(score(D, train, src, hp), rng, nb=50)
                key = f'{src}|{"hp" if hp else "plain"}|seed{sd}'; results[key] = dict(test=rt, train=rr)
                lines.append(f'| {src} (seed {sd}) | {"高通" if hp else "与训练一致"} | {rt["auc"]:.3f} [{rt["auc_ci"][0]:.3f},{rt["auc_ci"][1]:.3f}] | {rt["paired_acc"]:.3f} [{rt["paired_ci"][0]:.3f},{rt["paired_ci"][1]:.3f}] | {rt["patch_acc"]:.3f} | {rr["auc"]:.3f} |')
                print(lines[-1], f'({time.time() - t0:.0f}s)', flush=True)
    open(os.path.join(a.out, 'c2st.md'), 'w').write('\n'.join(lines)); json.dump(results, open(os.path.join(a.out, 'c2st.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
