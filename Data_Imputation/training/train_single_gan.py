#!/usr/bin/env python3
"""
单模型 (24 小时共用) 微调: FNO-CBAM 输出绝对 SST，底图 B 与 C 作输入；可选配对 GAN。

损失: MSE + alpha_grad*梯度 + alpha_boundary*边界 (洞内，旧 train_jaxa_hourly.py 的函数原样复用)
GAN (配对): 同一个洞、同一个 B。真残差=真值-B，假残差=预测(组合后)-B，按训练集残差标准差归一化。
  D 输入 [R*V, hole]，V = 洞向外膨胀 dil 像元 ∩ 有真值且离真实云边>=erode 像元的区域。
  D 只在感受野全落在 V 内的位置计分；仅大洞样本且有效块数 >= min_blocks 才参与。
  λ 调度: [0,warm) 为 0 (D 也不训练)，[warm,ramp_end) 线性升到 lam_final，之后固定。
用法 (8 卡): python training/train_single_gan.py --data DIR/samples_train.h5 --out DIR/run1 --gpus 8
消融: --lam-final 0 (无 GAN) / --no-c / --no-film / --no-pretrain / --no-cbam
冒烟: --gpus 1 --epochs 1 --max-steps 3
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datasets.single_model_dataset import SingleModelDataset, STD
from datasets.single_model_online import OnlineHoleDataset
from models.fno_cbam_single import FNO_CBAM_SST_Single, load_pretrained
from models.patchgan_discriminator import PatchDiscriminator, full_valid, d_loss_lsgan, g_loss_lsgan
from training.train_jaxa_hourly import masked_mse_loss, masked_gradient_loss, boundary_smoothness_loss

MODEL_KW = dict(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6)


def lam_at(ep, a):
    if a.lam_final <= 0 or ep < a.warm: return 0.0
    if ep >= a.ramp_end: return a.lam_final
    return a.lam_final * (ep - a.warm + 1) / (a.ramp_end - a.warm)


def compose(pred, b):
    """目标帧: 有观测处用输入，缺失(洞+真实云)处用预测，陆地保持输入"""
    last_in = b['input_sst_seq'][:, -1:]
    last_m = b['mask_seq'][:, -1:]
    land = b['land_mask'].unsqueeze(1)
    comp = last_in * (1 - last_m) + pred * last_m
    return comp * (1 - land) + last_in * land


def gan_region(b, dil, erode):
    """V = 洞膨胀 ∩ 有真值且离云边>=erode。返回 V(B,1,H,W)、hole(B,1,H,W)"""
    hole = b['hole'].unsqueeze(1)
    tv = ((b['loss_mask'] + b['original_obs_mask']) > 0).float().unsqueeze(1)    # 目标帧有真值
    if erode > 0:
        tv = 1 - F.max_pool2d(1 - tv, 2 * erode + 1, 1, erode)
    mdil = F.max_pool2d(hole, 2 * dil + 1, 1, dil)
    return mdil * tv, hole


def step_inputs(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def forward(model, b, a):
    pred = model(b['input_sst_seq'], b['mask_seq'],
                 b['c_seq'] if a.use_c else None, b['hour_feat'] if a.use_film else None)
    return pred, compose(pred, b)


def recon_loss(comp, b, a):
    tgt = b['ground_truth_sst'].unsqueeze(1)
    ref = ((b['loss_mask'] + b['original_obs_mask']) > 0).float()
    m = masked_mse_loss(comp, tgt, b['loss_mask'])
    g = masked_gradient_loss(comp, tgt, b['loss_mask'], ref) if a.alpha_grad > 0 else comp.new_zeros(())
    bd = boundary_smoothness_loss(comp, tgt, b['mask_seq'], ref) if a.alpha_boundary > 0 else comp.new_zeros(())
    return m + a.alpha_grad * g + a.alpha_boundary * bd, m, g, bd


def allreduce_grads(mod, world):
    if world > 1:
        for p in mod.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad); p.grad /= world


@torch.no_grad()
def evaluate(model, loader, dev, a):
    model.eval(); acc = torch.zeros(6, device=dev)   # se, ae, n, se_lowC, n_lowC, nsamp
    for i, b in enumerate(loader):
        if a.max_steps and i >= a.max_steps: break
        b = step_inputs(b, dev); _, comp = forward(model, b, a)
        err = (comp[:, 0] - b['ground_truth_sst']) * STD; w = b['loss_mask']
        low = w * (b['c_seq'][:, 0] * 48 <= 2).float()
        acc += torch.stack([(err ** 2 * w).sum(), (err.abs() * w).sum(), w.sum(),
                            (err ** 2 * low).sum(), low.sum(), torch.tensor(float(len(err)), device=dev)])
    if dist.is_initialized(): dist.all_reduce(acc)
    se, ae, n, sl, nl, ns = acc.tolist()
    return dict(rmse=(se / max(n, 1)) ** .5, mae=ae / max(n, 1), rmse_C_le2=(sl / max(nl, 1)) ** .5, n_px=n, n_samples=ns)


def worker(rank, world, a):
    dev = torch.device('cuda', rank) if torch.cuda.is_available() and a.gpus > 0 else torch.device('cpu')
    if world > 1:
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', str(a.port))
        dist.init_process_group('nccl', rank=rank, world_size=world); torch.cuda.set_device(rank)
    torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank)
    main = rank == 0

    probe = SingleModelDataset(a.data)
    from datetime import datetime
    cut = int(np.argmax(probe.times >= datetime.strptime(a.val_start, '%Y-%m-%d')))   # 槽下标即小时序号
    va_ds = SingleModelDataset(a.data, t_range=(cut, 10 ** 9))
    if a.store:      # 训练集在线换洞 (固定洞会被记住，见 experiments/single_model_v1/eval_memorization.py)
        tr_ds = OnlineHoleDataset(a.data, a.store, seed=a.seed + 1000 * rank, p_online=a.p_online, bank_slot_lt=cut - 80, t_range=(0, cut - 80))
    else:
        tr_ds = SingleModelDataset(a.data, t_range=(0, cut - 80))
    if main: print(f'train {len(tr_ds)} val {len(va_ds)} cut_slot {cut}', flush=True)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=sh,
                                   sampler=DistributedSampler(ds, world, rank, shuffle=sh) if world > 1 else None,
                                   shuffle=(sh and world == 1), persistent_workers=a.workers > 0)
    tl, vl = mk(tr_ds, True), mk(va_ds, False)

    model = FNO_CBAM_SST_Single(**MODEL_KW, use_cbam=not a.no_cbam, use_c=not a.no_c, use_film=not a.no_film).to(dev)
    if not a.no_pretrain:
        mi, un = load_pretrained(model, a.pretrained, 'cpu')
        if main: print('pretrained: missing', len(mi), 'unexpected', un[:3], flush=True)
    a.use_c, a.use_film = not a.no_c, not a.no_film
    gm = DDP(model, device_ids=[rank]) if world > 1 else model
    if world > 1 and a.fp16_comm:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        gm.register_comm_hook(None, default_hooks.fp16_compress_hook)      # 梯度以 fp16 传输，通信量减半
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)

    D = PatchDiscriminator(in_ch=2, n_down=3).to(dev)
    if world > 1:
        for p in D.parameters(): dist.broadcast(p.data, 0)
    optD = torch.optim.Adam(D.parameters(), lr=a.lr_d, betas=(0.5, 0.999))

    # 残差归一化常数: 训练集前若干个大洞样本上 (真值-B) 的标准差 (各进程相同样本，结果一致)
    rs = []
    for i in range(min(len(tr_ds), 200)):
        s = SingleModelDataset.__getitem__(tr_ds, i); m = s['loss_mask'] > 0     # 用文件里的固定洞，保证各进程一致
        if m.any(): rs.append(((s['ground_truth_sst'] - s['base_sst'])[m]).numpy())
    res_std = float(np.concatenate(rs).std()) if rs else 0.1
    if main: print(f'residual std (normalised units) {res_std:.4f} = {res_std * STD:.3f} K', flush=True)

    os.makedirs(a.out, exist_ok=True); hist = []; best = 1e9; start_ep = 0
    lastp = os.path.join(a.out, 'last.pth')
    if a.resume:
        ck = torch.load(lastp, map_location='cpu', weights_only=False)
        assert 'opt' in ck, 'last.pth 不含优化器状态 (旧版训练保存的)，无法续训'
        model.load_state_dict(ck['model_state_dict']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        D.load_state_dict(ck['D']); optD.load_state_dict(ck['optD']); hist, best, start_ep = ck['hist'], ck['best'], ck['epoch'] + 1
        if main: print(f'续训: 从 epoch {start_ep} 开始 (best val rmse {best:.4f})', flush=True)
        del ck
    for ep in range(start_ep, a.epochs):
        lam = lam_at(ep, a); gm.train(); D.train()
        if world > 1: tl.sampler.set_epoch(ep)
        t0 = time.time(); st = torch.zeros(8, device=dev)   # loss, mse, grad, bd, gan_g, d_loss, d_acc_sum, n_gan
        for it, b in enumerate(tl):
            if a.max_steps and it >= a.max_steps: break
            b = step_inputs(b, dev)
            sync = (it + 1) % a.accum == 0
            if world > 1: gm.require_backward_grad_sync = sync     # 梯度累积: 非边界步不做 all-reduce (等价 no_sync)
            pred, comp = forward(gm, b, a)
            loss, m, g, bd = recon_loss(comp, b, a)
            gan_g = comp.new_zeros(()); use_gan = False
            if lam > 0:
                V, hole = gan_region(b, a.dil, a.erode)
                w = full_valid(D.patch_fraction(V))                                  # (B,1,h,w)
                w = w * (b['big'].view(-1, 1, 1, 1) > 0).float()
                ok = (w.flatten(1).sum(1) >= a.min_blocks).float().view(-1, 1, 1, 1)  # 有效块数不足的样本不参与
                w = w * ok
                flag = torch.tensor(float(w.sum() > 0), device=dev)
                if world > 1: dist.all_reduce(flag)      # 各卡必须同进同出: D 梯度的 all-reduce 次数要一致，否则 NCCL 死锁
                use_gan = float(flag) > 0                # 本卡无有效块时 w 全 0 -> D/G 的 GAN 损失与梯度为 0，但仍参与通信
                if use_gan:
                    base = b['base_sst'].unsqueeze(1)
                    r_real = (b['ground_truth_sst'].unsqueeze(1) - base) / res_std * V
                    r_fake = (comp - base) / res_std * V
                    x_real, x_fake = torch.cat([r_real, hole * V], 1), torch.cat([r_fake, hole * V], 1)
                    # --- D 步 (假样本断开 G 的梯度)
                    for p in D.parameters(): p.requires_grad_(True)
                    sr, sf = D(x_real), D(x_fake.detach())
                    dl = d_loss_lsgan(sr, w, sf, w)
                    optD.zero_grad(set_to_none=True); dl.backward(); allreduce_grads(D, world); optD.step()
                    acc = 0.5 * (((sr > 0.5).float() * w).sum() + ((sf < 0.5).float() * w).sum()) / (w.sum() + 1e-8)
                    # --- G 的对抗项 (冻结 D 参数)
                    for p in D.parameters(): p.requires_grad_(False)
                    gan_g = g_loss_lsgan(D(x_fake), w); loss = loss + lam * gan_g
                    st[5] += dl.detach(); st[6] += acc.detach(); st[7] += 1
            if it % a.accum == 0: opt.zero_grad(set_to_none=True)
            (loss / a.accum).backward()
            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            st[:5] += torch.stack([loss.detach(), m.detach(), g.detach(), bd.detach(), gan_g.detach()])
            if main and it % 20 == 0:
                print(f'ep{ep} it{it} loss {loss.item():.4f} mse {m.item():.4f} lam {lam:.4f} gan_used {use_gan} '
                      f'{(time.time() - t0) / (it + 1):.2f}s/it', flush=True)
        sched.step()
        n_it = max(it + 1, 1)
        if world > 1: dist.all_reduce(st)
        st = st.tolist(); ng = max(st[7], 1)
        val = evaluate(gm, vl, dev, a)
        rec = dict(epoch=ep, lam=lam, loss=st[0] / (n_it * world), mse=st[1] / (n_it * world), grad=st[2] / (n_it * world),
                   bd=st[3] / (n_it * world), gan_g=st[4] / (n_it * world), d_loss=st[5] / ng, d_acc=st[6] / ng,
                   gan_steps=st[7], val=val, sec=time.time() - t0)
        hist.append(rec)
        if main:
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            if lam > 0 and st[7] > 0 and (rec['d_acc'] > 0.95 or rec['d_acc'] < 0.55):
                print(f'!! 警报: D 准确率 {rec["d_acc"]:.2f} (长期 >0.95 或 ≈0.5 要处理)', flush=True)
            json.dump(hist, open(os.path.join(a.out, 'history.json'), 'w'), ensure_ascii=False, indent=1)
            sd = {'model_state_dict': model.state_dict(), 'epoch': ep, 'val': val, 'args': vars(a)}
            if val['rmse'] < best:
                best = val['rmse']; torch.save(sd, os.path.join(a.out, 'best.pth'))
            full = dict(sd, opt=opt.state_dict(), sched=sched.state_dict(), D=D.state_dict(), optD=optD.state_dict(), hist=hist, best=best)
            torch.save(full, lastp + '.tmp'); os.replace(lastp + '.tmp', lastp)       # 原子替换，写到一半被杀不会毁掉上一份
            if ep % a.save_every == a.save_every - 1: torch.save(sd, os.path.join(a.out, f'epoch{ep + 1}.pth'))
    if world > 1: dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--pretrained', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments', 'ostia_pretrain_clean', 'best_model.pth'))
    ap.add_argument('--gpus', type=int, default=8); ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--lr-d', type=float, default=2e-4)
    ap.add_argument('--alpha-grad', type=float, default=0.2); ap.add_argument('--alpha-boundary', type=float, default=0.1)
    ap.add_argument('--lam-final', type=float, default=0.01)
    ap.add_argument('--warm', type=int, default=10); ap.add_argument('--ramp-end', type=int, default=30)
    ap.add_argument('--dil', type=int, default=4); ap.add_argument('--erode', type=int, default=2)
    ap.add_argument('--min-blocks', type=int, default=100)
    ap.add_argument('--val-start', default='2023-04-01'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--port', type=int, default=29611); ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--accum', type=int, default=1, help='梯度累积步数 (有效 batch = bs*gpus*accum)')
    ap.add_argument('--fp16-comm', action='store_true', help='DDP 梯度 fp16 压缩通信')
    ap.add_argument('--store', default=None, help='sst_hourly.h5 路径；给出则训练集在线换洞')
    ap.add_argument('--p-online', type=float, default=1.0, help='在线换洞的概率 (其余用文件里的固定洞)')
    ap.add_argument('--save-every', type=int, default=10, help='每隔多少个 epoch 另存 epochN.pth')
    ap.add_argument('--resume', action='store_true', help='从 --out/last.pth 续训 (含优化器/调度器/D)')
    ap.add_argument('--no-pretrain', action='store_true'); ap.add_argument('--no-c', action='store_true')
    ap.add_argument('--no-film', action='store_true'); ap.add_argument('--no-cbam', action='store_true')
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, 'history.json')) and not a.resume: sys.exit('输出目录已有结果，拒绝覆盖 (续训请加 --resume): ' + a.out)
    world = max(a.gpus, 1)
    if world > 1: mp.spawn(worker, args=(world, a), nprocs=world)
    else: worker(0, 1, a)


if __name__ == '__main__':
    main()
