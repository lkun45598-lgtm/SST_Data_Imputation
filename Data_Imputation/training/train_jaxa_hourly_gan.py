#!/usr/bin/env python3
"""
JAXA 微调 + PatchGAN 对抗训练 (单小时验证版)

以 train_jaxa_hourly.py 为蓝本: 模型结构、数据集、重建损失、验证流程全部复用，
只额外加入判别器和对抗损失。--lambda-adv 0 即为不带 GAN 的对照 (同一份代码、同一划分)。

要点:
1. 判别器只评价"感受野全部落在真实观测海洋内"的位置，真/假块都如此;
   假块还要求与人工挖洞区重叠。填充值不会进入判别器的监督。
2. 判别器梯度手动 all-reduce (不包 DDP)，生成器步里冻结判别器参数。
3. 前 warmup_epochs 个 epoch 只做重建 (判别器照常训练，生成器不受对抗项影响)。
4. 默认 训练 0-6 / 验证 7 / 测试 8 (与单模型方案一致)。
5. 新输出目录; 已存在 best_model.pth 时拒绝运行，不覆盖任何已有产物。
"""

import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler, Subset
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference.jaxa_inference_dataset import JAXAFinetuneDataset
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from models.patchgan_discriminator import (
    PatchDiscriminator, full_valid, any_overlap, masked_mean, d_loss_lsgan, g_loss_lsgan)
from datasets.lowmiss_real_frames import LowMissRealFrames
from training.train_jaxa_hourly import (
    output_composition, jaxa_combined_loss, valid_epoch, setup_distributed, cleanup_distributed)


def sync_from_rank0(module):
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, 0)


def allreduce_grads(module, world_size):
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad /= world_size


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def train_epoch_gan(model, D, opt_G, opt_D, train_loader, make_real_loader, device, epoch, rank,
                    world_size, norm_mean, norm_std, cfg):
    model.train()
    D.train()
    lam = cfg['lambda_adv']
    adv_on = lam > 0
    adv_active = adv_on and epoch >= cfg['warmup_epochs']
    real_iter = iter(make_real_loader(epoch)) if adv_on else None

    keys = ['loss', 'mse', 'grad', 'mae', 'd_loss', 'adv', 'd_real', 'd_fake', 'n']
    acc = {k: 0.0 for k in keys}
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") if rank == 0 else train_loader

    for step, batch in enumerate(pbar):
        if cfg.get('max_steps') and step >= cfg['max_steps']:
            break
        sst_seq = batch['input_sst_seq'].to(device).float()
        mask_seq = batch['mask_seq'].to(device).float()
        gt_sst = batch['ground_truth_sst'].to(device).unsqueeze(1).float()
        loss_mask = batch['loss_mask'].to(device).float()
        original_obs_mask = batch['original_obs_mask'].to(device).float()
        land_mask = batch['land_mask'].to(device).float()
        valid_ref_mask = original_obs_mask * (1 - land_mask)

        pred = model(sst_seq, mask_seq)
        composed = output_composition(pred, sst_seq, mask_seq, land_mask)

        loss_rec, loss_mse, loss_grad, loss_boundary = jaxa_combined_loss(
            composed, gt_sst, loss_mask, sst_seq=sst_seq, mask_seq=mask_seq,
            valid_ref_mask=valid_ref_mask, alpha_mse=1.0, alpha_grad=cfg['alpha_grad'],
            alpha_temporal=0.0, alpha_boundary=cfg['alpha_boundary'])

        d_loss = adv = d_real = d_fake = torch.zeros((), device=device)
        loss_adv = None
        if adv_on:
            # 假块: 感受野全在真实观测海洋内，且含人工挖洞像素
            fake_w = (full_valid(D.patch_fraction(valid_ref_mask.unsqueeze(1)))
                      * any_overlap(D.patch_fraction(loss_mask.unsqueeze(1))))

            try:
                real = next(real_iter)
            except StopIteration:
                real_iter = iter(make_real_loader(epoch))
                real = next(real_iter)
            real_x = real['sst'].to(device).float()
            real_w = full_valid(D.patch_fraction(real['valid'].to(device).float()))

            # 判别器步
            opt_D.zero_grad()
            s_real = D(real_x)
            s_fake = D(composed.detach())
            d_loss = d_loss_lsgan(s_real, real_w, s_fake, fake_w)
            d_loss.backward()
            allreduce_grads(D, world_size)
            opt_D.step()
            d_real = masked_mean(s_real.detach(), real_w)
            d_fake = masked_mean(s_fake.detach(), fake_w)

            if adv_active:
                set_requires_grad(D, False)
                loss_adv = g_loss_lsgan(D(composed), fake_w)
                set_requires_grad(D, True)
                adv = loss_adv.detach()

        total = loss_rec + lam * loss_adv if (adv_active and loss_adv is not None) else loss_rec

        # 定期记录对抗/重建梯度范数比 (输出层权重)，用于调 lambda
        if rank == 0 and adv_active and cfg['log_ratio_every'] and step % cfg['log_ratio_every'] == 0:
            try:
                w = model.module.fc2.weight
                g_rec = torch.autograd.grad(loss_rec, w, retain_graph=True)[0].norm()
                g_adv = torch.autograd.grad(lam * loss_adv, w, retain_graph=True)[0].norm()
                print(f"\n  [step {step}] grad-norm  rec={g_rec.item():.3e}  lam*adv={g_adv.item():.3e}"
                      f"  ratio={(g_adv / (g_rec + 1e-12)).item():.3f}")
            except RuntimeError as e:
                print(f"\n  [step {step}] grad-ratio skipped: {e}")

        opt_G.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_G.step()

        with torch.no_grad():
            pred_k = composed * norm_std + norm_mean
            gt_k = gt_sst * norm_std + norm_mean
            m = loss_mask.unsqueeze(1)
            mae = (torch.abs(pred_k - gt_k) * m).sum() / (m.sum() + 1e-8)
            bs = sst_seq.size(0)
            for k, v in [('loss', total), ('mse', loss_mse), ('grad', loss_grad), ('mae', mae),
                         ('d_loss', d_loss), ('adv', adv), ('d_real', d_real), ('d_fake', d_fake)]:
                acc[k] += float(v) * bs
            acc['n'] += bs
            if rank == 0:
                pbar.set_postfix({'loss': f"{acc['loss']/acc['n']:.4f}", 'MAE': f"{acc['mae']/acc['n']:.3f}K",
                                  'D': f"{acc['d_loss']/acc['n']:.3f}"})

    t = torch.tensor([acc[k] for k in keys], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vals = dict(zip(keys, t.tolist()))
    n = vals.pop('n')
    return {k: v / n for k, v in vals.items()}


def train_worker(rank, world_size, config):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    save_dir = config['save_dir']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    train_ids, valid_ids = config['train_series'], config['valid_series']
    ostia_mean, ostia_std = 299.9221, 2.6919  # 与 OSTIA 预训练一致

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"JAXA Hourly + PatchGAN - Hour {config['hour']:02d}  lambda_adv={config['lambda_adv']}")
        print("=" * 80)
        print(json.dumps({k: v for k, v in config.items()}, indent=2, default=str))

    ds_kwargs = dict(window_size=30, mask_ratio=0.2, min_mask_size=10, max_mask_size=50,
                     normalize=True, mean=ostia_mean, std=ostia_std,
                     mask_mode=config['mask_mode'], ratio_range=config['ratio_range'])
    train_dataset = JAXAFinetuneDataset(config['data_dir'], series_ids=train_ids,
                                        cache_size=100, seed=42, **ds_kwargs)
    valid_dataset = JAXAFinetuneDataset(config['data_dir'], series_ids=valid_ids,
                                        cache_size=50, seed=123, **ds_kwargs)
    if config['smoke']:
        train_dataset = Subset(train_dataset, range(0, 40, 5))
        valid_dataset = Subset(valid_dataset, range(0, 8, 2))
        train_dataset.mean, train_dataset.std = ostia_mean, ostia_std

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    nw = 0 if config['smoke'] else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=nw, pin_memory=True, prefetch_factor=2 if nw else None)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=min(nw, 2), pin_memory=True,
                              prefetch_factor=2 if nw else None)

    # 判别器真样本: 只取训练序列里的低缺失率帧
    make_real_loader = None
    if config['lambda_adv'] > 0:
        real_ds = LowMissRealFrames(config['data_dir'], train_ids, config['min_ocean_cov'],
                                    ostia_mean, ostia_std, month_balance=config['month_balance'])
        n_per_epoch = len(train_loader) * batch_size

        def make_real_loader(epoch):
            g = torch.Generator().manual_seed(1000 * epoch + rank)
            sampler = WeightedRandomSampler(real_ds.weights, n_per_epoch, replacement=True, generator=g)
            return DataLoader(real_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=min(nw, 2), pin_memory=True)

    # 生成器
    if config['smoke']:
        model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=8, modes2=8, width=16, depth=1,
                                      use_cbam=True).to(device)
    else:
        model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
                                      use_cbam=True).to(device)
        if config['no_pretrain']:
            if rank == 0:
                print("从零训练 (--no-pretrain)")
        else:
            ckpt = torch.load(config['pretrained_path'], map_location=device)
            missing, unexpected = model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            if rank == 0:
                print(f"加载预训练 {config['pretrained_path']}  missing={len(missing)} unexpected={len(unexpected)}")
    model = DDP(model, device_ids=[rank])

    opt_G = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=num_epochs, eta_min=1e-6)

    torch.manual_seed(1234)
    D = PatchDiscriminator(in_ch=1, base=32, n_down=config['d_down']).to(device)
    sync_from_rank0(D)
    opt_D = torch.optim.Adam(D.parameters(), lr=config['lr_d'], betas=(0.5, 0.999))
    torch.manual_seed(42 + rank)

    norm_mean, norm_std = ostia_mean, ostia_std
    best_val_mae = float('inf')
    history = {'train': [], 'valid': []}

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        tm = train_epoch_gan(model, D, opt_G, opt_D, train_loader, make_real_loader, device, epoch,
                             rank, world_size, norm_mean, norm_std, config)
        vm = valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std)
        sched.step()

        if rank == 0:
            history['train'].append(tm)
            history['valid'].append(vm)
            print(f"\nEpoch {epoch+1}/{num_epochs}  adv={'ON' if config['lambda_adv'] > 0 and epoch >= config['warmup_epochs'] else 'off'}")
            print(f"  Train - Loss {tm['loss']:.4f} | MAE {tm['mae']:.3f}K | D {tm['d_loss']:.4f} "
                  f"| D(real) {tm['d_real']:.3f} D(fake) {tm['d_fake']:.3f} | adv {tm['adv']:.4f}")
            print(f"  Valid - MAE {vm['mae']:.3f}K | RMSE {vm['rmse']:.3f}K")

            state = {'epoch': epoch, 'model_state_dict': model.module.state_dict(),
                     'norm_mean': norm_mean, 'norm_std': norm_std}
            if vm['mae'] < best_val_mae:
                best_val_mae = vm['mae']
                torch.save({**state, 'val_mae': best_val_mae, 'valid_metrics': vm},
                           f'{save_dir}/best_model.pth')
                print(f"  ✓ best saved, MAE {best_val_mae:.3f}K")
            if (epoch + 1) % 10 == 0:
                torch.save({**state, 'D_state_dict': D.state_dict(), 'val_mae': vm['mae']},
                           f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')
            with open(f'{save_dir}/training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

    if rank == 0:
        torch.save({'epoch': num_epochs - 1, 'model_state_dict': model.module.state_dict(),
                    'D_state_dict': D.state_dict(), 'norm_mean': norm_mean, 'norm_std': norm_std},
                   f'{save_dir}/final_model.pth')
        with open(f'{save_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"\n训练完成，最优验证 MAE {best_val_mae:.3f}K，保存于 {save_dir}")

    cleanup_distributed()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hour', type=int, required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=2, help='每卡')
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--train-series', type=str, default='0,1,2,3,4,5,6')
    p.add_argument('--valid-series', type=str, default='7')
    p.add_argument('--mask-mode', choices=['square', 'realcloud'], default='realcloud')
    p.add_argument('--ratio-lo', type=float, default=0.2)
    p.add_argument('--ratio-hi', type=float, default=0.3)
    p.add_argument('--alpha-grad', type=float, default=0.2)
    p.add_argument('--alpha-boundary', type=float, default=0.1)
    p.add_argument('--lambda-adv', type=float, default=0.01, help='0 = 不带GAN的对照')
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--min-ocean-cov', type=float, default=0.7)
    p.add_argument('--month-balance', action='store_true', help='真样本按月份均衡抽样 (默认按帧均匀)')
    p.add_argument('--d-down', type=int, default=2, help='判别器 stride-2 层数: 2=感受野18, 3=42')
    p.add_argument('--lr-d', type=float, default=1e-4)
    p.add_argument('--log-ratio-every', type=int, default=200)
    p.add_argument('--pretrained', type=str, default='experiments/ostia_pretrain_clean/best_model.pth')
    p.add_argument('--no-pretrain', action='store_true')
    p.add_argument('--save-dir', type=str, default=None, help='默认 experiments/gan_v1_hXX_lamYY')
    p.add_argument('--smoke', action='store_true', help='冒烟测试: 小模型、少量样本、存到 /tmp')
    p.add_argument('--max-steps', type=int, default=0)
    a = p.parse_args()

    repo = PROJECT_ROOT
    hh = f'{a.hour:02d}'
    save_dir = a.save_dir or f'{repo}/experiments/gan_v1_h{hh}_lam{a.lambda_adv:g}'
    if a.smoke:
        save_dir = a.save_dir or '/tmp/gan_smoke'
    if os.path.exists(f'{save_dir}/best_model.pth'):
        sys.exit(f'拒绝运行: {save_dir}/best_model.pth 已存在，请换 --save-dir')

    config = {
        'data_dir': f'{repo}/experiments/hourly_data/h{hh}',
        'save_dir': save_dir,
        'pretrained_path': f'{repo}/{a.pretrained}',
        'no_pretrain': a.no_pretrain,
        'batch_size': a.batch_size, 'num_epochs': a.epochs, 'lr': 5e-4, 'lr_d': a.lr_d,
        'hour': a.hour,
        'train_series': [int(x) for x in a.train_series.split(',')],
        'valid_series': [int(x) for x in a.valid_series.split(',')],
        'mask_mode': a.mask_mode, 'ratio_range': (a.ratio_lo, a.ratio_hi),
        'alpha_grad': a.alpha_grad, 'alpha_boundary': a.alpha_boundary,
        'lambda_adv': a.lambda_adv, 'warmup_epochs': a.warmup_epochs,
        'min_ocean_cov': a.min_ocean_cov, 'month_balance': a.month_balance, 'd_down': a.d_down,
        'log_ratio_every': a.log_ratio_every, 'smoke': a.smoke, 'max_steps': a.max_steps,
    }
    mp.spawn(train_worker, args=(a.gpus, config), nprocs=a.gpus, join=True)


if __name__ == '__main__':
    main()
