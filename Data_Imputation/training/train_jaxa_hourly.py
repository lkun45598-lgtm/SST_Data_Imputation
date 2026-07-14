#!/usr/bin/env python3
"""
JAXA Fine-tuning Training Script - 8 GPU DDP
从OSTIA预训练模型继续微调，使用30天输入序列

特点:
1. 加载OSTIA预训练权重
2. 双输入: sst_seq [30, H, W] + mask_seq [30, H, W]
3. Output Composition: 非挖空区域直接用输入，挖空区域用模型预测
4. Loss只在 loss_mask = artificial_mask ∩ original_obs_mask 区域计算
5. 梯度安全的masked loss（乘法而非索引）

作者: Claude Code
日期: 2026-01-20
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
import json
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference.jaxa_inference_dataset import JAXAFinetuneDataset
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal


# ============================================================================
# Loss Functions for JAXA Fine-tuning
# ============================================================================

def masked_mse_loss(pred, target, loss_mask):
    """
    梯度安全的Masked MSE Loss

    只在loss_mask=1的区域计算loss，使用乘法而非索引以保证梯度流

    Args:
        pred: (B, 1, H, W) 预测值
        target: (B, 1, H, W) 真值
        loss_mask: (B, H, W) Loss区域 (1=计算loss, 0=忽略)

    Returns:
        loss: scalar
    """
    mask = loss_mask.unsqueeze(1)  # (B, 1, H, W)
    diff_squared = (pred - target) ** 2
    masked_diff = diff_squared * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss


def masked_gradient_loss(pred, target, loss_mask):
    """
    梯度匹配损失（在loss_mask区域）

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        loss_mask: (B, H, W)

    Returns:
        loss: scalar
    """
    mask = loss_mask.unsqueeze(1)

    # Y方向梯度
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_grad_y = mask[:, :, 1:, :]

    # X方向梯度
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_grad_x = mask[:, :, :, 1:]

    # 计算梯度误差
    grad_loss_y = torch.abs(pred_grad_y - target_grad_y) * mask_grad_y
    grad_loss_x = torch.abs(pred_grad_x - target_grad_x) * mask_grad_x

    total_grad_loss = (grad_loss_y.sum() / (mask_grad_y.sum() + 1e-8) +
                       grad_loss_x.sum() / (mask_grad_x.sum() + 1e-8)) / 2

    return total_grad_loss


def boundary_smoothness_loss(composed, target, mask_seq):
    """
    边界平滑loss：约束mask边界跨越处的预测梯度 ≈ 真实梯度
    只在mask边界像素对（内侧↔外侧）上计算，不影响内部区域

    Args:
        composed: (B, 1, H, W) 输出组合结果（已做output_composition）
        target: (B, 1, H, W) 真值（归一化后）
        mask_seq: (B, 30, H, W) mask序列

    Returns:
        loss: scalar
    """
    last_mask = mask_seq[:, -1:, :, :]  # (B, 1, H, W)

    # X方向：找跨越边界的相邻像素对
    # enter_x: mask外 → mask内的边界像素（左侧是海洋，右侧是洞）
    enter_x = ((last_mask[:, :, :, 1:] == 1) & (last_mask[:, :, :, :-1] == 0)).float()
    # exit_x: mask内 → mask外的边界像素（左侧是洞，右侧是海洋）
    exit_x = ((last_mask[:, :, :, :-1] == 1) & (last_mask[:, :, :, 1:] == 0)).float()
    boundary_x = enter_x + exit_x

    pred_grad_x = composed[:, :, :, 1:] - composed[:, :, :, :-1]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    loss_x = (torch.abs(pred_grad_x - target_grad_x) * boundary_x).sum() / (boundary_x.sum() + 1e-8)

    # Y方向：同理
    enter_y = ((last_mask[:, :, 1:, :] == 1) & (last_mask[:, :, :-1, :] == 0)).float()
    exit_y = ((last_mask[:, :, :-1, :] == 1) & (last_mask[:, :, 1:, :] == 0)).float()
    boundary_y = enter_y + exit_y

    pred_grad_y = composed[:, :, 1:, :] - composed[:, :, :-1, :]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_y = (torch.abs(pred_grad_y - target_grad_y) * boundary_y).sum() / (boundary_y.sum() + 1e-8)

    return (loss_x + loss_y) / 2


def output_composition(pred, sst_seq, mask_seq, land_mask=None):
    """
    输出组合：非挖空区域用输入值，挖空区域用模型预测，陆地区域保持输入

    Args:
        pred: (B, 1, H, W) 模型预测
        sst_seq: (B, 30, H, W) 输入SST序列
        mask_seq: (B, 30, H, W) mask序列
        land_mask: (B, H, W) 陆地掩码，1=陆地，0=海洋

    Returns:
        composed: (B, 1, H, W) 组合后的输出
    """
    last_input = sst_seq[:, -1:, :, :]  # (B, 1, H, W)
    last_mask = mask_seq[:, -1:, :, :]  # (B, 1, H, W)
    # mask=1: 挖空区域用预测, mask=0: 非挖空区域用输入
    composed = last_input * (1 - last_mask) + pred * last_mask

    # 陆地区域强制使用输入值（保持不变）
    if land_mask is not None:
        land_mask_expanded = land_mask.unsqueeze(1)  # (B, 1, H, W)
        composed = composed * (1 - land_mask_expanded) + last_input * land_mask_expanded

    return composed


def jaxa_combined_loss(pred, target, loss_mask, sst_seq=None, mask_seq=None,
                       alpha_mse=1.0, alpha_grad=0.2, alpha_temporal=0.0, alpha_boundary=0.1):
    """
    JAXA微调的组合损失（使用output composition后简化版）

    Args:
        pred: (B, 1, H, W) 预测值（已经过output composition）
        target: (B, 1, H, W) 真值（归一化后）
        loss_mask: (B, H, W) Loss区域 = artificial_mask ∩ original_obs_mask
        sst_seq: (B, 30, H, W) 输入SST序列（用于时间连续性）
        mask_seq: (B, 30, H, W) mask序列（用于边界平滑约束）
        alpha_mse: MSE权重（挖空区域重建）
        alpha_grad: 梯度loss权重
        alpha_temporal: 时间连续性权重
        alpha_boundary: 边界平滑loss权重

    Returns:
        total_loss, loss_mse, loss_grad, loss_boundary
    """
    # MSE Loss (在挖空区域 - 重建缺失数据)
    loss_mse = masked_mse_loss(pred, target, loss_mask)

    # 梯度Loss (在挖空区域)
    if alpha_grad > 0:
        loss_grad = masked_gradient_loss(pred, target, loss_mask)
    else:
        loss_grad = torch.tensor(0.0, device=pred.device)

    # 时间连续性约束 (可选)
    # 注意：使用target(ground_truth)而不是被挖空的last_day来计算expected_change
    if sst_seq is not None and alpha_temporal > 0:
        prev_day = sst_seq[:, -2:-1, :, :]  # 第29天（真实数据）
        # expected_change应该用target（真值）而不是被挖空的输入
        expected_change = target - prev_day  # 真实的day30 - day29
        predicted_change = pred - prev_day   # 预测的day30 - day29
        temporal_penalty = F.relu(torch.abs(predicted_change - expected_change) - 0.5)
        loss_temporal = (temporal_penalty * loss_mask.unsqueeze(1)).sum() / (loss_mask.sum() + 1e-8)
    else:
        loss_temporal = torch.tensor(0.0, device=pred.device)

    # 边界平滑约束 (可选)
    if mask_seq is not None and alpha_boundary > 0:
        loss_boundary = boundary_smoothness_loss(pred, target, mask_seq)
    else:
        loss_boundary = torch.tensor(0.0, device=pred.device)

    total_loss = alpha_mse * loss_mse + alpha_grad * loss_grad + alpha_temporal * loss_temporal + alpha_boundary * loss_boundary

    return total_loss, loss_mse, loss_grad, loss_boundary


# ============================================================================
# DDP Setup
# ============================================================================

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29509'  # 换一个端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, rank, norm_mean, norm_std):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_grad = 0.0
    total_mae = 0.0
    n_samples = 0

    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    else:
        pbar = train_loader

    for batch in pbar:
        sst_seq = batch['input_sst_seq'].to(device).float()  # [B, 30, H, W]
        mask_seq = batch['mask_seq'].to(device).float()  # [B, 30, H, W]
        gt_sst = batch['ground_truth_sst'].to(device).unsqueeze(1).float()  # [B, 1, H, W]
        loss_mask = batch['loss_mask'].to(device).float()  # [B, H, W]
        land_mask = batch['land_mask'].to(device).float()  # [B, H, W]

        # Forward
        pred = model(sst_seq, mask_seq)

        # Output Composition: 非挖空区域用输入，挖空区域用预测，陆地保持不变
        pred = output_composition(pred, sst_seq, mask_seq, land_mask)

        # Loss (只在挖空区域计算)
        loss, loss_mse, loss_grad, loss_boundary = jaxa_combined_loss(
            pred, gt_sst, loss_mask,
            sst_seq=sst_seq,
            mask_seq=mask_seq,
            alpha_mse=1.0,
            alpha_grad=0.2,
            alpha_temporal=0.0,   # temporal loss excluded (ablation: no benefit)
            alpha_boundary=0.1    # boundary loss (deployed objective)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            # 反归一化计算MAE (只在挖空区域)
            pred_kelvin = pred * norm_std + norm_mean
            gt_kelvin = gt_sst * norm_std + norm_mean
            mask_expanded = loss_mask.unsqueeze(1)
            mae = (torch.abs(pred_kelvin - gt_kelvin) * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

            bs = sst_seq.size(0)
            total_loss += loss.item() * bs
            total_mse += loss_mse.item() * bs
            total_grad += loss_grad.item() * bs
            total_mae += mae.item() * bs
            n_samples += bs

            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/n_samples:.4f}',
                    'MAE': f'{total_mae/n_samples:.3f}K',
                    'boundary': f'{loss_boundary.item():.4f}'
                })

    # 同步所有进程
    metrics_tensor = torch.tensor([total_loss, total_mse, total_grad, total_mae, n_samples], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_loss, total_mse, total_grad, total_mae, n_samples = metrics_tensor.tolist()

    return {
        'loss': total_loss / n_samples,
        'mse': total_mse / n_samples,
        'grad': total_grad / n_samples,
        'mae': total_mae / n_samples
    }


def valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_samples = 0

    if rank == 0:
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]")
    else:
        pbar = valid_loader

    with torch.no_grad():
        for batch in pbar:
            sst_seq = batch['input_sst_seq'].to(device).float()
            mask_seq = batch['mask_seq'].to(device).float()
            gt_sst = batch['ground_truth_sst'].to(device).unsqueeze(1).float()
            loss_mask = batch['loss_mask'].to(device).float()
            land_mask = batch['land_mask'].to(device).float()

            pred = model(sst_seq, mask_seq)

            # Output Composition: 非挖空区域用输入，挖空区域用预测，陆地保持不变
            pred = output_composition(pred, sst_seq, mask_seq, land_mask)

            # Loss (只在挖空区域计算)
            loss, loss_mse, _, _ = jaxa_combined_loss(
                pred, gt_sst, loss_mask,
                sst_seq=sst_seq,
                mask_seq=mask_seq,
                alpha_grad=0,  # 验证时不计算梯度loss
                alpha_boundary=0  # 验证时不计算边界loss
            )

            # 反归一化
            pred_kelvin = pred * norm_std + norm_mean
            gt_kelvin = gt_sst * norm_std + norm_mean
            mask_expanded = loss_mask.unsqueeze(1)

            error = torch.abs(pred_kelvin - gt_kelvin) * mask_expanded
            mae = error.sum() / (mask_expanded.sum() + 1e-8)
            rmse = torch.sqrt(((pred_kelvin - gt_kelvin)**2 * mask_expanded).sum() / (mask_expanded.sum() + 1e-8))

            bs = sst_seq.size(0)
            total_loss += loss.item() * bs
            total_mse += loss_mse.item() * bs
            total_mae += mae.item() * bs
            total_rmse += rmse.item() * bs
            n_samples += bs

    metrics_tensor = torch.tensor([total_loss, total_mse, total_mae, total_rmse, n_samples], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_loss, total_mse, total_mae, total_rmse, n_samples = metrics_tensor.tolist()

    return {
        'loss': total_loss / n_samples,
        'mse': total_mse / n_samples,
        'mae': total_mae / n_samples,
        'rmse': total_rmse / n_samples
    }


# ============================================================================
# Main Training Worker
# ============================================================================

def train_worker(rank, world_size, config):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    # 配置
    data_dir = config['data_dir']
    save_dir = config['save_dir']
    pretrained_path = config['pretrained_path']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = config['lr']

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print("\n" + "="*80)
        print(f"JAXA Hourly Fine-tuning Training (4 GPU DDP) - Hour {config['hour']:02d}")
        print("="*80)
        print(f"\n配置:")
        print(f"  - 数据目录: {data_dir}")
        print(f"  - 基座模型(OSTIA原始): {pretrained_path}")
        print(f"  - 保存目录: {save_dir}")
        print(f"  - Batch size: {batch_size} per GPU × {world_size} GPUs = {batch_size*world_size}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - 输入: 30天序列 (60通道: 30×SST + 30×mask)")
        print(f"  - Loss: 只在 artificial_mask ∩ original_obs_mask 区域计算")
        print()

    # 数据集
    # 使用序列0-7做训练（8年），序列8做验证（1年）
    # 重要: 使用OSTIA预训练模型的归一化参数，确保输入分布一致
    ostia_mean = 299.9221  # OSTIA训练时的mean (Kelvin)
    ostia_std = 2.6919     # OSTIA训练时的std (Kelvin)

    mask_mode = config.get('mask_mode', 'square')
    ratio_range = config.get('ratio_range', None)

    train_dataset = JAXAFinetuneDataset(
        data_dir=data_dir,
        series_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # 8年训练数据
        window_size=30,
        mask_ratio=0.2,
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=ostia_mean,   # 使用OSTIA归一化参数
        std=ostia_std,     # 使用OSTIA归一化参数
        cache_size=100,
        seed=42,
        mask_mode=mask_mode,
        ratio_range=ratio_range,
    )

    valid_dataset = JAXAFinetuneDataset(
        data_dir=data_dir,
        series_ids=[8],  # 1年验证数据
        window_size=30,
        mask_ratio=0.2,
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=ostia_mean,   # 使用OSTIA归一化参数
        std=ostia_std,     # 使用OSTIA归一化参数
        cache_size=50,
        seed=123,  # 不同seed确保验证集挖空不同
        mask_mode=mask_mode,
        ratio_range=ratio_range,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=2, pin_memory=True, prefetch_factor=2
    )

    if rank == 0:
        print(f"数据:")
        print(f"  - Train: {len(train_dataset)} samples (序列 0-7, 8年)")
        print(f"  - Valid: {len(valid_dataset)} samples (序列 8, 1年)")
        print(f"  - Normalize: mean={train_dataset.mean:.2f}K, std={train_dataset.std:.2f}K")

    # 模型
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6).to(device)

    # 加载预训练权重
    if pretrained_path and os.path.exists(pretrained_path):
        if rank == 0:
            print(f"\n加载预训练模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if rank == 0:
            print("  ✓ 预训练权重加载成功")
    else:
        if rank == 0:
            print("\n⚠️ 未找到预训练模型，从头开始训练")

    model = DDP(model, device_ids=[rank])

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数量: {total_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 训练
    best_val_mae = float('inf')
    norm_mean, norm_std = train_dataset.mean, train_dataset.std
    history = {'train': [], 'valid': []}

    if rank == 0:
        print(f"\n开始训练 ({num_epochs} epochs)...\n")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, rank, norm_mean, norm_std)
        valid_metrics = valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std)

        scheduler.step()

        if rank == 0:
            history['train'].append(train_metrics)
            history['valid'].append(valid_metrics)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.3f}K")
            print(f"  Valid - Loss: {valid_metrics['loss']:.4f} | MAE: {valid_metrics['mae']:.3f}K | RMSE: {valid_metrics['rmse']:.3f}K")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            if valid_metrics['mae'] < best_val_mae:
                best_val_mae = valid_metrics['mae']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': best_val_mae,
                    'valid_metrics': valid_metrics,
                    'norm_mean': norm_mean,
                    'norm_std': norm_std
                }, f'{save_dir}/best_model.pth')
                print(f"  ✓ 最优模型已保存! MAE: {best_val_mae:.3f}K")

            # 定期保存checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mae': valid_metrics['mae'],
                    'norm_mean': norm_mean,
                    'norm_std': norm_std
                }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')

    if rank == 0:
        # 保存最终模型
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }, f'{save_dir}/final_model.pth')

        # 保存训练历史
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # 保存配置
        with open(f'{save_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print("\n" + "="*80)
        print(f"训练完成! 最优MAE: {best_val_mae:.3f}K")
        print(f"模型保存位置: {save_dir}")
        print("="*80)

    cleanup_distributed()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', type=int, required=True, help='目标微调的小时 (1-23)')
    parser.add_argument('--epochs', type=int, default=50, help='微调轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='每卡batch size')
    parser.add_argument('--gpus', type=int, default=8, help='DDP GPU 数')
    parser.add_argument('--mask-mode', choices=['square', 'realcloud'], default='square',
                        help='人工挖空方式: square(旧) 或 realcloud(真实云形状,只落观测区)')
    parser.add_argument('--ratio-lo', type=float, default=0.2,
                        help='realcloud 挖空占"观测区"的比例下界')
    parser.add_argument('--ratio-hi', type=float, default=0.3,
                        help='realcloud 挖空占"观测区"的比例上界(观测区本就只~20-30%,不宜遮太多)')
    parser.add_argument('--save-suffix', type=str, default=None,
                        help='保存目录后缀(不指定时 realcloud 自动加 _realmask,避免覆盖现有模型)')
    args = parser.parse_args()

    # 配置
    hour_str = f"{args.hour:02d}"
    suffix = args.save_suffix
    if suffix is None:
        suffix = '_realmask' if args.mask_mode == 'realcloud' else ''
    config = {
        'data_dir': f'/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h{hour_str}',
        'save_dir': f'/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h{hour_str}{suffix}',
        'pretrained_path': '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/ostia_pretrain/best_model.pth',
        'batch_size': args.batch_size,  # per GPU
        'num_epochs': args.epochs,
        'lr': 5e-4,  # 微调学习率
        'hour': args.hour,
        'mask_mode': args.mask_mode,
        'ratio_range': (args.ratio_lo, args.ratio_hi),
    }

    world_size = args.gpus  # DDP GPU 数 (8卡)
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
