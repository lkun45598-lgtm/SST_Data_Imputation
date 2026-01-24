"""
FNO_CBAM Temporal Training - 8 GPU DDP (Output Composition版)
基于原始成功训练脚本，添加输出组合功能

核心改进：
  - 输出组合：final = input*(1-mask) + pred*mask
  - 观测区域直接使用输入值，模型只负责填充缺失区域
  - 移除observed loss（因为输出组合已保证观测区域一致）

其他配置与原始版本完全一致（已验证可正常训练）
"""
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
import json

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.ostia_dataset import SSTDatasetTemporal
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from losses.temporal_loss import combined_loss_temporal


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29520'  # 新端口
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def output_composition(pred, sst_seq, mask_seq):
    """
    输出组合：观测区域用输入值，缺失区域用模型预测

    final = input * (1 - mask) + pred * mask

    Args:
        pred: (B, 1, H, W) 模型预测
        sst_seq: (B, 30, H, W) 输入SST序列
        mask_seq: (B, 30, H, W) mask序列 (1=缺失, 0=观测)

    Returns:
        composed: (B, 1, H, W) 组合后的输出
    """
    last_input = sst_seq[:, -1:, :, :]  # (B, 1, H, W) 最后一天的输入
    last_mask = mask_seq[:, -1:, :, :]  # (B, 1, H, W) 最后一天的mask

    # 观测区域(mask=0)用输入，缺失区域(mask=1)用预测
    composed = last_input * (1 - last_mask) + pred * last_mask

    return composed


def train_epoch(model, train_loader, optimizer, device, epoch, rank, norm_mean, norm_std):
    model.train()
    total_loss, total_missing, total_observed, total_grad = 0.0, 0.0, 0.0, 0.0
    total_mae = 0.0
    n_samples = 0

    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    else:
        pbar = train_loader

    for batch in pbar:
        sst_seq = batch['input_sst_seq'].to(device).float()
        mask_seq = batch['mask_seq'].to(device).float()
        gt_sst = batch['ground_truth_sst'].to(device).unsqueeze(1).float()
        land_mask = batch['land_mask'].to(device).float()
        missing_mask = batch['missing_mask'].to(device).float()
        ocean_mask = 1 - land_mask

        pred = model(sst_seq, mask_seq)

        # 【关键改进】输出组合：观测区域直接用输入值
        pred = output_composition(pred, sst_seq, mask_seq)

        # 移除observed loss（因为输出组合已保证观测区域一致）
        loss, loss_missing, loss_observed, loss_grad = combined_loss_temporal(
            pred, gt_sst, missing_mask, ocean_mask,
            sst_seq=sst_seq,
            alpha_missing=1.2,
            alpha_observed=0.0,  # 【改动】不需要observed loss
            gamma=0.2,
            beta_temporal=0.15, beta_range=0.01
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred_celsius = pred * norm_std + norm_mean
            gt_celsius = gt_sst * norm_std + norm_mean
            missing_ocean_mask = (missing_mask * ocean_mask).unsqueeze(1)
            mae = (torch.abs(pred_celsius - gt_celsius) * missing_ocean_mask).sum() / (missing_ocean_mask.sum() + 1e-8)

            bs = sst_seq.size(0)
            total_loss += loss.item() * bs
            total_missing += loss_missing.item() * bs
            total_observed += loss_observed.item() * bs
            total_grad += loss_grad.item() * bs
            total_mae += mae.item() * bs
            n_samples += bs

            if rank == 0:
                pbar.set_postfix({'loss': f'{total_loss/n_samples:.4f}', 'MAE': f'{total_mae/n_samples:.3f}°C'})

    metrics_tensor = torch.tensor([total_loss, total_missing, total_observed, total_grad, total_mae, n_samples],
                                  device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_loss, total_missing, total_observed, total_grad, total_mae, n_samples = metrics_tensor.tolist()

    return {
        'loss': total_loss / n_samples,
        'loss_missing': total_missing / n_samples,
        'loss_observed': total_observed / n_samples,
        'loss_grad': total_grad / n_samples,
        'mae': total_mae / n_samples
    }


def valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std):
    model.eval()
    total_loss, total_missing, total_observed, total_grad = 0.0, 0.0, 0.0, 0.0
    total_mae, total_rmse, total_max = 0.0, 0.0, 0.0
    # 分别统计缺失区域和观测区域的MAE
    total_mae_missing, total_mae_observed = 0.0, 0.0
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
            land_mask = batch['land_mask'].to(device).float()
            missing_mask = batch['missing_mask'].to(device).float()
            ocean_mask = 1 - land_mask

            pred = model(sst_seq, mask_seq)

            # 【关键改进】输出组合
            pred = output_composition(pred, sst_seq, mask_seq)

            loss, loss_missing, loss_observed, loss_grad = combined_loss_temporal(
                pred, gt_sst, missing_mask, ocean_mask,
                sst_seq=sst_seq,
                alpha_missing=1.2,
                alpha_observed=0.0,  # 【改动】不需要observed loss
                gamma=0.2,
                beta_temporal=0.15, beta_range=0.01
            )

            pred_celsius = pred * norm_std + norm_mean
            gt_celsius = gt_sst * norm_std + norm_mean

            # 缺失区域误差
            missing_ocean_mask = (missing_mask * ocean_mask).unsqueeze(1)
            error_missing = torch.abs(pred_celsius - gt_celsius) * missing_ocean_mask
            mae_missing = error_missing.sum() / (missing_ocean_mask.sum() + 1e-8)
            rmse = torch.sqrt(((pred_celsius - gt_celsius)**2 * missing_ocean_mask).sum() / (missing_ocean_mask.sum() + 1e-8))
            max_error = error_missing.max()

            # 观测区域误差（使用输出组合后应该接近0）
            observed_ocean_mask = ((1 - missing_mask) * ocean_mask).unsqueeze(1)
            error_observed = torch.abs(pred_celsius - gt_celsius) * observed_ocean_mask
            mae_observed = error_observed.sum() / (observed_ocean_mask.sum() + 1e-8)

            bs = sst_seq.size(0)
            total_loss += loss.item() * bs
            total_missing += loss_missing.item() * bs
            total_observed += loss_observed.item() * bs
            total_grad += loss_grad.item() * bs
            total_mae += mae_missing.item() * bs
            total_rmse += rmse.item() * bs
            total_max += max_error.item() * bs
            total_mae_missing += mae_missing.item() * bs
            total_mae_observed += mae_observed.item() * bs
            n_samples += bs

    metrics_tensor = torch.tensor([total_loss, total_missing, total_observed, total_grad, total_mae, total_rmse, total_max,
                                   total_mae_missing, total_mae_observed, n_samples], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    (total_loss, total_missing, total_observed, total_grad, total_mae, total_rmse, total_max,
     total_mae_missing, total_mae_observed, n_samples) = metrics_tensor.tolist()

    return {
        'loss': total_loss / n_samples,
        'loss_missing': total_missing / n_samples,
        'loss_observed': total_observed / n_samples,
        'loss_grad': total_grad / n_samples,
        'mae': total_mae / n_samples,
        'rmse': total_rmse / n_samples,
        'max_error': total_max / n_samples,
        'mae_missing': total_mae_missing / n_samples,
        'mae_observed': total_mae_observed / n_samples
    }


def train_worker(rank, world_size):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    # 数据路径
    data_dir = '/data/sst_data/sst_missing_value_imputation/processed_data'
    save_dir = '/home/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/temporal_30days_composition'
    batch_size = 4  # per GPU (总batch=32)

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print("\n" + "="*80)
        print("FNO_CBAM Temporal Training (8 GPU DDP) - Output Composition版")
        print("="*80)
        print(f"\n【核心改进】输出组合:")
        print(f"  final = input * (1-mask) + pred * mask")
        print(f"  → 观测区域直接使用输入值（保证一致性）")
        print(f"  → 模型只负责填充缺失区域")
        print(f"  → 移除observed loss（不再需要）")
        print(f"\n配置:")
        print(f"  - 数据路径: {data_dir}")
        print(f"  - 输入: 30天序列 (60通道: 30×SST + 30×mask)")
        print(f"  - Loss: 1.2×缺失 + 0.0×观测 + 0.2×梯度 + 0.15×时间连续性 + 0.01×温度范围")
        print(f"  - 优化器: AdamW (lr=1e-3, weight_decay=1e-4)")
        print(f"  - 调度器: StepLR (step_size=15, gamma=0.5)")
        print(f"  - Epochs: 60")
        print(f"  - Batch: {batch_size}×{world_size} = {batch_size*world_size}")
        print(f"  - 保存目录: {save_dir}\n")

    # 数据
    train_dataset = SSTDatasetTemporal(hdf5_path=f'{data_dir}/processed_sst_train.h5', normalize=True)
    valid_dataset = SSTDatasetTemporal(hdf5_path=f'{data_dir}/processed_sst_valid.h5', normalize=True,
                                      mean=train_dataset.mean, std=train_dataset.std)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                             num_workers=16, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                             num_workers=16, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    if rank == 0:
        print(f"数据:")
        print(f"  - Train: {len(train_dataset)} samples ({batch_size*world_size} effective batch)")
        print(f"  - Valid: {len(valid_dataset)} samples")

    # 模型
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6).to(device)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数量: {total_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 训练
    num_epochs = 60
    best_val_mae = float('inf')
    norm_mean, norm_std = train_dataset.mean, train_dataset.std
    history = {'train': [], 'valid': []}

    if rank == 0:
        print(f"\n开始训练 ({num_epochs} epochs)...\n")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, rank, norm_mean, norm_std)
        valid_metrics = valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std)

        if rank == 0:
            history['train'].append(train_metrics)
            history['valid'].append(valid_metrics)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.3f}°C")
            print(f"    ├─ Missing: {train_metrics['loss_missing']:.4f} | Grad: {train_metrics['loss_grad']:.4f}")
            print(f"  Valid - Loss: {valid_metrics['loss']:.4f}")
            print(f"    ├─ MAE(缺失): {valid_metrics['mae_missing']:.3f}°C | MAE(观测): {valid_metrics['mae_observed']:.4f}°C")
            print(f"    ├─ RMSE: {valid_metrics['rmse']:.3f}°C | Max: {valid_metrics['max_error']:.3f}°C")
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
                print(f"  ✓ 最优模型已保存! MAE: {best_val_mae:.3f}°C")

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

        scheduler.step()

    if rank == 0:
        # 保存最终模型
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }, f'{save_dir}/final_model.pth')

        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print("\n" + "="*80)
        print(f"训练完成! 最优MAE: {best_val_mae:.3f}°C")
        print(f"观测区域MAE应该接近0（输出组合保证）")
        print("="*80)

    cleanup_distributed()


def main():
    world_size = 8
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
