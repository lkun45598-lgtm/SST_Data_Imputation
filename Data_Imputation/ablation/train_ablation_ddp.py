#!/usr/bin/env python3
"""Ablation training with 4-GPU DDP.

Uses FNO_CBAM_Ablation (ablation/models/fno_cbam_ablation.py) to keep the
main model code untouched.

Variants:
  fno_only       : use_cbam=False, alpha_grad=0,   alpha_temporal=0,   alpha_boundary=0
  cbam_basic     : use_cbam=True,  alpha_grad=0.2, alpha_temporal=0,   alpha_boundary=0
  cbam_boundary  : use_cbam=True,  alpha_grad=0.2, alpha_temporal=0,   alpha_boundary=0.1
  cbam_full      : use_cbam=True,  alpha_grad=0.2, alpha_temporal=0.1, alpha_boundary=0.1

Outputs written to: Data_Imputation/ablation/experiments/{variant}/
"""
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
DATA_IMPUTATION_DIR = THIS_DIR.parent
sys.path.insert(0, str(DATA_IMPUTATION_DIR))   # so we can import inference/
sys.path.insert(0, str(THIS_DIR))              # so we can import models/

from inference.jaxa_inference_dataset import JAXAFinetuneDataset
from models.fno_cbam_ablation import FNO_CBAM_Ablation


# ============================================================
# Loss components (independent copy — do NOT import from training/)
# ============================================================

def masked_mse_loss(pred, target, loss_mask):
    mask = loss_mask.unsqueeze(1)
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)


def masked_gradient_loss(pred, target, loss_mask):
    mask = loss_mask.unsqueeze(1)
    pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_gy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_gy = mask[:, :, 1:, :]
    pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_gx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_gx = mask[:, :, :, 1:]
    ly = (torch.abs(pred_gy - tgt_gy) * mask_gy).sum() / (mask_gy.sum() + 1e-8)
    lx = (torch.abs(pred_gx - tgt_gx) * mask_gx).sum() / (mask_gx.sum() + 1e-8)
    return (lx + ly) / 2


def boundary_smoothness_loss(pred, target, mask_seq):
    cur_mask = mask_seq[:, -1, :, :].unsqueeze(1)
    pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_gx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_lx = cur_mask[:, :, :, :-1]
    mask_rx = cur_mask[:, :, :, 1:]
    boundary_x = (((mask_lx == 0) & (mask_rx == 1)) |
                  ((mask_lx == 1) & (mask_rx == 0))).float()
    lx = (torch.abs(pred_gx - tgt_gx) * boundary_x).sum() / (boundary_x.sum() + 1e-8)
    pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_gy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_ty = cur_mask[:, :, :-1, :]
    mask_by = cur_mask[:, :, 1:, :]
    boundary_y = (((mask_ty == 0) & (mask_by == 1)) |
                  ((mask_ty == 1) & (mask_by == 0))).float()
    ly = (torch.abs(pred_gy - tgt_gy) * boundary_y).sum() / (boundary_y.sum() + 1e-8)
    return (lx + ly) / 2


def output_composition(pred, sst_seq, mask_seq, land_mask):
    cur_input = sst_seq[:, -1:, :, :]
    cur_mask = mask_seq[:, -1:, :, :]
    composed = cur_input * (1 - cur_mask) + pred * cur_mask
    composed = composed * (1 - land_mask.unsqueeze(1))
    return composed


def combined_loss(pred, target, loss_mask, sst_seq, mask_seq,
                  alpha_mse, alpha_grad, alpha_temporal, alpha_boundary):
    loss_mse = masked_mse_loss(pred, target, loss_mask)
    loss_grad = (masked_gradient_loss(pred, target, loss_mask)
                 if alpha_grad > 0 else torch.tensor(0.0, device=pred.device))
    if alpha_temporal > 0:
        prev_day = sst_seq[:, -2:-1, :, :]
        expected = target - prev_day
        predicted = pred - prev_day
        penalty = F.relu(torch.abs(predicted - expected) - 0.5)
        loss_temporal = (penalty * loss_mask.unsqueeze(1)).sum() / (loss_mask.sum() + 1e-8)
    else:
        loss_temporal = torch.tensor(0.0, device=pred.device)
    loss_boundary = (boundary_smoothness_loss(pred, target, mask_seq)
                     if alpha_boundary > 0 else torch.tensor(0.0, device=pred.device))
    total = (alpha_mse * loss_mse + alpha_grad * loss_grad +
             alpha_temporal * loss_temporal + alpha_boundary * loss_boundary)
    return total, loss_mse, loss_grad, loss_temporal, loss_boundary


# ============================================================
# Variants
# ============================================================

VARIANTS = {
    "fno_only":      dict(use_cbam=False, alpha_grad=0.0, alpha_temporal=0.0, alpha_boundary=0.0),
    "cbam_basic":    dict(use_cbam=True,  alpha_grad=0.2, alpha_temporal=0.0, alpha_boundary=0.0),
    "cbam_boundary": dict(use_cbam=True,  alpha_grad=0.2, alpha_temporal=0.0, alpha_boundary=0.1),
    "cbam_full":     dict(use_cbam=True,  alpha_grad=0.2, alpha_temporal=0.1, alpha_boundary=0.1),
}


# ============================================================
# DDP helpers
# ============================================================

def setup_distributed(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


# ============================================================
# Epochs
# ============================================================

def train_epoch(model, loader, optimizer, device, epoch, rank, cfg, norm_mean, norm_std):
    model.train()
    s_loss = s_mse = s_grad = s_temp = s_bound = s_mae = 0.0
    n = 0
    pbar = (tqdm(loader, desc=f"Ep {epoch+1} [T]") if rank == 0 else loader)
    for batch in pbar:
        sst_seq = batch["input_sst_seq"].to(device).float()
        mask_seq = batch["mask_seq"].to(device).float()
        gt_sst = batch["ground_truth_sst"].to(device).unsqueeze(1).float()
        loss_mask = batch["loss_mask"].to(device).float()
        land_mask = batch["land_mask"].to(device).float()

        pred = model(sst_seq, mask_seq)
        pred = output_composition(pred, sst_seq, mask_seq, land_mask)
        total, lm, lg, lt, lb = combined_loss(
            pred, gt_sst, loss_mask, sst_seq, mask_seq,
            alpha_mse=1.0,
            alpha_grad=cfg["alpha_grad"],
            alpha_temporal=cfg["alpha_temporal"],
            alpha_boundary=cfg["alpha_boundary"],
        )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred_k = pred * norm_std + norm_mean
            gt_k = gt_sst * norm_std + norm_mean
            me = loss_mask.unsqueeze(1)
            mae = (torch.abs(pred_k - gt_k) * me).sum() / (me.sum() + 1e-8)
            bs = sst_seq.size(0)
            s_loss += total.item() * bs
            s_mse += lm.item() * bs
            s_grad += lg.item() * bs
            s_temp += lt.item() * bs
            s_bound += lb.item() * bs
            s_mae += mae.item() * bs
            n += bs
            if rank == 0:
                pbar.set_postfix({"MAE": f"{s_mae/n:.3f}K"})

    t = torch.tensor([s_loss, s_mse, s_grad, s_temp, s_bound, s_mae, n], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    s_loss, s_mse, s_grad, s_temp, s_bound, s_mae, n = t.tolist()
    return dict(loss=s_loss/n, mse=s_mse/n, grad=s_grad/n,
                temporal=s_temp/n, boundary=s_bound/n, mae=s_mae/n)


def valid_epoch(model, loader, device, rank, norm_mean, norm_std):
    model.eval()
    s_loss = s_mae = s_rmse = 0.0
    n = 0
    pbar = (tqdm(loader, desc="[V]", leave=False) if rank == 0 else loader)
    with torch.no_grad():
        for batch in pbar:
            sst_seq = batch["input_sst_seq"].to(device).float()
            mask_seq = batch["mask_seq"].to(device).float()
            gt_sst = batch["ground_truth_sst"].to(device).unsqueeze(1).float()
            loss_mask = batch["loss_mask"].to(device).float()
            land_mask = batch["land_mask"].to(device).float()
            pred = model(sst_seq, mask_seq)
            pred = output_composition(pred, sst_seq, mask_seq, land_mask)
            loss = masked_mse_loss(pred, gt_sst, loss_mask)
            pred_k = pred * norm_std + norm_mean
            gt_k = gt_sst * norm_std + norm_mean
            me = loss_mask.unsqueeze(1)
            mae = (torch.abs(pred_k - gt_k) * me).sum() / (me.sum() + 1e-8)
            rmse = torch.sqrt(((pred_k - gt_k) ** 2 * me).sum() / (me.sum() + 1e-8))
            bs = sst_seq.size(0)
            s_loss += loss.item() * bs
            s_mae += mae.item() * bs
            s_rmse += rmse.item() * bs
            n += bs
    t = torch.tensor([s_loss, s_mae, s_rmse, n], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    s_loss, s_mae, s_rmse, n = t.tolist()
    return dict(loss=s_loss/n, mae=s_mae/n, rmse=s_rmse/n)


# ============================================================
# Worker
# ============================================================

def train_worker(rank, world_size, config):
    gpu = config["gpu_ids"][rank]
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    setup_distributed(rank, world_size, config["port"])
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    variant = config["variant"]
    cfg = VARIANTS[variant]
    save_dir = Path(config["save_root"]) / variant

    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print(f"ABLATION: {variant}")
        print(f"  use_cbam={cfg['use_cbam']}, alpha_grad={cfg['alpha_grad']}, "
              f"alpha_temporal={cfg['alpha_temporal']}, alpha_boundary={cfg['alpha_boundary']}")
        print(f"  GPUs={config['gpu_ids']}, world_size={world_size}, "
              f"epochs={config['epochs']}, bs/GPU={config['batch_size']}, "
              f"effective batch={config['batch_size']*world_size}")
        print(f"  save_dir={save_dir}")
        print("=" * 70)

    ostia_mean = 299.9221
    ostia_std = 2.6919

    train_ds = JAXAFinetuneDataset(
        data_dir=config["data_dir"], series_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        window_size=30, mask_ratio=0.2, min_mask_size=10, max_mask_size=50,
        normalize=True, mean=ostia_mean, std=ostia_std,
        cache_size=100, seed=42,
    )
    valid_ds = JAXAFinetuneDataset(
        data_dir=config["data_dir"], series_ids=[8],
        window_size=30, mask_ratio=0.2, min_mask_size=10, max_mask_size=50,
        normalize=True, mean=ostia_mean, std=ostia_std,
        cache_size=50, seed=123,
    )
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              sampler=train_sampler, num_workers=4, pin_memory=True,
                              prefetch_factor=2)
    valid_loader = DataLoader(valid_ds, batch_size=config["batch_size"],
                              sampler=valid_sampler, num_workers=2, pin_memory=True,
                              prefetch_factor=2)

    model = FNO_CBAM_Ablation(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16, use_cbam=cfg["use_cbam"],
    ).to(device)

    if os.path.exists(config["pretrain"]):
        if rank == 0:
            print(f"Load pretrain: {config['pretrain']}")
        ckpt = torch.load(config["pretrain"], map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"  Missing={len(missing)}, Unexpected={len(unexpected)}")

    # find_unused_parameters=True required when use_cbam=False
    model = DDP(model, device_ids=[gpu],
                find_unused_parameters=(not cfg["use_cbam"]))

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6,
    )

    best_val_mae = float("inf")
    history = {"train": [], "valid": []}

    for epoch in range(config["epochs"]):
        train_sampler.set_epoch(epoch)
        tr = train_epoch(model, train_loader, optimizer, device, epoch, rank,
                         cfg, ostia_mean, ostia_std)
        va = valid_epoch(model, valid_loader, device, rank, ostia_mean, ostia_std)
        scheduler.step()

        if rank == 0:
            history["train"].append(tr)
            history["valid"].append(va)
            print(f"[{variant}] Ep {epoch+1}/{config['epochs']}: "
                  f"train_mae={tr['mae']:.4f}K  valid_mae={va['mae']:.4f}K  "
                  f"valid_rmse={va['rmse']:.4f}K  lr={optimizer.param_groups[0]['lr']:.6f}")
            if va["mae"] < best_val_mae:
                best_val_mae = va["mae"]
                torch.save({
                    "epoch": epoch, "model_state_dict": model.module.state_dict(),
                    "val_mae": va["mae"], "valid_metrics": va,
                    "variant": variant, "config": cfg,
                    "norm_mean": ostia_mean, "norm_std": ostia_std,
                }, save_dir / "best_model.pth")
                print(f"  ✓ best so far: {best_val_mae:.4f} K")
            if (epoch + 1) % 5 == 0 or epoch == config["epochs"] - 1:
                with open(save_dir / "training_history.json", "w") as f:
                    json.dump(history, f, indent=2)
                with open(save_dir / "config.json", "w") as f:
                    json.dump({**cfg, "variant": variant,
                              "epochs": config["epochs"], "lr": config["lr"],
                              "batch_size": config["batch_size"]}, f, indent=2)

    if rank == 0:
        torch.save({
            "epoch": config["epochs"] - 1,
            "model_state_dict": model.module.state_dict(),
            "variant": variant, "config": cfg,
            "norm_mean": ostia_mean, "norm_std": ostia_std,
        }, save_dir / "final_model.pth")
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n[{variant}] DONE. Best val MAE: {best_val_mae:.4f} K")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--gpu-ids", type=str, default="4,5,6,7",
                        help="comma-separated GPU ids")
    parser.add_argument("--port", type=int, default=29610)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--data-dir", type=str,
                        default="/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled")
    parser.add_argument("--pretrain", type=str,
                        default="/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/ostia_pretrain/best_model.pth")
    parser.add_argument("--save-root", type=str,
                        default=str(Path(__file__).resolve().parent / "experiments"))
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    world_size = len(gpu_ids)
    config = dict(
        variant=args.variant, gpu_ids=gpu_ids, port=args.port,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        data_dir=args.data_dir, pretrain=args.pretrain, save_root=args.save_root,
    )
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
