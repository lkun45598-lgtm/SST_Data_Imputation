#!/usr/bin/env python3
"""Quick test of training startup"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, 'Data_Imputation')
from inference.jaxa_inference_dataset import JAXAFinetuneDataset
from torch.utils.data import DataLoader
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

print("Starting training test...")
print(f"GPU available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}")

# Test dataset loading
print("\n[1/4] Loading dataset...")
train_dataset = JAXAFinetuneDataset(
    data_dir='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01',
    series_ids=[0],
    window_size=30,
    mask_ratio=0.2,
    min_mask_size=10,
    max_mask_size=50,
    normalize=True,
    mean=299.9221,
    std=2.6919,
    cache_size=100,
    seed=42
)
print(f"  ✓ Dataset loaded: {len(train_dataset)} samples")

print("\n[2/4] Creating DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, pin_memory=True)
print(f"  ✓ DataLoader created")

print("\n[3/4] Creating model...")
model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"  ✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")

print("\n[4/4] Testing forward pass...")
batch = next(iter(train_loader))
sst_seq = batch['input_sst_seq'].to(device).float()
mask_seq = batch['mask_seq'].to(device).float()
print(f"  ✓ Batch shapes: sst={sst_seq.shape}, mask={mask_seq.shape}")

with torch.no_grad():
    pred = model(sst_seq, mask_seq)
    print(f"  ✓ Forward pass OK: pred shape = {pred.shape}")

print("\n✓ All tests passed!")
