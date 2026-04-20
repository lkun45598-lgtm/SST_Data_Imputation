#!/usr/bin/env python3
"""Single GPU training test"""
import os
import sys
import torch
sys.path.insert(0, 'Data_Imputation')
from inference.jaxa_inference_dataset import JAXAFinetuneDataset
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from training.train_jaxa_hourly import output_composition, jaxa_combined_loss
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device(f'cuda:0')  # 当CUDA_VISIBLE_DEVICES=2时，cuda:0指向GPU 2
torch.manual_seed(42)

print("=" * 70)
print(f"Single GPU Training Test (cuda:0 -> GPU {torch.cuda.current_device()})")
print("=" * 70)

# Load dataset
print("\nLoading dataset (2 series for quick test)...")
train_dataset = JAXAFinetuneDataset(
    data_dir='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01',
    series_ids=[0, 1],
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
print(f"  ✓ {len(train_dataset)} samples")

train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0)

# Model
print("\nCreating model...")
model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6).to(device)
print(f"  ✓ {sum(p.numel() for p in model.parameters()):,} params")

# Pretrained
pretrained = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune/best_model.pth'
if os.path.exists(pretrained):
    print(f"\nLoading pretrained...")
    ckpt = torch.load(pretrained, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    print("  ✓ Loaded")

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training test
print(f"\nTraining 2 batches...")
model.train()

for idx, batch in enumerate(train_loader):
    if idx >= 2:
        break
    
    sst_seq = batch['input_sst_seq'].to(device).float()
    mask_seq = batch['mask_seq'].to(device).float()
    gt_sst = batch['ground_truth_sst'].to(device).unsqueeze(1).float()
    loss_mask = batch['loss_mask'].to(device).float()
    land_mask = batch['land_mask'].to(device).float()
    
    pred = model(sst_seq, mask_seq)
    pred = output_composition(pred, sst_seq, mask_seq, land_mask)
    loss, loss_mse, loss_grad = jaxa_combined_loss(pred, gt_sst, loss_mask, sst_seq=sst_seq,
                                                     alpha_mse=1.0, alpha_grad=0.02, alpha_temporal=0.1)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"  Batch {idx+1}: loss={loss.item():.4f}")

print("\n✓ Training test successful!")
