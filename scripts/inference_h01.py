#!/usr/bin/env python3
"""H=1 model inference and visualization"""
import sys
import os
import torch
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, 'Data_Imputation')
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.jaxa_inference_dataset import JAXAFinetuneDataset

print("=" * 70)
print("H=1 Model Inference")
print("=" * 70)

device = torch.device('cuda:0')
torch.manual_seed(42)

# Load best model
model_path = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h01/best_model.pth'
print(f"\nLoading model from {model_path}...")

model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
norm_mean = checkpoint['norm_mean']
norm_std = checkpoint['norm_std']
print(f"✓ Model loaded (norm: mean={norm_mean:.2f}, std={norm_std:.2f})")

# Load test dataset (series 8)
print("\nLoading test dataset (series 8)...")
test_dataset = JAXAFinetuneDataset(
    data_dir='/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data/h01',
    series_ids=[8],
    window_size=30,
    mask_ratio=0.0,
    normalize=True,
    mean=norm_mean,
    std=norm_std,
    cache_size=50,
    seed=42
)
print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

# Get some test samples
model.eval()
predictions_list = []
targets_list = []
sample_indices = [0, len(test_dataset)//4, len(test_dataset)//2, 3*len(test_dataset)//4]

print(f"\nRunning inference on {len(sample_indices)} samples...")
with torch.no_grad():
    for idx, sample_idx in enumerate(sample_indices):
        batch = test_dataset[sample_idx]
        
        sst_seq = torch.from_numpy(batch['input_sst_seq']).unsqueeze(0).to(device).float()
        mask_seq = torch.from_numpy(batch['mask_seq']).unsqueeze(0).to(device).float()
        gt_sst = torch.from_numpy(batch['ground_truth_sst']).unsqueeze(0).unsqueeze(1).to(device).float()
        land_mask = torch.from_numpy(batch['land_mask']).unsqueeze(0).to(device).float()
        
        # Forward pass
        pred = model(sst_seq, mask_seq)
        
        # Denormalize
        pred_kelvin = pred * norm_std + norm_mean
        gt_kelvin = gt_sst * norm_std + norm_mean
        
        # To numpy
        pred_kelvin = pred_kelvin.squeeze().cpu().numpy()
        gt_kelvin = gt_kelvin.squeeze().cpu().numpy()
        land_mask_np = land_mask.squeeze().cpu().numpy()
        
        pred_kelvin[land_mask_np == 1] = np.nan
        gt_kelvin[land_mask_np == 1] = np.nan
        
        predictions_list.append(pred_kelvin)
        targets_list.append(gt_kelvin)
        
        # Metrics
        ocean_mask = land_mask_np == 0
        mae = np.abs(pred_kelvin[ocean_mask] - gt_kelvin[ocean_mask]).mean()
        rmse = np.sqrt(((pred_kelvin[ocean_mask] - gt_kelvin[ocean_mask])**2).mean())
        
        print(f"  Sample {idx+1}: MAE={mae:.3f}K, RMSE={rmse:.3f}K")

# Save results
print("\nSaving inference results...")
output_dir = Path('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/h01_inference')
output_dir.mkdir(exist_ok=True)

with h5py.File(output_dir / 'inference_results.h5', 'w') as f:
    f.create_dataset('predictions', data=np.array(predictions_list))
    f.create_dataset('targets', data=np.array(targets_list))
    f.attrs['norm_mean'] = norm_mean
    f.attrs['norm_std'] = norm_std

print(f"✓ Results saved to {output_dir}")
print("\n" + "=" * 70)
print("Inference complete!")
print("=" * 70)
