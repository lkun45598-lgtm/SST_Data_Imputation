#!/usr/bin/env python3
"""Visualize H=1 inference results"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

print("Loading inference results...")
results_file = '/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/h01_inference/inference_results.h5'

with h5py.File(results_file, 'r') as f:
    predictions = f['predictions'][:]
    targets = f['targets'][:]
    norm_mean = f.attrs['norm_mean']
    norm_std = f.attrs['norm_std']

print(f"✓ Loaded {predictions.shape[0]} samples")

# Create comparison figure
fig, axes = plt.subplots(predictions.shape[0], 3, figsize=(15, 12))
fig.suptitle('H=1 Model Inference: Predictions vs Targets', fontsize=16, fontweight='bold')

# Colormap and normalization
cmap = plt.cm.RdYlBu_r
vmin, vmax = 285, 310

for i in range(predictions.shape[0]):
    pred = predictions[i]
    target = targets[i]
    
    # Compute error
    error = np.abs(pred - target)
    
    # Predicted
    im1 = axes[i, 0].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f'Sample {i+1}: Prediction', fontsize=11, fontweight='bold')
    axes[i, 0].axis('off')
    
    # Target
    im2 = axes[i, 1].imshow(target, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f'Sample {i+1}: Target', fontsize=11, fontweight='bold')
    axes[i, 1].axis('off')
    
    # Error
    im3 = axes[i, 2].imshow(error, cmap='Reds', vmin=0, vmax=2)
    axes[i, 2].set_title(f'Sample {i+1}: Error (MAE={np.nanmean(error):.2f}K)', fontsize=11, fontweight='bold')
    axes[i, 2].axis('off')
    
    # Add colorbars
    if i == 0:
        cbar1 = plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cbar1.set_label('SST (K)', fontsize=9)
        cbar2 = plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        cbar2.set_label('SST (K)', fontsize=9)
        cbar3 = plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        cbar3.set_label('Error (K)', fontsize=9)

plt.tight_layout()
output_path = '/data1/user/lz/SST_Data_Imputation/h01_inference_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved comparison figure to {output_path}")

# Statistics figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('H=1 Model Error Statistics', fontsize=14, fontweight='bold')

errors = []
for i in range(predictions.shape[0]):
    pred = predictions[i]
    target = targets[i]
    ocean_mask = ~np.isnan(target)
    err = np.abs(pred[ocean_mask] - target[ocean_mask])
    errors.extend(err.flatten())

errors = np.array(errors)

# Error distribution
axes[0].hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}K')
axes[0].axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}K')
axes[0].set_xlabel('Absolute Error (K)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Per-sample statistics
sample_names = [f'Sample {i+1}' for i in range(predictions.shape[0])]
mae_list = []
rmse_list = []

for i in range(predictions.shape[0]):
    pred = predictions[i]
    target = targets[i]
    ocean_mask = ~np.isnan(target)
    mae = np.abs(pred[ocean_mask] - target[ocean_mask]).mean()
    rmse = np.sqrt(((pred[ocean_mask] - target[ocean_mask])**2).mean())
    mae_list.append(mae)
    rmse_list.append(rmse)

x = np.arange(len(sample_names))
width = 0.35
axes[1].bar(x - width/2, mae_list, width, label='MAE', color='steelblue', alpha=0.8)
axes[1].bar(x + width/2, rmse_list, width, label='RMSE', color='coral', alpha=0.8)
axes[1].set_xlabel('Sample', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Error (K)', fontsize=11, fontweight='bold')
axes[1].set_title('Per-Sample Metrics', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(sample_names)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path2 = '/data1/user/lz/SST_Data_Imputation/h01_inference_statistics.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved statistics figure to {output_path2}")

# Summary
print("\n" + "=" * 70)
print("Overall Inference Statistics")
print("=" * 70)
print(f"Mean Error: {np.mean(errors):.3f}K")
print(f"Median Error: {np.median(errors):.3f}K")
print(f"Std Dev: {np.std(errors):.3f}K")
print(f"Max Error: {np.max(errors):.3f}K")
print(f"95th Percentile: {np.percentile(errors, 95):.3f}K")
print("=" * 70)

plt.close('all')
print("✓ Visualization complete!")
