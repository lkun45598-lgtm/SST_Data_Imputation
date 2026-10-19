#!/usr/bin/env python3
"""
Generate comparison plots for all methods evaluation results.
Reads: cache/all_methods_full.npz
Outputs: output/method_comparison_*.pdf/png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

CACHE = Path(__file__).parent / "cache"
OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# Load data
npz = np.load(CACHE / "all_methods_full.npz", allow_pickle=True)
records = npz['records']
df = pd.DataFrame(records.tolist())

# Method order and colors
method_order = ['fno', 'convlstm', 'unet', 'dincae', 'oi', 'knn', 'dineof']
method_labels = {
    'fno': 'FNO (Ours)',
    'convlstm': 'ConvLSTM',
    'unet': 'UNet',
    'dincae': 'DINCAE',
    'oi': 'OI',
    'knn': 'KNN',
    'dineof': 'DINEOF'
}
colors = {
    'fno': '#d62728',      # red
    'convlstm': '#ff7f0e', # orange
    'unet': '#2ca02c',     # green
    'dincae': '#1f77b4',   # blue
    'oi': '#9467bd',       # purple
    'knn': '#8c564b',      # brown
    'dineof': '#7f7f7f'    # gray
}

# ============================================================
# Figure 1: MAE by missingness level (grouped bar chart)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

levels = ['low', 'mid', 'high']
level_labels = ['Low (30%)', 'Mid (55%)', 'High (75%)']
x = np.arange(len(levels))
width = 0.12

for i, method in enumerate(method_order):
    mae_vals = []
    for lv in levels:
        subset = df[(df['method']==method) & (df['level']==lv)]['mae']
        mae_vals.append(subset.mean() if len(subset) > 0 else np.nan)

    offset = (i - len(method_order)/2 + 0.5) * width
    ax.bar(x + offset, mae_vals, width, label=method_labels[method],
           color=colors[method], alpha=0.9)

ax.set_xlabel('Missingness Level', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE (K)', fontsize=12, fontweight='bold')
ax.set_title('Reconstruction Error by Missingness Level (Series 8, 657 frames)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(level_labels)
ax.legend(ncol=4, loc='upper left', framealpha=0.95)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / "method_comparison_mae_by_level.pdf", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT / "method_comparison_mae_by_level.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT}/method_comparison_mae_by_level.pdf/png")
plt.close()

# ============================================================
# Figure 2: Overall MAE + error bars (mean ± std)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

stats = []
for method in method_order:
    subset = df[df['method']==method]['mae']
    stats.append({
        'method': method_labels[method],
        'mean': subset.mean(),
        'std': subset.std(),
        'color': colors[method]
    })

stats_df = pd.DataFrame(stats)
y_pos = np.arange(len(stats_df))

ax.barh(y_pos, stats_df['mean'], xerr=stats_df['std'],
        color=[s['color'] for s in stats], alpha=0.9,
        capsize=5, error_kw={'linewidth': 1.5})

ax.set_yticks(y_pos)
ax.set_yticklabels(stats_df['method'])
ax.invert_yaxis()
ax.set_xlabel('MAE (K) [mean ± std]', fontsize=12, fontweight='bold')
ax.set_title('Overall Reconstruction Error (657 frames)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(stats_df['mean'], stats_df['std'])):
    ax.text(mean + std + 0.02, i, f'{mean:.3f}±{std:.3f}',
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT / "method_comparison_overall_mae.pdf", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT / "method_comparison_overall_mae.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT}/method_comparison_overall_mae.pdf/png")
plt.close()

# ============================================================
# Figure 3: Box plot comparison (Deep models only, exclude outliers)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

deep_methods = ['fno', 'convlstm', 'unet', 'dincae']
deep_data = df[df['method'].isin(deep_methods)].copy()
deep_data['Method'] = deep_data['method'].map(method_labels)

# Create box plot
bp = ax.boxplot([deep_data[deep_data['method']==m]['mae'].values for m in deep_methods],
                 labels=[method_labels[m] for m in deep_methods],
                 patch_artist=True, showfliers=False, widths=0.6)

# Color boxes
for patch, method in zip(bp['boxes'], deep_methods):
    patch.set_facecolor(colors[method])
    patch.set_alpha(0.7)

ax.set_ylabel('MAE (K)', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Reconstruction Errors (Deep Learning Methods)',
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / "method_comparison_boxplot.pdf", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT / "method_comparison_boxplot.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT}/method_comparison_boxplot.pdf/png")
plt.close()

# ============================================================
# Figure 4: Gradient Correlation vs MAE scatter
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for method in method_order:
    subset = df[df['method']==method]
    ax.scatter(subset['mae'], subset['gradcorr'],
               label=method_labels[method], color=colors[method],
               alpha=0.5, s=20)

ax.set_xlabel('MAE (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Gradient Correlation', fontsize=12, fontweight='bold')
ax.set_title('Reconstruction Accuracy vs Spatial Structure Preservation',
             fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / "method_comparison_mae_vs_gradcorr.pdf", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT / "method_comparison_mae_vs_gradcorr.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT}/method_comparison_mae_vs_gradcorr.pdf/png")
plt.close()

print("\n=== All comparison plots generated ===")
