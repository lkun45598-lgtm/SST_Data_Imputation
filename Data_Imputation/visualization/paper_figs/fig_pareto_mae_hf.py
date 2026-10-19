#!/usr/bin/env python3
"""
Generate MAE vs hf_log_ratio Pareto plot for all methods
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set plot style
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['legend.fontsize'] = 9

# Load data
data_path = Path(__file__).parent / "cache" / "all_methods_stats_v6_merged.npz"
data = np.load(data_path, allow_pickle=True)
records = data['records']

# Method configurations
method_configs = {
    'fno': {'label': 'FNO (Ours)', 'color': '#e74c3c', 'marker': 'o', 's': 100, 'zorder': 10},
    'dincae': {'label': 'DINCAE', 'color': '#3498db', 'marker': 's', 's': 80, 'zorder': 5},
    'unet': {'label': 'U-Net', 'color': '#2ecc71', 'marker': '^', 's': 80, 'zorder': 5},
    'convlstm': {'label': 'ConvLSTM', 'color': '#9b59b6', 'marker': 'v', 's': 80, 'zorder': 5},
    'dineof': {'label': 'DINEOF', 'color': '#f39c12', 'marker': 'D', 's': 60, 'zorder': 3},
    'oi': {'label': 'OI', 'color': '#95a5a6', 'marker': 'p', 's': 70, 'zorder': 3},
    'knn': {'label': 'KNN', 'color': '#34495e', 'marker': 'h', 's': 70, 'zorder': 3},
}

# Level configurations
level_configs = {
    'low': {'label': '10-20% missing', 'alpha': 0.3, 'edgewidth': 0.5},
    'medium': {'label': '20-30% missing', 'alpha': 0.6, 'edgewidth': 0.8},
    'high': {'label': '30-40% missing', 'alpha': 1.0, 'edgewidth': 1.2},
}

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Group by method and level, compute mean MAE and hf_log_ratio
method_level_stats = {}
for rec in records:
    method = rec['method']
    level = rec['level']
    mae = rec['mae']
    hf_log_ratio = rec['hf_log_ratio']

    key = (method, level)
    if key not in method_level_stats:
        method_level_stats[key] = {'mae': [], 'hf_log_ratio': []}

    method_level_stats[key]['mae'].append(mae)
    method_level_stats[key]['hf_log_ratio'].append(hf_log_ratio)

# Compute mean for each method-level combination
mean_stats = {}
for key, vals in method_level_stats.items():
    mean_stats[key] = {
        'mae': np.mean(vals['mae']),
        'hf_log_ratio': np.mean(vals['hf_log_ratio'])
    }

# Plot each method with different levels
legend_handles = []
legend_labels = []

# First pass: plot all points
for method in ['fno', 'dincae', 'unet', 'convlstm', 'dineof', 'oi', 'knn']:
    if method not in method_configs:
        continue

    config = method_configs[method]

    for level in ['low', 'medium', 'high']:
        key = (method, level)
        if key not in mean_stats:
            continue

        stats = mean_stats[key]
        level_cfg = level_configs[level]

        # Plot point
        h = ax.scatter(
            stats['hf_log_ratio'],
            stats['mae'],
            color=config['color'],
            marker=config['marker'],
            s=config['s'],
            alpha=level_cfg['alpha'],
            edgecolors='white',
            linewidths=level_cfg['edgewidth'],
            zorder=config['zorder'],
            label=config['label'] if level == 'high' else None
        )

    # Add to legend (only once per method, using high level)
    if (method, 'high') in mean_stats:
        legend_handles.append(h)
        legend_labels.append(config['label'])

# Add level indicators in legend
for level in ['low', 'medium', 'high']:
    level_cfg = level_configs[level]
    h = ax.scatter([], [], color='gray', marker='o', s=80,
                   alpha=level_cfg['alpha'], edgecolors='white',
                   linewidths=level_cfg['edgewidth'])
    legend_handles.append(h)
    legend_labels.append(level_cfg['label'])

# Identify Pareto frontier
# Lower MAE is better (minimize), higher hf_log_ratio is better (maximize)
# So Pareto optimal means: for any two points (x1, y1) and (x2, y2),
# neither (x1 > x2 AND y1 < y2) nor (x1 < x2 AND y1 > y2)

all_points = []
for key, stats in mean_stats.items():
    all_points.append((stats['hf_log_ratio'], stats['mae'], key))

# Sort by hf_log_ratio (descending) for Pareto frontier detection
all_points_sorted = sorted(all_points, key=lambda p: -p[0])

# Find Pareto frontier (maximize hf_log_ratio, minimize MAE)
pareto_points = []
min_mae_so_far = float('inf')

for hf, mae, key in all_points_sorted:
    if mae < min_mae_so_far:
        pareto_points.append((hf, mae, key))
        min_mae_so_far = mae

# Draw Pareto frontier
if len(pareto_points) > 1:
    pareto_points_sorted = sorted(pareto_points, key=lambda p: p[0])
    x_pareto = [p[0] for p in pareto_points_sorted]
    y_pareto = [p[1] for p in pareto_points_sorted]
    ax.plot(x_pareto, y_pareto, 'k--', alpha=0.3, linewidth=1.5,
            label='Pareto frontier', zorder=1)
    legend_handles.append(plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.3, linewidth=1.5))
    legend_labels.append('Pareto frontier')

# Set labels and title
ax.set_xlabel('High-frequency preservation (hf_log_ratio, dB)', fontsize=11)
ax.set_ylabel('Mean Absolute Error (K)', fontsize=11)
ax.set_title('Pareto Plot: MAE vs High-Frequency Content', fontsize=12, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

# Set better limits
ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.05)

# Add legend
ax.legend(legend_handles, legend_labels, loc='upper right', framealpha=0.9,
          ncol=2, columnspacing=1.0, handletextpad=0.5)

# Tight layout
plt.tight_layout()

# Save
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "fig_pareto_mae_hf.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_pareto_mae_hf.pdf", bbox_inches='tight')

print(f"Pareto plot saved to {output_dir}")
print(f"\nPareto frontier points ({len(pareto_points)}):")
for hf, mae, key in sorted(pareto_points, key=lambda p: -p[0]):
    method, level = key
    print(f"  {method:10s} ({level:6s}): MAE={mae:.4f}K, hf_log_ratio={hf:.2f}dB")

plt.close()
