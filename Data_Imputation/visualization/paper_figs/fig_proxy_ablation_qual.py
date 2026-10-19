#!/usr/bin/env python3
"""
Proxy construction ablation: qualitative comparison.
选择代表性大云区帧，对比 Full / w/o Gaussian / w/o 3D KNN 的重建效果。
所有变体用统一的干净真值（原始 JAXA 观测）。
"""
import sys
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baselines"))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
from style import CMAPS, LAND_COLOR

DEVICE = torch.device('cuda:7')
OUTPUT = Path(__file__).parent / "output"
NORM_MEAN, NORM_STD = 299.9221, 2.6919
W = 30

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# 三个变体（按顺序：Full, nogauss, noknn）
VARIANTS = [
    ('full',     'Full',           REPO/'experiments/hourly_data/h12/jaxa_knn_filled_08.h5',
                                   REPO/'experiments/jaxa_finetune_h12_realmask_clean/best_model.pth'),
    ('nogauss',  'w/o Gaussian',   REPO/'experiments/hourly_data/h12_nogauss/jaxa_knn_filled_08.h5',
                                   REPO/'experiments/jaxa_finetune_h12_nogauss_realmask_clean/best_model.pth'),
    ('noknn',    'w/o 3-D KNN',    REPO/'experiments/hourly_data/h12_noknn/jaxa_knn_filled_08.h5',
                                   REPO/'experiments/jaxa_finetune_h12_noknn_realmask_clean/best_model.pth'),
]

# 代表性帧（中等遮挡，有结构）
FRAMES = [266, 87, 183]

def load_model(p):
    m = FNO_CBAM_SST_Temporal(out_size=(451,351), modes1=80, modes2=64, width=64, depth=6).to(DEVICE)
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m

def run_model(model, sst_seq, miss_seq, mask):
    si = sst_seq.copy(); si[-1] = np.where(mask>0, NORM_MEAN, si[-1])
    ms = miss_seq.copy().astype(np.float32); ms[-1] = mask
    st = torch.from_numpy(np.nan_to_num((si-NORM_MEAN)/NORM_STD, nan=0.0)).unsqueeze(0).float().to(DEVICE)
    mt = torch.from_numpy(ms).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(st, mt).squeeze().cpu().numpy()
    return pred*NORM_STD + NORM_MEAN

def main():
    # 统一真值（从 noknn h5，观测区最干净）
    gt_h5 = REPO/'experiments/hourly_data/h12_noknn/jaxa_knn_filled_08.h5'
    with h5py.File(gt_h5, 'r') as f:
        gt_all = f['sst_data'][:].astype(np.float32)
        obs_all = f['original_obs_mask'][:].astype(np.uint8)
        land = f['land_mask'][:].astype(np.uint8)

    # 生成 mask（mid level, 55%）
    h5_paths = [str(REPO/f'experiments/hourly_data/h12/jaxa_knn_filled_{i:02d}.h5') for i in [0,3,6]]
    bank = build_cloud_bank(h5_paths, max_donors=400, seed=7)
    gen = RealCloudMaskGenerator(bank, seed=550)

    # 加载三个模型
    models = {key: load_model(mpath) for key, _, _, mpath in VARIANTS}

    # 对每个模型加载其输入 proxy
    proxies = {}
    for key, _, h5p, _ in VARIANTS:
        with h5py.File(h5p, 'r') as f:
            proxies[key] = {
                'sst': f['sst_data'][:].astype(np.float32),
                'miss': f['original_missing_mask'][:].astype(np.uint8),
            }

    # 开始画图（3 行 × 4 列：GT | Full | nogauss | noknn）
    fig = plt.figure(figsize=(13.5, 10))
    gs = GridSpec(3, 4, figure=fig, wspace=0.15, hspace=0.25,
                  left=0.06, right=0.98, top=0.93, bottom=0.06)

    ocean = (land == 0)

    for row_i, idx in enumerate(FRAMES):
        elig = (obs_all[idx]*ocean).astype(np.float32)
        mask = gen.generate(elig, target_ratio=0.55)
        ev = (mask * elig).astype(bool)
        gt = gt_all[idx]

        # 推理三个变体
        preds = {}
        for key, label, _, _ in VARIANTS:
            sst_seq = np.stack([proxies[key]['sst'][max(0, idx-(W-1)+t)] for t in range(W)])
            miss_seq = np.stack([proxies[key]['miss'][max(0, idx-(W-1)+t)] for t in range(W)])
            preds[key] = run_model(models[key], sst_seq, miss_seq, mask)

        # 列 0: Ground Truth
        ax = fig.add_subplot(gs[row_i, 0])
        field_gt = np.where(land==1, np.nan, gt)
        im = ax.imshow(field_gt, cmap=CMAPS['sst'], vmin=290, vmax=304, interpolation='nearest')
        ax.set_title('Ground Truth' if row_i==0 else '', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Frame {idx}', fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # 列 1-3: 三个变体
        for col_i, (key, label, _, _) in enumerate(VARIANTS, start=1):
            ax = fig.add_subplot(gs[row_i, col_i])
            pred = preds[key]
            field_pred = np.where(land==1, np.nan, pred)
            ax.imshow(field_pred, cmap=CMAPS['sst'], vmin=290, vmax=304, interpolation='nearest')

            # 计算该变体的 MAE（挖空区，用统一真值）
            mae = np.abs(pred[ev] - gt[ev]).mean()

            ax.set_title(label if row_i==0 else '', fontsize=11, fontweight='bold')
            ax.text(0.98, 0.04, f'MAE={mae:.3f}K', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none'))
            ax.set_xticks([]); ax.set_yticks([])

    # Colorbar（右侧统一）
    cbar_ax = fig.add_axes([0.99, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('SST (K)', fontsize=10, fontweight='bold')

    fig.suptitle('Proxy-construction ablation: reconstruction at 55% masking (Series 8)',
                 fontsize=13, fontweight='bold', y=0.97)

    plt.savefig(OUTPUT / 'fig_proxy_ablation_qual.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT / 'fig_proxy_ablation_qual.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT}/fig_proxy_ablation_qual.pdf/png")

if __name__ == '__main__':
    main()
