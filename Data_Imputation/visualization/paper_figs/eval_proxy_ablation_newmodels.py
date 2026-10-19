#!/usr/bin/env python3
"""
只评估 nogauss 和 noknn 两个新消融变体（用统一干净真值）。
Full 结果直接用 table1 的数字（之前评估正确的）。
"""
import sys
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baselines"))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
from losses.gradient_masks import central_gradient_magnitude

DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
CACHE = Path(__file__).parent / "cache"
NORM_MEAN, NORM_STD = 299.9221, 2.6919
W, SEED = 30, 42
NUM_SAMPLES = 9999  # 全量
LEVELS = [('low', 0.30), ('mid', 0.55), ('high', 0.75)]

VARIANTS = {
    'nogauss': {
        'label': 'No Gaussian (Temporal+KNN)',
        'h5_path': REPO / 'experiments/hourly_data/h12_nogauss/jaxa_knn_filled_08.h5',
        'model_path': REPO / 'experiments/jaxa_finetune_h12_nogauss_realmask_clean/best_model.pth',
    },
    'noknn': {
        'label': 'No KNN (Temporal+Gaussian)',
        'h5_path': REPO / 'experiments/hourly_data/h12_noknn/jaxa_knn_filled_08.h5',
        'model_path': REPO / 'experiments/jaxa_finetune_h12_noknn_realmask_clean/best_model.pth',
    },
}

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

def gradmag(field, valid_ref_mask):
    return central_gradient_magnitude(field, valid_ref_mask)[0]

def evaluate_variant(vkey, gt_all, obs_all, land):
    cfg = VARIANTS[vkey]
    print(f"\n{'='*70}")
    print(f"Variant: {cfg['label']}")
    print(f"  H5(输入proxy): {cfg['h5_path'].name}")
    print(f"  Model: {cfg['model_path'].relative_to(REPO)}")
    print('='*70)

    model = load_model(cfg['model_path'])

    with h5py.File(cfg['h5_path'], 'r') as f:
        sst_all = f['sst_data'][:].astype(np.float32)
        miss_all = f['original_missing_mask'][:].astype(np.uint8)

    ocean = (land == 0)
    cand = [i for i in range(W-1, sst_all.shape[0]) if (obs_all[i]*ocean).sum() > 4000]

    h5_paths = [str(REPO/f'experiments/hourly_data/h12/jaxa_knn_filled_{i:02d}.h5') for i in [0,3,6]]
    bank = build_cloud_bank(h5_paths, max_donors=400, seed=7)

    results = []
    rng = np.random.default_rng(SEED)

    for level_name, ratio in LEVELS:
        idxs = rng.choice(cand, size=min(NUM_SAMPLES, len(cand)), replace=False)
        gen = RealCloudMaskGenerator(bank, seed=int(ratio*1000))

        for idx in tqdm(idxs, desc=f'  {level_name}', leave=False):
            elig = (obs_all[idx]*ocean).astype(np.float32)
            if elig.sum() < 2000:
                continue

            mask = gen.generate(elig, target_ratio=ratio)
            ev = (mask*elig).astype(bool)
            if ev.sum() < 200:
                continue

            sst_seq = np.stack([sst_all[max(0, idx-(W-1)+t)] for t in range(W)])
            miss_seq = np.stack([miss_all[max(0, idx-(W-1)+t)] for t in range(W)])
            gt = gt_all[idx]  # 统一真值

            pred = run_model(model, sst_seq, miss_seq, mask)

            e = pred[ev] - gt[ev]
            valid_ref = obs_all[idx].astype(bool) & (land == 0)
            gtg = gradmag(gt, valid_ref)
            pg = gradmag(pred, valid_ref)
            vv = ev & np.isfinite(pg) & np.isfinite(gtg)

            grad_ratio = float(pg[vv].mean()/(gtg[vv].mean()+1e-9)) if vv.sum()>0 else np.nan
            gradcorr = float(np.corrcoef(pg[vv], gtg[vv])[0,1]) if vv.sum()>1 else np.nan

            results.append({
                'variant': vkey,
                'level': level_name,
                'idx': int(idx),
                'mae': float(np.abs(e).mean()),
                'rmse': float(np.sqrt((e**2).mean())),
                'bias': float(e.mean()),
                'grad_ratio': grad_ratio,
                'gradcorr': gradcorr,
            })

    return results

def main():
    print("="*70)
    print("Proxy 消融：只评估 nogauss 和 noknn（用统一真值）")
    print("Full 结果直接用 table1 的（之前正确的）")
    print("="*70)

    # 统一真值（noknn 的观测区最干净）
    gt_h5 = REPO/'experiments/hourly_data/h12_noknn/jaxa_knn_filled_08.h5'
    with h5py.File(gt_h5, 'r') as f:
        gt_all = f['sst_data'][:].astype(np.float32)
        obs_all = f['original_obs_mask'][:].astype(np.uint8)
        land = f['land_mask'][:].astype(np.uint8)
    print(f"GT: {gt_all.shape}, 观测像素均值={int((obs_all*(land==0)).sum(axis=(1,2)).mean())}\n")

    all_results = []
    for vkey in ['nogauss', 'noknn']:
        all_results.extend(evaluate_variant(vkey, gt_all, obs_all, land))

    records = np.array(all_results, dtype=object)
    out = CACHE / 'proxy_ablation_newmodels.npz'
    np.savez_compressed(out, records=records)
    print(f"\n{'='*70}")
    print(f"Saved: {out}  ({len(records)} records)")

    # 汇总
    print(f"\n{'='*70}")
    print("Summary (nogauss vs noknn)")
    print('='*70)
    print(f"{'Variant':<30} {'Level':<6} {'MAE':<12} {'RMSE':<12} {'Bias':<10} {'GradRatio':<12} {'GradCorr':<10}")
    print('-'*70)

    for vkey in ['nogauss', 'noknn']:
        for lname in ['low','mid','high']:
            sub = [r for r in all_results if r['variant']==vkey and r['level']==lname]
            if not sub: continue
            mae = np.mean([s['mae'] for s in sub])
            mae_std = np.std([s['mae'] for s in sub])
            rmse = np.mean([s['rmse'] for s in sub])
            bias = np.mean([s['bias'] for s in sub])
            gr = np.nanmean([s['grad_ratio'] for s in sub])
            gc = np.nanmean([s['gradcorr'] for s in sub])
            print(f"{VARIANTS[vkey]['label']:<30} {lname:<6} {mae:.3f}±{mae_std:.3f}  {rmse:.3f}      {bias:+.3f}    {gr:.3f}       {gc:.3f}")

    print(f"\n{'='*70}")
    print("Full 结果（从 table1）:")
    print("  low:  MAE=0.327, RMSE=0.423, Bias=+0.011, GradRatio=1.179, GradCorr=0.348")
    print("  mid:  MAE=0.337, RMSE=0.438, Bias=+0.013, GradRatio=1.135, GradCorr=0.363")
    print("  high: MAE=0.360, RMSE=0.466, Bias=+0.032, GradRatio=1.084, GradCorr=0.380")
    print('='*70)

if __name__ == '__main__':
    main()
