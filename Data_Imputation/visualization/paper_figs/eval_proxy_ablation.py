#!/usr/bin/env python3
"""
Proxy construction 消融评估：Full vs 去Gaussian vs 去3D KNN
Series 8 held-out evaluation，三组 missingness level，全量评估
输出：cache/proxy_ablation_stats.npz + 对比表 + 存重建结果供画图
"""
import sys
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm

# Add project paths
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baselines"))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
from losses.gradient_masks import central_gradient_magnitude

DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

# 固定归一化参数
NORM_MEAN = 299.9221
NORM_STD = 2.6919
W = 30
SERIES_ID = 8
SEED = 0
NUM_SAMPLES = 9999  # 全量

# 三个变体
VARIANTS = {
    'full': {
        'h5_path': REPO / 'experiments/hourly_data/h12/jaxa_knn_filled_08.h5',
        'model_path': REPO / 'experiments/jaxa_finetune_h12_realmask_clean/best_model.pth',
        'label': 'Full (Temporal+Gaussian+KNN)'
    },
    'noknn': {
        'h5_path': REPO / 'experiments/hourly_data/h12_noknn/jaxa_knn_filled_08.h5',
        'model_path': REPO / 'experiments/jaxa_finetune_h12_noknn_realmask_clean/best_model.pth',
        'label': 'No KNN (Temporal+Gaussian)'
    },
    'nogauss': {
        'h5_path': REPO / 'experiments/hourly_data/h12_nogauss/jaxa_knn_filled_08.h5',
        'model_path': REPO / 'experiments/jaxa_finetune_h12_nogauss_realmask_clean/best_model.pth',
        'label': 'No Gaussian (Temporal+KNN)'
    }
}

LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]

def load_model(ckpt_path):
    """加载 FNO 模型"""
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def run_model(model, sst_seq, miss_seq, mask):
    """单帧推理"""
    si = sst_seq.copy()
    si[-1] = np.where(mask > 0, NORM_MEAN, si[-1])
    ms = miss_seq.copy().astype(np.float32)
    ms[-1] = mask
    st = torch.from_numpy(np.nan_to_num((si - NORM_MEAN) / NORM_STD, nan=0.0)).unsqueeze(0).float().to(DEVICE)
    mt = torch.from_numpy(ms).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(st, mt).squeeze().cpu().numpy()
    return pred * NORM_STD + NORM_MEAN

def gradmag(a, valid_ref_mask):
    """严格中心差分；无效中心或四邻域不生成梯度样本。"""
    return central_gradient_magnitude(a, valid_ref_mask)[0]

def evaluate_variant(variant_key, gt_all, obs_all, land):
    """评估一个变体。
    gt_all: 与主比较实验相同的JAXA参考SST场 (T,H,W)，
            三变体共用同一份，保证指标与主结果可直接比较。
    obs_all/land: 同样全局共用。
    只有输入 proxy (sst_all/miss_all) 因变体而不同。
    """
    cfg = VARIANTS[variant_key]
    print(f"\n{'='*70}")
    print(f"Variant: {cfg['label']}")
    print(f"  H5(输入proxy): {cfg['h5_path'].name}")
    print(f"  Model: {cfg['model_path'].relative_to(REPO)}")
    print('='*70)

    # 加载模型 + 该变体自己的输入 proxy
    model = load_model(cfg['model_path'])

    with h5py.File(cfg['h5_path'], 'r') as f:
        sst_all = f['sst_data'][:].astype(np.float32)   # 输入proxy(变体不同)
        miss_all = f['original_missing_mask'][:].astype(np.uint8)

    ocean = (land == 0)

    # 候选帧：有足够观测的
    cand = [i for i in range(W-1, sst_all.shape[0]) if (obs_all[i] * ocean).sum() > 4000]

    # 云模板库
    h5_paths = [str(REPO / f'experiments/hourly_data/h12/jaxa_knn_filled_{i:02d}.h5') for i in [0,3,6]]
    bank = build_cloud_bank(h5_paths, max_donors=400, seed=7)

    results = []
    rng = np.random.default_rng(SEED)

    for level_name, ratio in LEVELS:
        idxs = rng.choice(cand, size=min(NUM_SAMPLES, len(cand)), replace=False)
        gen = RealCloudMaskGenerator(bank, seed=int(ratio * 1000))

        for idx in tqdm(idxs, desc=f'  {level_name}', leave=False):
            elig = (obs_all[idx] * ocean).astype(np.float32)
            if elig.sum() < 2000:
                continue

            mask = gen.generate(elig, target_ratio=ratio)
            ev = (mask * elig).astype(bool)
            if ev.sum() < 200:
                continue

            # 准备序列：输入用该变体自己的 proxy，真值用统一干净真值
            sst_seq = np.stack([sst_all[max(0, idx - (W-1) + t)] for t in range(W)])
            miss_seq = np.stack([miss_all[max(0, idx - (W-1) + t)] for t in range(W)])
            gt = gt_all[idx]

            # 推理
            pred = run_model(model, sst_seq, miss_seq, mask)

            # 计算指标
            e = pred[ev] - gt[ev]
            valid_ref = (obs_all[idx].astype(bool) & (land == 0))
            gtg = gradmag(gt, valid_ref)
            pg = gradmag(pred, valid_ref)
            vv = ev & np.isfinite(pg) & np.isfinite(gtg)

            grad_ratio = float(pg[vv].mean() / (gtg[vv].mean() + 1e-9)) if vv.sum() > 0 else np.nan
            gradcorr = float(np.corrcoef(pg[vv], gtg[vv])[0,1]) if vv.sum() > 1 else np.nan
            gdiff = (pg - gtg)[vv]
            gradrmse = float(np.sqrt((gdiff**2).mean())) if vv.sum() > 0 else np.nan
            gradmae = float(np.abs(gdiff).mean()) if vv.sum() > 0 else np.nan

            bias = float(e.mean())
            crmse = float(np.sqrt(((e - e.mean())**2).mean()))
            medae = float(np.median(np.abs(e)))
            q75, q25 = np.percentile(np.abs(e), [75, 25])
            iqr = float(q75 - q25)

            results.append({
                'variant': variant_key,
                'level': level_name,
                'idx': int(idx),
                'mae': float(np.abs(e).mean()),
                'rmse': float(np.sqrt((e**2).mean())),
                'bias': bias,
                'crmse': crmse,
                'medae': medae,
                'iqr': iqr,
                'grad_ratio': grad_ratio,
                'gradcorr': gradcorr,
                'gradrmse': gradrmse,
                'gradmae': gradmae,
                'n': int(ev.sum())
            })

    return results

def main():
    print("="*70)
    print("Proxy Construction Ablation Evaluation")
    print("  Series 8 (219 target fields, 657 field-mask pairs), 3 missingness levels")
    print("  Variants: Full / No Gaussian / No 3D KNN")
    print(f"  Device: {DEVICE}")
    print("="*70)

    print("\nLoading the common JAXA reference SST used by the main comparison...")
    gt_h5 = REPO / 'experiments/hourly_data/h12/jaxa_knn_filled_08.h5'
    with h5py.File(gt_h5, 'r') as f:
        gt_all = f['sst_data'][:].astype(np.float32)
        obs_all = f['original_obs_mask'][:].astype(np.uint8)
        land = f['land_mask'][:].astype(np.uint8)
    print(f"  GT shape: {gt_all.shape}, frames: {gt_all.shape[0]}")
    print(f"  Observed pixels per frame (mean): {(obs_all*(land==0)).sum(axis=(1,2)).mean():.0f}")
    print()

    all_results = []
    for vkey in ['full', 'noknn', 'nogauss']:
        all_results.extend(evaluate_variant(vkey, gt_all, obs_all, land))

    # 保存
    records = np.array(all_results, dtype=object)
    out_path = CACHE / 'proxy_ablation_stats.npz'
    np.savez_compressed(out_path, records=records)
    print(f"\n{'='*70}")
    print(f"Saved: {out_path}")
    print(f"  Total records: {len(records)}")

    # 汇总表
    print(f"\n{'='*70}")
    print("Summary Table (mean ± std)")
    print('='*70)
    print(f"{'Variant':<35} {'Level':<6} {'MAE':<13} {'RMSE':<13} {'Bias':<10} {'GradRatio':<13} {'GradCorr':<10}")
    print('-'*70)

    for vkey in ['full', 'noknn', 'nogauss']:
        vlabel = VARIANTS[vkey]['label']
        for lname in ['low', 'mid', 'high']:
            subset = [r for r in all_results if r['variant']==vkey and r['level']==lname]
            if not subset:
                continue

            mae = [r['mae'] for r in subset]
            rmse = [r['rmse'] for r in subset]
            bias = [r['bias'] for r in subset]
            gr = [r['grad_ratio'] for r in subset]
            gc = [r['gradcorr'] for r in subset]

            print(f"{vlabel:<35} {lname:<6} "
                  f"{np.mean(mae):.3f}±{np.std(mae):.3f}  "
                  f"{np.mean(rmse):.3f}±{np.std(rmse):.3f}  "
                  f"{np.mean(bias):+.3f}    "
                  f"{np.mean(gr):.3f}±{np.std(gr):.3f}  "
                  f"{np.mean(gc):.3f}")
    print('='*70)

if __name__ == '__main__':
    main()
