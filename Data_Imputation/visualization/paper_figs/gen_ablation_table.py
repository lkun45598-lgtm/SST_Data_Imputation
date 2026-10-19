"""Component-ablation TABLE (no post-processing smoothing — aligns with Table 1).

Evaluates ONLY the deep-learning ablation variants on held-out series 8, under
the SAME frames/masks as gen_fig8_stats.py (small, real-cloud) and
gen_fig8_largeblob.py (large blobs), but WITHOUT the Gaussian sigma=1 post-filter
those figure caches use. This makes every metric — including gradient ratio —
directly comparable to the main-result Table 1, which states
"Predictions use no post-processing smoothing".

Variants (rows):  FNO only  ->  + CBAM (+grad)  ->  + boundary (deployed=main)
Metrics (cols):   MAE, boundary MAE, gradient MAE, gradient ratio (->1),
                  gradient correlation.

Output: cache/ablation_table_stats.npz  +  a LaTeX table printed to stdout.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy.spatial import cKDTree  # noqa: F401 (kept for parity / future baselines)
from scipy.ndimage import binary_dilation, binary_erosion

DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
ABLATION_DIR = DATA_IMPUTATION_DIR / "ablation"
sys.path.insert(0, str(DATA_IMPUTATION_DIR))
sys.path.insert(0, str(ABLATION_DIR))

from models.fno_cbam_ablation import FNO_CBAM_Ablation
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
from losses.gradient_masks import central_gradient_magnitude

DATA_DIR = DATA_IMPUTATION_DIR / "experiments/hourly_data/h12"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAIN_MODEL_PATH = DATA_IMPUTATION_DIR / "experiments/jaxa_finetune_h12_realmask_clean/best_model.pth"

WINDOW_SIZE = 30
GPU_ID = 0
SERIES_ID = 8
SEED = 0
# 全部候选帧：219个目标场 × 3个缺失等级 = 657个field-mask pairs
NUM_SMALL = 9999                  # per level — 取全部 real-cloud small-mask 候选帧
NUM_LARGE = 0                     # Table 1 只用 real-cloud, large-blob 不进表, 跳过
MASK_LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]
MIN_BLOB_SIZE, MAX_BLOB_SIZE = 80, 200

# DL variants only (rows of the ablation table). fno_only/cbam_basic/cbam_boundary
# load via FNO_CBAM_Ablation; "main" is the deployed model (论文主结果).
ABLATION_VARIANTS = {
    "fno_only":      dict(use_cbam=False),
    "cbam_basic":    dict(use_cbam=True),
    "cbam_boundary": dict(use_cbam=True),
}


class LargeBlobMaskGenerator:
    """80-200 px square patches — identical to gen_fig8_largeblob.py."""
    def __init__(self, mask_ratio, min_size=MIN_BLOB_SIZE, max_size=MAX_BLOB_SIZE, seed=None):
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_mask):
        H, W = valid_mask.shape
        art = np.zeros((H, W), dtype=np.float32)
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return art
        target = int(valid_count * self.mask_ratio)
        current = 0
        ys, xs = np.where(valid_mask == 1)
        if len(ys) == 0:
            return art
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        attempts, max_attempts = 0, 3000
        while current < target and attempts < max_attempts:
            size = self.rng.integers(self.min_size, self.max_size + 1)
            if y_max - size < y_min or x_max - size < x_min:
                attempts += 1
                continue
            y0 = self.rng.integers(y_min, max(y_min + 1, y_max - size + 1))
            x0 = self.rng.integers(x_min, max(x_min + 1, x_max - size + 1))
            y1 = min(y0 + size, H)
            x1 = min(x0 + size, W)
            region = valid_mask[y0:y1, x0:x1].copy()
            art[y0:y1, x0:x1] = np.where(region == 1, 1.0, art[y0:y1, x0:x1])
            current = (art * valid_mask).sum()
            attempts += 1
        return art


def gradmag(a, valid_ref_mask):
    return central_gradient_magnitude(a, valid_ref_mask)[0]


def boundary_pixels(mask, dilation=2):
    dilated = binary_dilation(mask.astype(bool), iterations=dilation)
    eroded = binary_erosion(mask.astype(bool), iterations=dilation)
    return dilated & ~eroded


def predict_model(model, sst_seq, miss_seq, art_mask, norm_mean, norm_std, device):
    sst_input = sst_seq.copy()
    sst_input[-1] = np.where(art_mask > 0, norm_mean, sst_input[-1])
    mask_seq = miss_seq.copy().astype(np.float32)
    mask_seq[-1] = art_mask
    sst_norm = np.nan_to_num((sst_input - norm_mean) / norm_std, nan=0.0)
    st = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mt = torch.from_numpy(mask_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(st, mt)
    pred_k = pred.squeeze().cpu().numpy() * norm_std + norm_mean
    # 返回纯模型输出场 (不把GT合成回观测区), 与 baselines/eval_all_methods.py
    # 的 run_torch_model 完全一致 —— 这样梯度指标(gradmag/ratio)在 mask 边界
    # 用的是模型输出邻居而非真实GT邻居, 口径与 Table 1 对齐。
    # MAE/RMSE 只读 masked 像元, 那里合成与否都等于 pred_k, 故不受影响。
    return pred_k   # NO Gaussian post-filter, NO GT compositing


def load_ablation(variant, device):
    p = ABLATION_DIR / "experiments" / variant / "best_model.pth"
    if not p.exists():
        return None, None, None
    ckpt = torch.load(p, map_location=device, weights_only=False)
    cfg = ABLATION_VARIANTS[variant]
    model = FNO_CBAM_Ablation(out_size=(451, 351), modes1=80, modes2=64,
                              width=64, depth=6, cbam_reduction_ratio=16,
                              use_cbam=cfg["use_cbam"]).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        assert not cfg["use_cbam"], f"{variant} 缺参数但 use_cbam=True: {missing[:3]}"
        assert all("cbam" in k.lower() for k in missing), \
            f"{variant} 缺失非CBAM参数: {[k for k in missing if 'cbam' not in k.lower()][:3]}"
    assert not unexpected, f"{variant} 有多余参数: {unexpected[:3]}"
    model.eval()
    return model, ckpt.get("norm_mean", 299.9221), ckpt.get("norm_std", 2.6919)


def load_main(device):
    ckpt = torch.load(MAIN_MODEL_PATH, map_location=device, weights_only=False)
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64,
                                  width=64, depth=6, cbam_reduction_ratio=16).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("norm_mean", 299.9221), ckpt.get("norm_std", 2.6919)


def metrics_for(pred, gt, gtg, valid_ref_mask, eval_mask, bnd_mask):
    """与 baselines/eval_all_methods.py / Table 1 完全同口径的指标。"""
    diff = pred[eval_mask] - gt[eval_mask]
    if eval_mask.sum() > 0:
        mae = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))
        bias = float(diff.mean())
    else:
        mae = rmse = bias = np.nan
    bnd_mae = (float(np.abs(pred[bnd_mask] - gt[bnd_mask]).mean())
               if bnd_mask.sum() else np.nan)
    pg = gradmag(pred, valid_ref_mask)
    vv = eval_mask & np.isfinite(pg) & np.isfinite(gtg)
    if vv.sum() > 1:
        gdiff = (pg - gtg)[vv]
        grad_mae = float(np.abs(gdiff).mean())
        grad_ratio = float(pg[vv].mean() / (gtg[vv].mean() + 1e-9))
        grad_corr = float(np.corrcoef(pg[vv], gtg[vv])[0, 1])
    else:
        grad_mae = grad_ratio = grad_corr = np.nan
    return dict(mae=mae, rmse=rmse, bias=bias, bnd_mae=bnd_mae,
                grad_mae=grad_mae, grad_ratio=grad_ratio, grad_corr=grad_corr)


def build_seq(sst_all, miss_all, idx):
    sst_seq = np.zeros((WINDOW_SIZE, *sst_all.shape[1:]), dtype=np.float32)
    miss_seq = np.zeros_like(sst_seq, dtype=np.float32)
    for t in range(WINDOW_SIZE):
        s = max(0, idx - (WINDOW_SIZE - 1) + t)
        sst_seq[t] = sst_all[s]
        miss_seq[t] = miss_all[s]
    return sst_seq, miss_seq


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Loading DL variants (no-smoothing eval)...")
    models = {}
    for v in ABLATION_VARIANTS:
        m, mu, sd = load_ablation(v, device)
        if m is None:
            print(f"  ablation/{v}: MISSING"); continue
        models[v] = (m, mu, sd); print(f"  ablation/{v}: loaded")
    models["main"], _mm, _ms = (load_main(device)[0], None, None)
    _m, mmean, mstd = load_main(device)
    models["main"] = (_m, mmean, mstd); print("  main (deployed): loaded")

    with h5py.File(DATA_DIR / f"jaxa_knn_filled_{SERIES_ID:02d}.h5", "r") as f:
        sst_all = f["sst_data"][:]; obs_all = f["original_obs_mask"][:]
        miss_all = f["original_missing_mask"][:]; land = f["land_mask"][:]
    T = sst_all.shape[0]; ocean = 1 - land
    start = WINDOW_SIZE - 1
    candidates = np.array([
        i for i in range(start, T)
        if (obs_all[i] * ocean).sum() > 4000
    ])

    # cloud bank for the SMALL real-cloud masks (same as gen_fig8_stats.py)
    bank = build_cloud_bank([str(DATA_DIR / f"jaxa_knn_filled_{i:02d}.h5")
                             for i in [0, 3, 6]], max_donors=400, seed=7)

    records = []

    # ---- SMALL real-cloud masks (RNG/gen identical to gen_fig8_stats.py) ----
    rng = np.random.default_rng(SEED)
    for name, level in MASK_LEVELS:
        idxs = rng.choice(candidates, size=min(NUM_SMALL, len(candidates)),
                          replace=False)
        gen = RealCloudMaskGenerator(bank, seed=int(level * 1000))
        for idx in tqdm(idxs, desc=f"small/{name}", leave=False):
            sst_seq, miss_seq = build_seq(sst_all, miss_all, idx)
            elig = (obs_all[idx] * ocean)
            if elig.sum() < 2000:
                continue
            mask = gen.generate(elig.astype(np.float32), target_ratio=level)
            ev = (mask * elig).astype(bool)
            valid_ref = elig.astype(bool)
            gt = sst_seq[-1].copy(); gtg = gradmag(gt, valid_ref)
            bnd = boundary_pixels(mask, dilation=2) & ev
            for v, (mdl, mu, sd) in models.items():
                pred = predict_model(mdl, sst_seq, miss_seq, mask, mu, sd, device)
                rec = dict(method=v, regime="small", level=name, idx=int(idx))
                rec.update(metrics_for(pred, gt, gtg, valid_ref, ev, bnd))
                records.append(rec)

    # ---- LARGE blobs (RNG/gen identical to gen_fig8_largeblob.py) ----
    rng = np.random.default_rng(SEED)
    for name, level in MASK_LEVELS:
        idxs = rng.choice(candidates, size=min(NUM_LARGE, len(candidates)),
                          replace=False)
        gen = LargeBlobMaskGenerator(mask_ratio=level, seed=int(level * 1000) + 7)
        for idx in tqdm(idxs, desc=f"large/{name}", leave=False):
            sst_seq, miss_seq = build_seq(sst_all, miss_all, idx)
            elig = (obs_all[idx] * ocean)
            if elig.sum() < 2000:
                continue
            art = gen.generate(elig.astype(np.float32))
            if art.sum() == 0:
                continue
            ev = (art * elig).astype(bool)
            valid_ref = elig.astype(bool)
            gt = sst_seq[-1].copy(); gtg = gradmag(gt, valid_ref)
            bnd = boundary_pixels(art, dilation=2) & ev
            for v, (mdl, mu, sd) in models.items():
                pred = predict_model(mdl, sst_seq, miss_seq, art, mu, sd, device)
                rec = dict(method=v, regime="large", level=name, idx=int(idx))
                rec.update(metrics_for(pred, gt, gtg, valid_ref, ev, bnd))
                records.append(rec)

    out = CACHE_DIR / "ablation_table_stats.npz"
    np.savez_compressed(out, records=np.array(records, dtype=object))
    print(f"\nsaved -> {out}  ({len(records)} records)")

    emit_table(records)


# 3 行消融: 最后一行=部署模型(=+boundary 完整配置, 与主结果同权重)。
# 受控 cbam_boundary(val_mae 0.316) 与部署行冗余且梯度指标为离群点, 不单列。
ROWS = [
    ("fno_only",   "FNO only"),
    ("cbam_basic", "+ CBAM (+grad)"),
    ("main",       "+ boundary (full model)"),
]


TABLE1_REGIME = "small"   # real-cloud masks = Table 1 的评估协议


def _mean(records, method, metric, level=None):
    v = [r[metric] for r in records
         if r["method"] == method and r["regime"] == TABLE1_REGIME
         and (level is None or r["level"] == level)
         and np.isfinite(r[metric])]
    return float(np.mean(v)) if v else np.nan


def emit_table(records):
    # 与 Table 1 完全同列: MAE(30/55/75) | RMSE | Bias | Grad MAE | Grad Ratio
    print("\n=== Console summary (no smoothing, real-cloud full-set = Table 1 protocol) ===")
    hdr = (f"{'configuration':<24}{'MAE30':>8}{'MAE55':>8}{'MAE75':>8}"
           f"{'RMSE':>8}{'Bias':>8}{'gMAE':>9}{'gRatio':>8}")
    print(hdr)
    for mk, lbl in ROWS:
        m30 = _mean(records, mk, "mae", "low")
        m55 = _mean(records, mk, "mae", "mid")
        m75 = _mean(records, mk, "mae", "high")
        rmse = _mean(records, mk, "rmse")
        bias = _mean(records, mk, "bias")
        gm = _mean(records, mk, "grad_mae")
        gr = _mean(records, mk, "grad_ratio")
        print(f"{lbl:<24}{m30:>8.3f}{m55:>8.3f}{m75:>8.3f}"
              f"{rmse:>8.3f}{bias:>+8.3f}{gm:>9.4f}{gr:>8.3f}")

    # ---- LaTeX (same column layout as Table~\ref{tab:overall}) ----
    print("\n=== LaTeX (paste into §Ablation) ===\n")
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Component ablation on the held-out test series (Series~8, "
        r"219 target fields and 657 field--mask pairs), stratified by missingness "
        r"level. Evaluation protocol and "
        r"metrics are identical to Table~\ref{tab:overall}: real-cloud-shaped "
        r"masks, no post-processing smoothing, metrics on the withheld ocean "
        r"pixels. Lower is better for all metrics except gradient ratio "
        r"($\rightarrow 1$ ideal). The deployed configuration corresponds to the "
        r"main-result model.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{MAE (K) $\downarrow$} & RMSE & Bias "
        r"& \multicolumn{2}{c}{Gradient}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){7-8}",
        r"Configuration & 30\% & 55\% & 75\% & $\downarrow$ & (K) "
        r"& MAE~$\downarrow$ & Ratio~$\rightarrow 1$\\",
        r"\midrule",
    ]
    for mk, lbl in ROWS:
        m30 = _mean(records, mk, "mae", "low")
        m55 = _mean(records, mk, "mae", "mid")
        m75 = _mean(records, mk, "mae", "high")
        rmse = _mean(records, mk, "rmse")
        bias = _mean(records, mk, "bias")
        gm = _mean(records, mk, "grad_mae")
        gr = _mean(records, mk, "grad_ratio")
        latex_lbl = lbl.replace("+", r"$+$")
        lines.append(f"{latex_lbl} & {m30:.3f} & {m55:.3f} & {m75:.3f} "
                     f"& {rmse:.3f} & ${bias:+.3f}$ & {gm:.4f} & {gr:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(lines))


# __TABLE_END__


if __name__ == "__main__":
    main()
