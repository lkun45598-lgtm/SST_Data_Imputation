"""Unified head-to-head comparison of all reconstruction methods on held-out
JAXA series 8, same frames + same real-cloud masks per method (fair comparison).

Methods: FNO-CBAM (h12 realmask), DINCAE, U-Net, ConvLSTM, DINEOF, OI/Kriging, KNN/IDW.
Metrics:
  - MAE, RMSE: pixel-level accuracy over the withheld (artificially-masked) region.
  - SSIM: mean standard 7x7 local structural similarity at extra-mask centers.
    Each complete window supplies the standard local statistics, including
    valid pixels outside the extra mask. Windows touching land, natural gaps,
    nonfinite predictions, or array boundaries are excluded. Each field-mask
    pair uses the reference field's valid-ocean P99-P1 SST range for every
    method.
  - grad_ratio: legacy scalar mean-gradient ratio, kept for continuity only.
  - hf_log_ratio: structural fidelity via log-spectral distance in the
    high-wavenumber band; 0 = matches truth's fine-scale/frontal energy on a
    log scale; <0 = over-smoothed (missing high-freq); >0 = over-sharp/noisy.
    Unlike grad_ratio this cannot be fooled by noise whose mean magnitude
    happens to match truth, and unlike a linear power ratio it does not blow
    up when a patch's true high-freq energy is near zero (routine for calm,
    texture-free ocean — see baselines/spectral_metrics.py docstring).
  - gradrmse: pixelwise RMSE of the gradient-magnitude field vs truth (not a
    ratio of means) — penalizes spatially-misplaced structure that grad_ratio
    would miss.
  - gradcorr: Pearson correlation between predicted and true gradient-magnitude
    fields at withheld pixels. Disambiguates WHY hf_log_ratio/gradrmse show
    elevated high-frequency energy: high gradcorr (fronts correctly located,
    amplitude just miscalibrated — a fixable loss-weight issue) vs near-zero/
    negative gradcorr (spatially incoherent noise — a real defect, not a
    calibration gap). hf_log_ratio and gradrmse are both magnitude-only and
    cannot tell these apart on their own (per MAESSTRO, Ocean Sci. 2024, which
    recommends gradient correlation specifically for this reason).
  - Distance-stratified variants (mae/rmse/gradrmse suffixed _edge/_mid/_deep):
    withheld pixels are binned by great-circle-ish distance (km, via per-axis
    lat/lon pixel spacing) to the nearest *actually observed* pixel that day.
    Pixels near the observation boundary can be recovered by local interpolation
    alone; only the _deep bin (>=30km from any real observation) tests genuine
    large-gap extrapolation. Un-stratified metrics conflate the two and can make
    a method that only interpolates well look like it "recovers large gaps".

Usage: python3 baselines/eval_all_methods.py --gpu 7
SSIM only: python3 baselines/eval_all_methods.py --gpu 7 --methods fno \
               --ssim-only --out-name fno.npz
Saves: visualization/paper_figs/cache/all_methods_stats.npz
SSIM-only saves: visualization/paper_figs/cache/ssim_standard_v2_obspreserve/<out-name>
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata

DI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DI))
sys.path.insert(0, str(DI / "baselines"))
sys.path.insert(0, str(DI / "visualization/paper_figs"))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
from baselines.deep_models import build_model
from baselines.dineof import dineof_fill
from baselines.optimal_interpolation import oi_reconstruct
from baselines.spectral_metrics import psd_ratio_at_mask
from baselines.structural_metrics import masked_window_ssim, reference_robust_range
from gen_eval_data import knn_inpaint_2d, gauss_filter
from losses.gradient_masks import central_gradient_magnitude

H12 = DI / "experiments/hourly_data/h12"
# FNO checkpoint 路径：默认用原 h12_realmask（旧 base），可用环境变量 FNO_CKPT 覆盖
# 指向修复 bug 后的 clean 版本 (experiments/jaxa_finetune_h12_realmask_clean/best_model.pth)
import os as _os
FNO_PATH = Path(_os.environ.get(
    "FNO_CKPT", str(DI / "experiments/jaxa_finetune_h12_realmask_clean/best_model.pth")))
CACHE_DIR = DI / "visualization/paper_figs/cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SSIM_CACHE_DIR_NAME = "ssim_standard_v2_obspreserve"
FNO_MEAN, FNO_STD = 299.9221, 2.6919
W = 30
SERIES_ID = 8
GAUSS_SIGMA = 1.0
SEED = 0
NUM_SAMPLES_PER_LEVEL = 9999  # 全量评估，不限制样本数
MASK_LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]
DIST_BINS = [("edge", 0.0, 10.0), ("mid", 10.0, 30.0), ("deep", 30.0, np.inf)]
SSIM_WIN_SIZE = 7
SSIM_METRIC_DEFINITION = (
    "macro-average standard 7x7 local SSIM at artificial extra-mask centers; "
    "observation-preserving reconstruction with complete finite reference-ocean "
    "prediction/reference windows; valid unwithheld observations contribute "
    "window context; reference valid-ocean P99-P1 data range per field-mask pair"
)


def sample_level_indices(candidates, nsamples, requested_level=None, seed=SEED):
    """Reproduce full-run frame sampling when evaluating only one level."""
    rng = np.random.default_rng(seed)
    sampled = []
    for level_name, ratio in MASK_LEVELS:
        idxs = rng.choice(
            candidates, size=min(nsamples, len(candidates)), replace=False
        )
        if requested_level is None or level_name == requested_level:
            sampled.append((level_name, ratio, idxs))
    return sampled


def ssim_result_fields(ssim_value, n_ssim, n_ssim_candidates):
    """Build the per-pair SSIM fields, including eligible-center coverage."""
    coverage = (
        float(n_ssim / n_ssim_candidates)
        if n_ssim_candidates > 0 else np.nan
    )
    return {
        "ssim": float(ssim_value),
        "n_ssim": int(n_ssim),
        "n_ssim_candidates": int(n_ssim_candidates),
        "ssim_coverage": coverage,
    }


def ssim_only_record(*, method, level, target_ratio, idx, n_withheld,
                     data_range, ssim_value, n_ssim, n_ssim_candidates):
    """Build the compact record written by ``--ssim-only`` evaluations."""
    record = {
        "method": method,
        "level": level,
        "target_ratio": float(target_ratio),
        "idx": int(idx),
        "n": int(n_withheld),
        "ssim_data_range": float(data_range),
    }
    record.update(ssim_result_fields(
        ssim_value, n_ssim, n_ssim_candidates
    ))
    return record


def compose_observation_preserving_prediction(prediction, reference,
                                               evaluation_mask):
    """Use predictions only inside the artificial extra mask for SSIM."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    if pred.shape != ref.shape or evaluation.shape != ref.shape:
        raise ValueError("prediction, reference, and evaluation_mask must have the same shape")
    return np.where(evaluation, pred, ref)


def resolve_output_path(out_name, *, ssim_only=False, cache_dir=CACHE_DIR):
    """Resolve an output path and refuse to overwrite isolated SSIM caches."""
    if Path(out_name).name != out_name:
        raise ValueError("out-name must be a file name without directory components")
    normalized_name = out_name if out_name.endswith(".npz") else f"{out_name}.npz"
    base_dir = Path(cache_dir)
    if ssim_only:
        base_dir = base_dir / SSIM_CACHE_DIR_NAME
    output_path = base_dir / normalized_name
    if ssim_only and output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing SSIM cache: {output_path}")
    return output_path


def load_fno(device):
    m = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64,
                              depth=6, cbam_reduction_ratio=16).to(device)
    ck = torch.load(FNO_PATH, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


def load_deep(name, device):
    from pathlib import Path
    m = build_model(name).to(device)
    # Auto-match {name}_*_h12 or {name}_h12
    exp_dir = DI / "baselines/experiments"
    candidates = sorted(exp_dir.glob(f"{name}_*_h12"))
    if not candidates:
        candidates = [exp_dir / f"{name}_h12"]
    ckpt_path = candidates[0] / "best_model.pth"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


def run_torch_model(model, sst_seq, miss_seq, mask, mean, std, device):
    si = sst_seq.copy()
    si[-1] = np.where(mask > 0, mean, si[-1])
    ms = miss_seq.copy().astype(np.float32)
    ms[-1] = mask
    st = torch.from_numpy(np.nan_to_num((si - mean) / std, nan=0.0)).unsqueeze(0).float().to(device)
    mt = torch.from_numpy(ms).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(st, mt)
        pred = out[0] if isinstance(out, (tuple, list)) else out
    return pred.squeeze().cpu().numpy() * std + mean


def linear_interp_2d(sst, mask, land, obs_mask):
    """2D 线性插值 (scipy.griddata). 只用真实观测点; 插值域外用 nearest 兜底.
    sst: 已把 mask 区与非观测区设为 NaN 的场; mask/obs_mask: bool/0-1 数组."""
    H, Wd = sst.shape
    ocean = (land == 0)
    valid = (~mask) & ocean & (~np.isnan(sst)) & (obs_mask > 0)
    if valid.sum() < 3:
        return sst.copy()
    yy, xx = np.mgrid[0:H, 0:Wd]
    pts = np.column_stack([yy[valid], xx[valid]])
    vals = sst[valid]
    interp_mask = mask & ocean
    if interp_mask.sum() == 0:
        return sst.copy()
    tgt = np.column_stack([yy[interp_mask], xx[interp_mask]])
    res = sst.copy()
    res[interp_mask] = griddata(pts, vals, tgt, method="linear", fill_value=np.nan)
    still = interp_mask & np.isnan(res)
    if still.sum() > 0:
        res[still] = griddata(pts, vals,
                              np.column_stack([yy[still], xx[still]]), method="nearest")
    return res


def cubic_interp_2d(sst, mask, land, obs_mask):
    """2D 三次插值; NaN 依次回退 linear -> nearest. 观测点不足回退 linear."""
    H, Wd = sst.shape
    ocean = (land == 0)
    valid = (~mask) & ocean & (~np.isnan(sst)) & (obs_mask > 0)
    if valid.sum() < 10:
        return linear_interp_2d(sst, mask, land, obs_mask)
    yy, xx = np.mgrid[0:H, 0:Wd]
    pts = np.column_stack([yy[valid], xx[valid]])
    vals = sst[valid]
    interp_mask = mask & ocean
    if interp_mask.sum() == 0:
        return sst.copy()
    tgt = np.column_stack([yy[interp_mask], xx[interp_mask]])
    res = sst.copy()
    res[interp_mask] = griddata(pts, vals, tgt, method="cubic", fill_value=np.nan)
    still = interp_mask & np.isnan(res)
    if still.sum() > 0:
        res[still] = griddata(pts, vals,
                              np.column_stack([yy[still], xx[still]]), method="linear", fill_value=np.nan)
    still = interp_mask & np.isnan(res)
    if still.sum() > 0:
        res[still] = griddata(pts, vals,
                              np.column_stack([yy[still], xx[still]]), method="nearest")
    return res


def gradmag(a, valid_ref_mask):
    """Strict central gradient magnitude; returns NaN at invalid stencils."""
    return central_gradient_magnitude(a, valid_ref_mask)[0]


def dist_to_nearest_obs_km(obs_mask, lat, lon):
    """Per-pixel great-circle-ish distance (km) to the nearest pixel where
    obs_mask==1, via EDT with real lat/lon pixel spacing (not assumed square)."""
    dy_km = abs(lat[1] - lat[0]) * 111.0
    lat0 = float(np.median(lat))
    dx_km = abs(lon[1] - lon[0]) * 111.0 * np.cos(np.deg2rad(lat0))
    if obs_mask.sum() == 0:
        return np.full(obs_mask.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(obs_mask == 0, sampling=(dy_km, dx_km)).astype(np.float32)


def stratified_metrics(prefix, e, pred_g, gt_g, dist_km, ev):
    """Distance-stratified MAE/RMSE/GradRMSE dict, keyed f'{metric}_{bin_name}'."""
    out = {}
    d_ev = dist_km[ev]
    for bname, lo, hi in DIST_BINS:
        sel = (d_ev >= lo) & (d_ev < hi)
        out[f"{prefix}n_{bname}"] = int(sel.sum())
        if sel.sum() == 0:
            out[f"{prefix}mae_{bname}"] = np.nan
            out[f"{prefix}rmse_{bname}"] = np.nan
            out[f"{prefix}gradrmse_{bname}"] = np.nan
            continue
        e_sel = e[sel]
        out[f"{prefix}mae_{bname}"] = float(np.abs(e_sel).mean())
        out[f"{prefix}rmse_{bname}"] = float(np.sqrt((e_sel ** 2).mean()))
        gdiff = (pred_g[ev] - gt_g[ev])[sel]
        vv = np.isfinite(gdiff)
        out[f"{prefix}gradrmse_{bname}"] = float(np.sqrt((gdiff[vv] ** 2).mean())) if vv.sum() > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--nsamples", type=int, default=NUM_SAMPLES_PER_LEVEL)
    ap.add_argument("--level", type=str, default=None, choices=["low", "mid", "high"],
                    help="只跑單一雲量檔; 預設全跑")
    ap.add_argument("--methods", type=str, default=None,
                    help="逗號分隔的方法子集 (供多卡並行拆分, 一模型一卡), "
                         "如 'fno' 或 'dincae,unet'; 預設全部7種")
    ap.add_argument("--out-name", type=str, default="all_methods_stats.npz",
                    help="輸出檔名 (並行拆分時各進程用不同檔名, 之後再合併)")
    ap.add_argument("--ssim-only", action="store_true",
                    help=f"只计算标准局部 SSIM，并写入独立 {SSIM_CACHE_DIR_NAME} 目录")
    args = ap.parse_args()
    out_path = resolve_output_path(
        args.out_name,
        ssim_only=args.ssim_only,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    all_methods = ["fno", "dincae", "unet", "convlstm", "lama", "partialconv",
                   "dineof", "oi", "knn", "linear", "cubic"]
    methods = all_methods if args.methods is None else args.methods.split(",")
    print(f"methods={methods}")

    print("loading models...")
    fno = load_fno(device) if "fno" in methods else None
    dincae = load_deep("dincae", device) if "dincae" in methods else None
    unet = load_deep("unet", device) if "unet" in methods else None
    convlstm = load_deep("convlstm", device) if "convlstm" in methods else None
    lama = load_deep("lama", device) if "lama" in methods else None
    partialconv = load_deep("partialconv", device) if "partialconv" in methods else None

    path = H12 / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:].astype(np.float32)
        obs_all = f["original_obs_mask"][:].astype(np.uint8)
        miss_all = f["original_missing_mask"][:].astype(np.uint8)
        land = f["land_mask"][:].astype(np.uint8)
        lat = f["latitude"][:]
        lon = f["longitude"][:]
    ocean = (land == 0)
    cand = [i for i in range(W - 1, sst_all.shape[0]) if (obs_all[i] * ocean).sum() > 4000]
    bank = build_cloud_bank([str(H12 / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]],
                            max_donors=400, seed=7)

    records = []

    sampled_levels = sample_level_indices(cand, args.nsamples, args.level)
    for name, ratio, idxs in sampled_levels:
        gen = RealCloudMaskGenerator(bank, seed=int(ratio * 1000))
        for idx in tqdm(idxs, desc=f"level={name}"):
            elig = (obs_all[idx] * ocean).astype(np.float32)
            if elig.sum() < 2000:
                continue
            mask = gen.generate(elig, target_ratio=ratio)
            ev = (mask * elig).astype(bool)
            if ev.sum() < 200:
                continue
            sst_seq = np.stack([sst_all[max(0, idx - (W - 1) + t)] for t in range(W)])
            miss_seq = np.stack([miss_all[max(0, idx - (W - 1) + t)] for t in range(W)])
            gt = sst_seq[-1]
            valid_ref = (obs_all[idx] * ocean).astype(bool)
            ssim_data_range = reference_robust_range(gt, valid_ref)
            if not args.ssim_only:
                gtg = gradmag(gt, valid_ref)
                # Distance to the nearest observation visible after masking.
                visible_obs = (
                    obs_all[idx].astype(np.uint8) * (mask == 0)
                ).astype(np.uint8)
                dist_km = dist_to_nearest_obs_km(visible_obs, lat, lon)

            preds = {}
            if "fno" in methods:
                preds["fno"] = run_torch_model(fno, sst_seq, miss_seq, mask, FNO_MEAN, FNO_STD, device)
            if "dincae" in methods:
                preds["dincae"] = run_torch_model(dincae, sst_seq, miss_seq, mask, FNO_MEAN, FNO_STD, device)
            if "unet" in methods:
                preds["unet"] = run_torch_model(unet, sst_seq, miss_seq, mask, FNO_MEAN, FNO_STD, device)
            if "convlstm" in methods:
                preds["convlstm"] = run_torch_model(convlstm, sst_seq, miss_seq, mask, FNO_MEAN, FNO_STD, device)
            if "lama" in methods:
                preds["lama"] = run_torch_model(lama, sst_seq, miss_seq, mask, FNO_MEAN, FNO_STD, device)
            if "partialconv" in methods:
                preds["partialconv"] = run_torch_model(partialconv, sst_seq, miss_seq, mask, FNO_MEAN, FNO_STD, device)

            if "dineof" in methods:
                # DINEOF: 30-day stack, mask only the day-30 artificial region
                valid = (np.tile(ocean[None], (W, 1, 1)) == 1) & ~np.isnan(sst_seq)
                valid[-1] = valid[-1] & (mask == 0)
                data = np.where(valid, sst_seq, np.nan)
                dineof_full = dineof_fill(data.astype(np.float32), valid.astype(np.uint8),
                                          k=20, max_iter=20, tol=5e-4, center=True)
                out = gt.copy(); out[mask > 0] = dineof_full[-1][mask > 0]
                preds["dineof"] = out

            if "oi" in methods:
                # OI (spatial kriging from real obs)
                oi_out, _ = oi_reconstruct(gt, mask, obs_all[idx].astype(np.uint8))
                preds["oi"] = oi_out

            if "knn" in methods:
                # KNN/IDW
                ki = gt.copy(); ki[mask > 0] = np.nan
                preds["knn"] = knn_inpaint_2d(ki, mask, obs_all[idx].astype(np.uint8), k=20)

            if "linear" in methods or "cubic" in methods:
                # 经典插值: 输入只保留真实观测点 (mask 区 + 非观测区置 NaN)
                obs_frame = obs_all[idx].astype(np.uint8)
                mask_b = mask.astype(bool)
                ii = gt.copy()
                ii[obs_frame == 0] = np.nan
                ii[mask_b] = np.nan
                if "linear" in methods:
                    preds["linear"] = linear_interp_2d(ii, mask_b, land, obs_frame)
                if "cubic" in methods:
                    preds["cubic"] = cubic_interp_2d(ii, mask_b, land, obs_frame)

            for name_m in methods:
                pred = preds[name_m]
                # pred = gauss_filter(pred, land, GAUSS_SIGMA, fill_region=mask)  # 关闭后处理平滑
                ssim_prediction = compose_observation_preserving_prediction(
                    pred, gt, ev
                )
                ssim_value, n_ssim, n_ssim_candidates = masked_window_ssim(
                    ssim_prediction, gt, ev, valid_ref,
                    win_size=SSIM_WIN_SIZE,
                    data_range=ssim_data_range,
                )
                if args.ssim_only:
                    records.append(ssim_only_record(
                        method=name_m,
                        level=name,
                        target_ratio=ratio,
                        idx=idx,
                        n_withheld=ev.sum(),
                        data_range=ssim_data_range,
                        ssim_value=ssim_value,
                        n_ssim=n_ssim,
                        n_ssim_candidates=n_ssim_candidates,
                    ))
                    continue

                e = pred[ev] - gt[ev]
                pg = gradmag(pred, valid_ref)
                vv = ev & np.isfinite(pg) & np.isfinite(gtg)
                grad_ratio = float(pg[vv].mean() / (gtg[vv].mean() + 1e-9)) if vv.sum() > 0 else np.nan
                gdiff_all = (pg - gtg)[vv]
                gradrmse = float(np.sqrt((gdiff_all ** 2).mean())) if vv.sum() > 0 else np.nan
                gradmae = float(np.abs(gdiff_all).mean()) if vv.sum() > 0 else np.nan   # gradient MAE (梯度场绝对误差)
                gradcorr = float(np.corrcoef(pg[vv], gtg[vv])[0, 1]) if vv.sum() > 1 else np.nan
                psd_curve, hf_log_ratio = psd_ratio_at_mask(pred, gt, ocean, mask)
                bias = float(e.mean())                                 # signed bias (pred - target, >0偏暖/<0偏冷)
                crmse = float(np.sqrt(((e - e.mean()) ** 2).mean()))   # centered RMSE (去掉整体偏差后的空间变化误差)
                medae = float(np.median(np.abs(e)))                    # median absolute error (抗极端值)
                q75, q25 = np.percentile(np.abs(e), [75, 25])
                iqr = float(q75 - q25)                                 # |error| 的四分位距 (误差离散度)
                rec = dict(
                    method=name_m, level=name, target_ratio=ratio, idx=int(idx),
                    mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                    bias=bias, crmse=crmse, medae=medae, iqr=iqr,
                    grad_ratio=grad_ratio, hf_log_ratio=hf_log_ratio, gradrmse=gradrmse,
                    gradmae=gradmae, gradcorr=gradcorr, psd_curve=psd_curve, n=int(ev.sum()),
                )
                rec.update(ssim_result_fields(
                    ssim_value, n_ssim, n_ssim_candidates
                ))
                rec.update(stratified_metrics("", e, pg, gtg, dist_km, ev))
                records.append(rec)

    save_payload = {"records": np.array(records, dtype=object)}
    if args.ssim_only:
        save_payload.update(
            metric_definition=np.array(SSIM_METRIC_DEFINITION),
            source_mode=np.array("prediction regeneration; SSIM-only"),
        )
    np.savez_compressed(out_path, **save_payload)
    print(f"\nsaved {len(records)} records -> {out_path}")

    if args.ssim_only:
        print(f"\n{'method':<10}{'level':<6}{'SSIM':>10}{'coverage':>12}{'n':>6}")
        for name_m in methods:
            for lvl, _, _ in sampled_levels:
                sub = [
                    r for r in records
                    if r["method"] == name_m and r["level"] == lvl
                ]
                if not sub:
                    continue
                ssim_mean = np.nanmean([r["ssim"] for r in sub])
                coverage = np.nanmean([r["ssim_coverage"] for r in sub])
                print(
                    f"{name_m:<10}{lvl:<6}{ssim_mean:>10.4f}"
                    f"{coverage:>12.4f}{len(sub):>6}"
                )
        return

    # ===== 主汇总表：对每个 method×level，对样本做 mean / median / IQR 汇总 =====
    # 每帧已用相同的 extra mask 评估过全部方法（fair comparison），此处只做统计汇总。
    print(f"\n{'method':<10}{'level':<6}{'MAE':>8}{'MAE_med':>8}{'MAE_IQR':>8}"
          f"{'RMSE':>8}{'cRMSE':>8}{'Bias':>8}{'gradMAE':>9}{'gradRMSE':>9}{'gradCorr':>9}"
          f"{'hf_logR':>9}{'SSIM':>8}{'SSIMcov':>9}{'n':>5}")
    for name_m in methods:
        for lvl, _, _ in sampled_levels:
            sub = [r for r in records if r["method"] == name_m and r["level"] == lvl]
            if not sub:
                continue
            mae_vals = [r["mae"] for r in sub]
            mae = np.mean(mae_vals)
            mae_med = np.median([r["medae"] for r in sub])      # 样本级中位AE的中位数
            mae_iqr = np.median([r["iqr"] for r in sub])        # 样本级|error| IQR的中位数
            rmse = np.mean([r["rmse"] for r in sub])
            crmse = np.mean([r["crmse"] for r in sub])
            bias = np.mean([r["bias"] for r in sub])
            gmae = np.nanmean([r["gradmae"] for r in sub])
            grmse = np.nanmean([r["gradrmse"] for r in sub])
            gcorr = np.nanmean([r["gradcorr"] for r in sub])
            hf_med = np.nanmedian([r["hf_log_ratio"] for r in sub])
            ssim_mean = np.nanmean([r["ssim"] for r in sub])
            ssim_coverage = np.nanmean([r["ssim_coverage"] for r in sub])
            print(f"{name_m:<10}{lvl:<6}{mae:>8.4f}{mae_med:>8.4f}{mae_iqr:>8.4f}"
                  f"{rmse:>8.4f}{crmse:>8.4f}{bias:>+8.4f}{gmae:>9.4f}{grmse:>9.4f}{gcorr:>9.3f}"
                  f"{hf_med:>9.3f}{ssim_mean:>8.4f}{ssim_coverage:>9.3f}{len(sub):>5}")


    print(f"\ndistance-stratified (pooled across levels) — edge(<10km) / mid(10-30km) / deep(>=30km):")
    print(f"{'method':<10}{'bin':<6}{'MAE':>8}{'RMSE':>8}{'gradRMSE':>10}{'n':>8}")
    for name_m in methods:
        sub = [r for r in records if r["method"] == name_m]
        if not sub:
            continue
        for bname, _, _ in DIST_BINS:
            mae_vals = [r[f"mae_{bname}"] for r in sub if r.get(f"n_{bname}", 0) > 0]
            rmse_vals = [r[f"rmse_{bname}"] for r in sub if r.get(f"n_{bname}", 0) > 0]
            grmse_vals = [r[f"gradrmse_{bname}"] for r in sub if r.get(f"n_{bname}", 0) > 0]
            n_tot = sum(r.get(f"n_{bname}", 0) for r in sub)
            if not mae_vals:
                continue
            print(f"{name_m:<10}{bname:<6}{np.nanmean(mae_vals):>8.4f}{np.nanmean(rmse_vals):>8.4f}"
                  f"{np.nanmean(grmse_vals):>10.4f}{n_tot:>8}")


if __name__ == "__main__":
    main()
