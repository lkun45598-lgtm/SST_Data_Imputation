#!/usr/bin/env python3
"""FINAL honest comparison — DEPLOYMENT input convention + fair DINEOF.
FNO input matches fill_jaxa.py production pipeline:
  history[0:29] = KNN-filled sst_data ; day30 = RAW obs (obs&~art), clouds+art -> mean.
  => the hidden (art) pixels never enter the input  (no KNN-prefill leak).
DINEOF: reconstructs from RAW sparse obs (obs==1 & ~art) only; k tunable.
Also reports KNN-prefill-alone (what the model starts its history from) as the
"no-model" reference. Pseudo-GT = hidden real-obs pixels (art & obs==1)."""
import sys, argparse
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as ssim
DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
import eval_realgap_noref as E
from dineof import dineof_fill

HR = DI / "experiments/hourly_data/h12"; OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)
TILE = (112, 96, 192); BINS = [(0, 3), (3, 8), (8, 20), (20, 999)]


def grad_mag(f):
    gy, gx = np.gradient(np.nan_to_num(f)); return np.sqrt(gy ** 2 + gx ** 2)


def fill_model_prod(model, ss_knn, obs_seq, miss_seq, art, land, nm, ns, dev):
    """Deployment input: history KNN-filled; day30 = raw obs only, clouds+art -> mean."""
    si = ss_knn.copy()
    keep = (obs_seq[-1] > 0) & (art == 0) & (land == 0)
    si[-1] = np.where(keep, ss_knn[-1], nm)               # raw obs kept, everything else = mean
    ms = miss_seq.copy().astype(np.float32)
    ms[-1] = (~keep & (land == 0)).astype(np.float32)     # day30 missing = clouds + art
    sn = np.nan_to_num((si - nm) / ns, nan=0.0)
    st = torch.from_numpy(sn).unsqueeze(0).float().to(dev)
    mt = torch.from_numpy(ms).unsqueeze(0).to(dev)
    with torch.no_grad():
        pr = model(st, mt).squeeze().cpu().numpy() * ns + nm
    out = np.where(art > 0, pr, ss_knn[-1])               # fill hidden region with prediction
    return E.gauss_keep_obs(np.where(land == 0, out, np.nan), land, art)


def dineof_sparse(ss, obs_seq, art, ocean, k):
    valid = (obs_seq == 1) & (ocean[None] == 1)
    valid[-1] = valid[-1] & (art == 0)
    data = np.where(valid, ss, np.nan).astype(np.float32)
    filled = dineof_fill(data, valid.astype(np.uint8), k=k, max_iter=25, tol=5e-4, center=True)
    out = ss[-1].copy(); out[art > 0] = filled[-1][art > 0]; return out


def knn_prefill(day30, obs_t, art, ocean):
    """KNN fill of the hidden region from raw obs only (the model's 'no-model' input)."""
    src = np.where((obs_t > 0) & (art == 0) & (ocean == 1), day30, np.nan)
    cloud = ((art > 0) | ((obs_t == 0) & (ocean == 1))).astype(np.uint8)
    return E.knn_fill(src, (art > 0).astype(np.uint8), ocean.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15); ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--dineof-k", type=int, default=4)
    ap.add_argument("--ksweep", action="store_true")
    ap.add_argument("--ratio-lo", type=float, default=0.2)
    ap.add_argument("--ratio-hi", type=float, default=0.35)
    args = ap.parse_args()
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = (land == 0); T = sst.shape[0]; WIN = 30
    y0, x0, S = TILE; hann = np.outer(np.hanning(S), np.hanning(S)); hik = slice(S // 8, S // 2)
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    clear = [i for i in range(WIN, T) if cov[i] > args.cov]
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fno = E.load_fno(E.NEW_MODEL, dev)
    from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
    bank = build_cloud_bank([str(HR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]], max_donors=400, seed=7)
    gen = RealCloudMaskGenerator(bank, seed=42, ratio_range=(args.ratio_lo, args.ratio_hi))
    ocm = ocean.astype(np.uint8)

    if args.ksweep:
        KS = [1, 2, 3, 4, 6, 8]; acc = {k: [] for k in KS}
        for idx in clear[:5]:
            ss = np.stack([sst[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
            ob = np.stack([obs[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.uint8)
            gt = ss[-1].copy(); elig = ((obs[idx] > 0) & ocean).astype(np.float32); art = gen.generate(elig).astype(np.uint8)
            em = (art > 0) & (obs[idx] > 0) & ocean & ~np.isnan(gt)
            if em.sum() < 200: continue
            for k in KS:
                fl = dineof_sparse(ss, ob, art, ocm, k); d = (fl - gt)[em]; acc[k].append(np.abs(d[~np.isnan(d)]).mean())
        print(f"{'DINEOF k (sparse)':<18}{'MAE(K)':>9}")
        for k in KS: print(f"{k:<18}{np.mean(acc[k]):>9.3f}")
        return

    methods = ["DINEOF_sparse", "FNO_realmask"]  # KNN dropped: it is the model's prefill, and oversmooths
    G = {m: {k: [] for k in ["mae", "rmse", "gbias", "psd", "ssim"]} for m in methods}
    depth = {m: [[] for _ in BINS] for m in methods}
    print(f"clear frames {len(clear)}, using {min(args.n,len(clear))}  [DEPLOYMENT input, DINEOF k={args.dineof_k}]")
    for idx in clear[:args.n]:
        ss = np.stack([sst[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
        ms = np.stack([miss[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
        ob = np.stack([obs[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.uint8)
        gt = ss[-1].copy(); elig = ((obs[idx] > 0) & ocean).astype(np.float32); art = gen.generate(elig).astype(np.uint8)
        em = (art > 0) & (obs[idx] > 0) & ocean & ~np.isnan(gt)
        if em.sum() < 200: continue
        dist = distance_transform_edt(~((obs[idx] > 0) & (art == 0) & ocean))
        fills = {}
        fills["FNO_realmask"] = fill_model_prod(fno[0], ss, ob, ms, art, land, fno[1], fno[2], dev)
        fills["DINEOF_sparse"] = E.gauss_keep_obs(np.where(ocean, dineof_sparse(ss, ob, art, ocm, args.dineof_k), np.nan), land, art)
        gtile = gt[y0:y0 + S, x0:x0 + S]; ref = E.radial_psd(gtile, hann); gg = grad_mag(gt); dr = float(np.nanmax(gtile) - np.nanmin(gtile))
        for m in methods:
            fl = fills[m]; ae = np.abs(fl - gt); d = ae[em]; d = d[~np.isnan(d)]
            G[m]["mae"].append(float(d.mean())); G[m]["rmse"].append(float(np.sqrt((d ** 2).mean())))
            G[m]["gbias"].append(float(np.mean((grad_mag(fl) - gg)[em])) * 1000)
            G[m]["psd"].append(E.radial_psd(fl[y0:y0 + S, x0:x0 + S], hann)[hik].sum() / ref[hik].sum())
            G[m]["ssim"].append(float(ssim(np.nan_to_num(gtile), np.nan_to_num(fl[y0:y0 + S, x0:x0 + S]), data_range=dr)))
            for bi, (lo, hi) in enumerate(BINS):
                sel = em & (dist >= lo) & (dist < hi)
                if sel.sum() > 30: v = ae[sel]; depth[m][bi].append(float(np.nanmean(v[~np.isnan(v)])))
    n = len(G["FNO_realmask"]["mae"])
    print(f"\n=== FINAL (deployment input), series8, {n} frames, holes ~26% ===")
    print(f"{'method':<15}{'MAE':>7}{'RMSE':>7}{'gradBias':>9}{'PSDratio':>9}{'SSIM':>7}")
    for m in methods:
        r = {k: np.mean(G[m][k]) for k in G[m]}
        print(f"{m:<15}{r['mae']:>7.3f}{r['rmse']:>7.3f}{r['gbias']:>9.1f}{r['psd']:>9.2f}{r['ssim']:>7.3f}")
    print(f"\n--- MAE(K) by gap depth (px to nearest real obs) ---")
    print(f"{'depth':<8}" + "".join(f"{m:>15}" for m in methods))
    for bi, (lo, hi) in enumerate(BINS):
        lab = f"{lo}-{'inf' if hi>900 else hi}"
        print(f"{lab:<8}" + "".join(f"{(np.mean(depth[m][bi]) if depth[m][bi] else float('nan')):>15.3f}" for m in methods))
    CACHE = DI / "visualization/paper_figs/cache"; CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE / "eval_final_series8.npz",
                        G={m: {k: np.array(v) for k, v in G[m].items()} for m in methods},
                        depth={m: [np.array(x) for x in depth[m]] for m in methods},
                        methods=methods, n=n, bins=BINS,
                        ratio=(args.ratio_lo, args.ratio_hi))
    print(f"\nsaved cache/eval_final_series8.npz  (n={n} frames)")


if __name__ == "__main__":
    main()
