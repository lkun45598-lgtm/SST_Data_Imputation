#!/usr/bin/env python3
"""FNO vs DINEOF (traditional numerical gap-filling) — the publishable comparison.
Controlled clear frames + real-cloud holes + same-frame GT. Structure-aware
metrics + stratification by GAP DEPTH (distance to nearest observation), so we
see where the model actually beats DINEOF: deep inside large gaps / on structure.
"""
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
import eval_baselines as eb

HR = DI / "experiments/hourly_data/h12"
OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)
TILE = (112, 96, 192)
BINS = [(0, 3), (3, 8), (8, 20), (20, 999)]   # gap-depth buckets (px to nearest obs)


def grad_mag(f):
    gy, gx = np.gradient(np.nan_to_num(f)); return np.sqrt(gy ** 2 + gx ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dineof-k", type=int, default=4)
    args = ap.parse_args()
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = land == 0; T = sst.shape[0]; WIN = 30
    y0, x0, S = TILE; hann = np.outer(np.hanning(S), np.hanning(S)); hik = slice(S // 8, S // 2)
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    clear = [i for i in range(WIN, T) if cov[i] > args.cov]

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fno_sq = E.load_fno(E.OLD_MODEL, dev); fno_rm = E.load_fno(E.NEW_MODEL, dev)
    methods = ["DINEOF", "FNO_realmask"]
    from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
    bank = build_cloud_bank([str(HR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]], max_donors=400, seed=7)
    gen = RealCloudMaskGenerator(bank, seed=42, ratio_range=(0.2, 0.35))

    G = {m: {k: [] for k in ["mae", "rmse", "gbias", "psd", "ssim"]} for m in methods}
    depth = {m: [[] for _ in BINS] for m in methods}   # abs error per depth bin
    print(f"clear frames: {len(clear)}, using {min(args.n,len(clear))}")
    for j, idx in enumerate(clear[:args.n]):
        ss = np.zeros((WIN, *sst.shape[1:]), np.float32); ms = np.zeros_like(ss)
        for t in range(WIN):
            s = max(0, idx - (WIN - 1) + t); ss[t] = sst[s]; ms[t] = miss[s]
        gt = ss[-1].copy()
        elig = ((obs[idx] > 0) & (ocean == 1)).astype(np.float32)
        art = gen.generate(elig).astype(np.uint8)
        em = (art > 0) & (obs[idx] > 0) & (ocean == 1) & ~np.isnan(gt)
        if em.sum() < 200: continue
        # gap depth = distance from each held-out pixel to nearest source (obs & ~art)
        src = (obs[idx] > 0) & (art == 0) & (ocean == 1)
        dist = distance_transform_edt(~src)
        fills = {}
        fills["DINEOF"] = eb.baseline_dineof(ss, ms, art, ocean.astype(np.uint8), k=args.dineof_k)
        fills["DINEOF"] = E.gauss_keep_obs(np.where(ocean, fills["DINEOF"], np.nan), land, art)
        for nm, mdl in [("FNO_realmask", fno_rm)]:
            fills[nm] = E.fill_model(mdl[0], ss, ms, art, land, mdl[1], mdl[2], dev)
        gtile = gt[y0:y0 + S, x0:x0 + S]; ref_psd = E.radial_psd(gtile, hann)
        gt_gm = grad_mag(gt); dr = float(np.nanmax(gtile) - np.nanmin(gtile))
        for m in methods:
            fl = fills[m]; ae = np.abs(fl - gt)
            d = ae[em]; d = d[~np.isnan(d)]
            G[m]["mae"].append(float(d.mean())); G[m]["rmse"].append(float(np.sqrt((d ** 2).mean())))
            G[m]["gbias"].append(float(np.mean((grad_mag(fl) - gt_gm)[em])) * 1000)
            G[m]["psd"].append(E.radial_psd(fl[y0:y0 + S, x0:x0 + S], hann)[hik].sum() / ref_psd[hik].sum())
            G[m]["ssim"].append(float(ssim(np.nan_to_num(gtile), np.nan_to_num(fl[y0:y0 + S, x0:x0 + S]), data_range=dr)))
            for bi, (lo, hi) in enumerate(BINS):
                sel = em & (dist >= lo) & (dist < hi)
                if sel.sum() > 30:
                    v = ae[sel]; depth[m][bi].append(float(np.nanmean(v[~np.isnan(v)])))

    n = len(G["DINEOF"]["mae"])
    print(f"\n=== FNO vs DINEOF, series8, {n} clear frames, real-cloud holes ~26% ===")
    print(f"{'method':<14}{'MAE':>7}{'RMSE':>7}{'gradBias':>9}{'PSDratio':>9}{'SSIM':>7}")
    for m in methods:
        r = {k: np.mean(G[m][k]) for k in G[m]}
        print(f"{m:<14}{r['mae']:>7.3f}{r['rmse']:>7.3f}{r['gbias']:>9.1f}{r['psd']:>9.2f}{r['ssim']:>7.3f}")
    print(f"\n--- MAE(K) stratified by gap depth (distance to nearest obs) ---")
    print(f"{'depth(px)':<14}" + "".join(f"{m:>14}" for m in methods))
    for bi, (lo, hi) in enumerate(BINS):
        lab = f"{lo}-{'inf' if hi>900 else hi}"
        row = "".join(f"{(np.mean(depth[m][bi]) if depth[m][bi] else float('nan')):>14.3f}" for m in methods)
        print(f"{lab:<14}{row}")

    # figure: MAE vs gap depth
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(BINS)); labs = [f"{lo}-{'∞' if hi>900 else hi}" for lo, hi in BINS]
    col = {"DINEOF": "C5", "FNO_square": "C0", "FNO_realmask": "C4"}
    for m in methods:
        y = [np.mean(depth[m][bi]) if depth[m][bi] else np.nan for bi in range(len(BINS))]
        ax[0].plot(x, y, "o-", color=col[m], lw=2, label=m)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_xlabel("gap depth: px to nearest observation")
    ax[0].set_ylabel("MAE (K)"); ax[0].set_title("MAE vs gap depth\n(model should win deep in large gaps)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    for m in methods:
        ax[1].bar(0, 0)  # placeholder to keep colors; real bars below
    keys = ["mae", "ssim", "gbias"]; xg = np.arange(len(keys)); w = 0.25
    for i, m in enumerate(methods):
        vals = [np.mean(G[m]["mae"]), np.mean(G[m]["ssim"]), abs(np.mean(G[m]["gbias"]))/100.0]
        ax[1].bar(xg + (i-1)*w, vals, w, color=col[m], label=m)
    ax[1].set_xticks(xg); ax[1].set_xticklabels(["MAE(K)↓", "SSIM↑", "|gradBias|/100↓"])
    ax[1].set_title("overall: accuracy + structure"); ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle("FNO vs DINEOF (traditional numerical method) — controlled, structure-aware", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "09_fno_vs_dineof.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT}/09_fno_vs_dineof.png")


if __name__ == "__main__":
    main()
