#!/usr/bin/env python3
"""Structure-aware controlled evaluation (WITH same-frame GT).
Punch real-cloud holes into clear frames, fill, and compare the FILLED field's
STRUCTURE to the true field -- not just pointwise error:
  - MAE / RMSE (K)                      pointwise (rewards oversmoothing)
  - grad-MAE (mK/px)                    |‖∇fill‖-‖∇truth‖| at held-out pixels
  - grad-bias (mK/px)                   mean(‖∇fill‖-‖∇truth‖); <0 = oversmoothed
  - PSD hi-k ratio (fill/truth, =1)     mesoscale structure preserved?
  - SSIM (tile, vs truth)               structural similarity
A method can win MAE yet lose every structure metric (that's the whole point).
"""
import sys, argparse
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
import eval_realgap_noref as E
import eval_controlled as C

HR = DI / "experiments/hourly_data/h12"
OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)
TILE = (112, 96, 192)


def grad_mag(f):
    gy, gx = np.gradient(np.nan_to_num(f)); return np.sqrt(gy ** 2 + gx ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = land == 0; T = sst.shape[0]; WIN = 30
    y0, x0, S = TILE; hann = np.outer(np.hanning(S), np.hanning(S)); hik = slice(S // 8, S // 2)
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    clear = [i for i in range(WIN, T) if cov[i] > args.cov]

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    models = {"FNO_square": E.load_fno(E.OLD_MODEL, dev),
              "FNO_realmask": E.load_fno(E.NEW_MODEL, dev)}
    methods = ["linear", "cubic", "knn", "FNO_square", "FNO_realmask"]
    from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
    bank = build_cloud_bank([str(HR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]], max_donors=400, seed=7)
    gen = RealCloudMaskGenerator(bank, seed=42, ratio_range=(0.2, 0.35))

    M = {m: {k: [] for k in ["mae", "rmse", "gmae", "gbias", "psd", "ssim"]} for m in methods}
    print(f"clear frames: {len(clear)}")
    for idx in clear[:args.n]:
        ss = np.zeros((WIN, *sst.shape[1:]), np.float32); ms = np.zeros_like(ss)
        for t in range(WIN):
            s = max(0, idx - (WIN - 1) + t); ss[t] = sst[s]; ms[t] = miss[s]
        gt = ss[-1].copy()
        elig = ((obs[idx] > 0) & (ocean == 1)).astype(np.float32)
        art = gen.generate(elig).astype(np.uint8)
        em = (art > 0) & (obs[idx] > 0) & (ocean == 1) & ~np.isnan(gt)
        if em.sum() < 200: continue
        fills = C.fill_all(ss, ms, gt, art, obs[idx], land, models, dev)
        gtile = gt[y0:y0 + S, x0:x0 + S]; ref_psd = E.radial_psd(gtile, hann)
        gt_gm = grad_mag(gt)
        dr = float(np.nanmax(gtile) - np.nanmin(gtile))
        for m in methods:
            fl = fills[m]; d = (fl - gt)[em]; d = d[~np.isnan(d)]
            M[m]["mae"].append(float(np.abs(d).mean()))
            M[m]["rmse"].append(float(np.sqrt((d ** 2).mean())))
            fgm = grad_mag(fl)
            M[m]["gmae"].append(float(np.mean(np.abs(fgm - gt_gm)[em])) * 1000)
            M[m]["gbias"].append(float(np.mean((fgm - gt_gm)[em])) * 1000)
            ftile = fl[y0:y0 + S, x0:x0 + S]
            M[m]["psd"].append(E.radial_psd(ftile, hann)[hik].sum() / ref_psd[hik].sum())
            M[m]["ssim"].append(float(ssim(np.nan_to_num(gtile), np.nan_to_num(ftile), data_range=dr)))

    n = len(M["linear"]["mae"])
    print(f"\n=== Structure-aware controlled eval, series8, {n} clear frames, real-cloud holes ~26% ===")
    print(f"{'method':<14}{'MAE':>7}{'RMSE':>7}{'gradMAE':>9}{'gradBias':>9}{'PSDratio':>9}{'SSIM':>7}")
    print(f"{'(units)':<14}{'K':>7}{'K':>7}{'mK/px':>9}{'mK/px':>9}{'/truth=1':>9}{'0-1':>7}")
    for m in sorted(methods, key=lambda x: -np.mean(M[x]["ssim"])):
        r = {k: np.mean(M[m][k]) for k in M[m]}
        print(f"{m:<14}{r['mae']:>7.3f}{r['rmse']:>7.3f}{r['gmae']:>9.1f}{r['gbias']:>9.1f}{r['psd']:>9.2f}{r['ssim']:>7.3f}")

    # bar figure: MAE (lower=better) vs SSIM (higher=better) vs gradBias (closer 0)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    order = ["linear", "cubic", "knn", "FNO_square", "FNO_realmask"]
    cols = ["C1", "C3", "C2", "C0", "C4"]
    for a, key, tit, note in [
        (ax[0], "mae", "MAE (K)  ↓ lower better", "rewards oversmoothing"),
        (ax[1], "ssim", "SSIM vs truth  ↑ higher better", "structure similarity"),
        (ax[2], "gbias", "grad bias (mK/px)  →0 better", "<0 = structure smoothed away")]:
        vals = [np.mean(M[m][key]) for m in order]
        a.bar(range(len(order)), vals, color=cols)
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=25, ha="right")
        a.set_title(f"{tit}\n({note})", fontsize=11); a.grid(axis="y", alpha=0.3)
        if key == "gbias": a.axhline(0, color="k", lw=1)
    fig.suptitle("Accuracy vs structure: linear wins MAE but smooths structure away; "
                 "FNO_realmask preserves gradients/SSIM", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "08_structure_bars.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT}/08_structure_bars.png")


if __name__ == "__main__":
    main()
