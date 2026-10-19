#!/usr/bin/env python3
"""FNO vs DINEOF as a function of MISSING FRACTION (the paper's main experiment).
Large contiguous real-cloud holes on clear frames (deployment input, pseudo-GT).
Shows the model's advantage grows with gap size. No KNN (it's the prefill, and
oversmooths -> structurally unrealistic)."""
import sys, argparse
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
import eval_realgap_noref as E, eval_final as F
from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
HR = DI / "experiments/hourly_data/h12"; OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)
CACHE = DI / "visualization/paper_figs/cache"; CACHE.mkdir(exist_ok=True)
TILE = (112, 96, 192)


def gmag(f):
    gy, gx = np.gradient(np.nan_to_num(f)); return np.sqrt(gy ** 2 + gx ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]; miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = (land == 0); T = sst.shape[0]; WIN = 30; ocm = ocean.astype(np.uint8)
    y0, x0, S = TILE; hann = np.outer(np.hanning(S), np.hanning(S)); hik = slice(S // 8, S // 2)
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    clear = [i for i in range(WIN, T) if cov[i] > 0.85]
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fno = E.load_fno(E.NEW_MODEL, dev)
    bank = build_cloud_bank([str(HR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]], max_donors=400, seed=7)
    RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]
    res = {r: {m: {k: [] for k in ["mae", "rmse", "gbias", "psd", "ssim"]} for m in ["DINEOF", "FNO"]} for r in RATIOS}
    for r in RATIOS:
        gen = RealCloudMaskGenerator(bank, seed=7, ratio_range=(max(0.05, r - 0.05), min(0.85, r + 0.05)))
        for idx in clear[:args.n]:
            ss = np.stack([sst[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
            ms = np.stack([miss[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
            ob = np.stack([obs[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.uint8)
            gt = ss[-1].copy(); elig = ((obs[idx] > 0) & ocean).astype(np.float32); art = gen.generate(elig).astype(np.uint8)
            em = (art > 0) & (obs[idx] > 0) & ocean & ~np.isnan(gt)
            if em.sum() < 200: continue
            gtile = gt[y0:y0 + S, x0:x0 + S]; ref = E.radial_psd(gtile, hann); gg = gmag(gt); dr = float(np.nanmax(gtile) - np.nanmin(gtile))
            fills = {"FNO": F.fill_model_prod(fno[0], ss, ob, ms, art, land, fno[1], fno[2], dev),
                     "DINEOF": E.gauss_keep_obs(np.where(ocean, F.dineof_sparse(ss, ob, art, ocm, args.k), np.nan), land, art)}
            for m in ["DINEOF", "FNO"]:
                fl = fills[m]; d = (fl - gt)[em]; d = d[~np.isnan(d)]
                res[r][m]["mae"].append(float(np.abs(d).mean())); res[r][m]["rmse"].append(float(np.sqrt((d ** 2).mean())))
                res[r][m]["gbias"].append(float(np.mean((gmag(fl) - gg)[em])) * 1000)
                res[r][m]["psd"].append(E.radial_psd(fl[y0:y0 + S, x0:x0 + S], hann)[hik].sum() / ref[hik].sum())
                res[r][m]["ssim"].append(float(ssim(np.nan_to_num(gtile), np.nan_to_num(fl[y0:y0 + S, x0:x0 + S]), data_range=dr)))
        print(f"ratio {r:.0%}: DINEOF MAE={np.mean(res[r]['DINEOF']['mae']):.3f} SSIM={np.mean(res[r]['DINEOF']['ssim']):.3f} | "
              f"FNO MAE={np.mean(res[r]['FNO']['mae']):.3f} SSIM={np.mean(res[r]['FNO']['ssim']):.3f}", flush=True)
    np.savez_compressed(CACHE / "ratiosweep.npz", res={str(r): {m: {k: np.array(v) for k, v in res[r][m].items()} for m in res[r]} for r in RATIOS}, ratios=RATIOS)
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 5)); rs = [r * 100 for r in RATIOS]
    for a, key, tit, lo in [(ax[0], "mae", "MAE (K) ↓", True), (ax[1], "ssim", "SSIM ↑", False), (ax[2], "gbias", "grad bias (mK/px) →0", False)]:
        for m, c in [("DINEOF", "C5"), ("FNO", "C4")]:
            y = [np.mean(res[r][m][key]) for r in RATIOS]; a.plot(rs, y, "o-", color=c, lw=2.2, label=m)
        a.set_xlabel("missing fraction of observations (%)"); a.set_title(tit); a.grid(alpha=0.3); a.legend()
        if key == "gbias": a.axhline(0, color="k", lw=1)
    fig.suptitle("FNO vs DINEOF vs missing fraction — model advantage grows with gap size (deployment input, pseudo-GT)", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "10_ratiosweep_fno_vs_dineof.png", dpi=120, bbox_inches="tight")
    print(f"saved {OUT}/10_ratiosweep_fno_vs_dineof.png")


if __name__ == "__main__":
    main()
