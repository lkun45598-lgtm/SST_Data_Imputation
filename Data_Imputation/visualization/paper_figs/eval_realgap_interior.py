#!/usr/bin/env python3
"""COMPLETE no-reference structure eval on REAL JAXA cloud gaps, series 8.
For every cloudy frame (deployment pipeline input) compute FNO and DINEOF fills
ONCE, then measure high-k power TWO ways:
  * full-field tile (192, includes obs<->fill seam)  -> vs clear-sky climatology
  * gap-INTERIOR window (48, fully inside the real cloud, seam-free)
      -> vs a CLEAN real-observed spectrum (same window size, clear frames)
Also stores per-frame arrays so the figure can show bootstrap CIs.
Saves cache/realgap_interior.npz.  Run: python3 eval_realgap_interior.py --gpu 0
"""
import sys, argparse, time
from pathlib import Path
import numpy as np, h5py, torch
from scipy.ndimage import uniform_filter
DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
import eval_realgap_noref as E
from gen_fig_realgap import fno_realgap, dineof_realgap
HR = DI / "experiments/hourly_data/h12"; CACHE = DI / "visualization/paper_figs/cache"
TILE = (112, 96, 192)


def best_window(R, S, H, W):
    frac = uniform_filter(R.astype(float), size=S, mode="constant", cval=0)
    valid = frac > 0.999
    valid[:S // 2, :] = False; valid[-(S // 2):, :] = False
    valid[:, :S // 2] = False; valid[:, -(S // 2):] = False
    if not valid.any():
        return None
    ys, xs = np.where(valid); c = len(ys) // 2
    return ys[c] - S // 2, xs[c] - S // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lo", type=float, default=0.2); ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--win", type=int, default=48); ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = land == 0; ocm = ocean.astype(np.uint8); T, H, W = sst.shape; WIN = 30
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    y0, x0, S = TILE; hannF = np.outer(np.hanning(S), np.hanning(S)); hikF = slice(S // 8, S // 2)
    Sw = args.win; hannW = np.outer(np.hanning(Sw), np.hanning(Sw)); hikW = slice(Sw // 8, Sw // 2)

    # full-field climatology reference (192 clear tiles)
    tile_cov = obs[:, y0:y0 + S, x0:x0 + S].reshape(T, -1).mean(1)
    clearF = [i for i in range(T) if tile_cov[i] > 0.95][:30]
    refF = np.mean([E.radial_psd(sst[i, y0:y0 + S, x0:x0 + S], hannF) for i in clearF], 0)
    # clean real-observed window reference (Sw, from fully-observed windows of clear frames)
    refW_list = []
    for i in [j for j in range(T) if cov[j] > 0.9]:
        w = best_window((obs[i] > 0) & ocean, Sw, H, W)
        if w:
            refW_list.append(E.radial_psd(sst[i, w[0]:w[0] + Sw, w[1]:w[1] + Sw], hannW))
    refW = np.mean(refW_list, 0)
    print(f"refs: climatology(192)={len(clearF)} frames, clean-obs-window({Sw})={len(refW_list)} windows")

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fno = E.load_fno(E.NEW_MODEL, dev)
    cloudy = [i for i in range(WIN, T) if args.lo < cov[i] < args.hi]
    if args.n:
        cloudy = cloudy[:args.n]
    print(f"cloudy frames: {len(cloudy)}  (deployment input, DINEOF k={args.k}, win={Sw})")

    R = {"fullF": [], "fullD": [], "intF": [], "intD": [], "int_miss": [],
         "curveF_int": [], "curveD_int": []}
    t0 = time.time()
    for n, idx in enumerate(cloudy):
        ss = np.stack([sst[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
        ms = np.stack([miss[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
        ob = np.stack([obs[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.uint8)
        fn = fno_realgap(fno, ss, ob, ms, land, dev)
        dn = dineof_realgap(ss, ob, ocm, land, args.k)
        fnO = np.where(ocean, fn, np.nan); dnO = np.where(ocean, dn, np.nan)
        # full-field (seam-inclusive) vs climatology
        R["fullF"].append(E.radial_psd(fnO[y0:y0 + S, x0:x0 + S], hannF)[hikF].sum() / refF[hikF].sum())
        R["fullD"].append(E.radial_psd(dnO[y0:y0 + S, x0:x0 + S], hannF)[hikF].sum() / refF[hikF].sum())
        # gap-interior (seam-free) vs clean real-observed spectrum
        w = best_window((obs[idx] == 0) & ocean, Sw, H, W)
        if w:
            wy, wx = w
            cF = E.radial_psd(fnO[wy:wy + Sw, wx:wx + Sw], hannW)
            cD = E.radial_psd(dnO[wy:wy + Sw, wx:wx + Sw], hannW)
            R["intF"].append(cF[hikW].sum() / refW[hikW].sum())
            R["intD"].append(cD[hikW].sum() / refW[hikW].sum())
            R["int_miss"].append(float(1 - cov[idx]))
            R["curveF_int"].append(cF); R["curveD_int"].append(cD)
        if (n + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(cloudy)}  ({el/ (n+1):.1f}s/frame, ETA {el/(n+1)*(len(cloudy)-n-1)/60:.0f}min)", flush=True)

    A = {k: np.array(v) for k, v in R.items() if not k.startswith("curve")}
    curveF = np.mean(R["curveF_int"], 0); curveD = np.mean(R["curveD_int"], 0)

    def ci(a):
        a = np.asarray(a); return float(a.mean()), float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))

    print(f"\n=== series8 real-gap structure (n_full={len(A['fullF'])}, n_interior={len(A['intF'])}) ===")
    print(f"{'metric':<26}{'FNO':>22}{'DINEOF':>22}")
    print(f"{'full-field hi-k (clim=1)':<26}{ci(A['fullF'])[0]:>10.2f} (med {ci(A['fullF'])[1]:.2f}){ci(A['fullD'])[0]:>10.2f} (med {ci(A['fullD'])[1]:.2f})")
    print(f"{'interior hi-k (realobs=1)':<26}{ci(A['intF'])[0]:>10.2f} (med {ci(A['intF'])[1]:.2f}){ci(A['intD'])[0]:>10.2f} (med {ci(A['intD'])[1]:.2f})")

    np.savez_compressed(CACHE / "realgap_interior.npz",
                        fullF=A["fullF"], fullD=A["fullD"], intF=A["intF"], intD=A["intD"],
                        int_miss=A["int_miss"], curveF_int=curveF, curveD_int=curveD,
                        refW=refW, refF=refF, win=Sw, tile=TILE,
                        n_full=len(A["fullF"]), n_int=len(A["intF"]))
    print(f"\nsaved cache/realgap_interior.npz")


if __name__ == "__main__":
    main()
