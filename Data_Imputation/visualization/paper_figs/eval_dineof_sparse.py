#!/usr/bin/env python3
"""CORRECTED comparison: methods reconstruct from RAW SPARSE observations only.
Previous bug: sst_data is fully KNN/temporal-filled (0% NaN, only ~38% real obs);
feeding that to DINEOF as 'valid' leaked the answer. Here DINEOF's known set =
real obs (obs==1) ONLY; everything else (fill + artificial holes) is missing.
FNO uses its deployed pipeline (sparse -> KNN prefill -> FNO). Both start from
the SAME sparse obs. Pseudo-GT = hidden real-obs pixels (art & obs==1)."""
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

HR = DI / "experiments/hourly_data/h12"
OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)
TILE = (112, 96, 192)
BINS = [(0, 3), (3, 8), (8, 20), (20, 999)]


def grad_mag(f):
    gy, gx = np.gradient(np.nan_to_num(f)); return np.sqrt(gy ** 2 + gx ** 2)


def dineof_sparse(ss, obs_seq, art, ocean, k=4):
    """DINEOF known set = REAL observations only (obs==1). Last frame hides art."""
    T = ss.shape[0]
    valid = (obs_seq == 1) & (ocean[None] == 1)
    valid[-1] = valid[-1] & (art == 0)
    data = np.where(valid, ss, np.nan).astype(np.float32)
    filled = dineof_fill(data, valid.astype(np.uint8), k=k, max_iter=25, tol=5e-4, center=True)
    out = ss[-1].copy(); out[art > 0] = filled[-1][art > 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = (land == 0); T = sst.shape[0]; WIN = 30
    y0, x0, S = TILE; hann = np.outer(np.hanning(S), np.hanning(S)); hik = slice(S // 8, S // 2)
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    clear = [i for i in range(WIN, T) if cov[i] > args.cov]

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fno_rm = E.load_fno(E.NEW_MODEL, dev)
    methods = ["DINEOF_sparse", "FNO_realmask"]
    from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
    bank = build_cloud_bank([str(HR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]], max_donors=400, seed=7)
    gen = RealCloudMaskGenerator(bank, seed=42, ratio_range=(0.2, 0.35))

    G = {m: {k: [] for k in ["mae", "rmse", "gbias", "psd", "ssim"]} for m in methods}
    depth = {m: [[] for _ in BINS] for m in methods}
    ocm = ocean.astype(np.uint8)
    print(f"clear frames: {len(clear)}, using {min(args.n,len(clear))}  (DINEOF from SPARSE obs only)")
    for idx in clear[:args.n]:
        ss = np.zeros((WIN, *sst.shape[1:]), np.float32); ms = np.zeros_like(ss); ob = np.zeros_like(ss, np.uint8)
        for t in range(WIN):
            s = max(0, idx - (WIN - 1) + t); ss[t] = sst[s]; ms[t] = miss[s]; ob[t] = obs[s]
        gt = ss[-1].copy()
        elig = ((obs[idx] > 0) & (ocean == 1)).astype(np.float32)
        art = gen.generate(elig).astype(np.uint8)
        em = (art > 0) & (obs[idx] > 0) & (ocean == 1) & ~np.isnan(gt)
        if em.sum() < 200: continue
        src = (obs[idx] > 0) & (art == 0) & (ocean == 1)
        dist = distance_transform_edt(~src)
        fills = {}
        fills["DINEOF_sparse"] = E.gauss_keep_obs(np.where(ocean, dineof_sparse(ss, ob, art, ocm, k=args.k), np.nan), land, art)
        fills["FNO_realmask"] = E.fill_model(fno_rm[0], ss, ms, art, land, fno_rm[1], fno_rm[2], dev)
        gtile = gt[y0:y0 + S, x0:x0 + S]; ref_psd = E.radial_psd(gtile, hann)
        gt_gm = grad_mag(gt); dr = float(np.nanmax(gtile) - np.nanmin(gtile))
        for m in methods:
            fl = fills[m]; ae = np.abs(fl - gt); d = ae[em]; d = d[~np.isnan(d)]
            G[m]["mae"].append(float(d.mean())); G[m]["rmse"].append(float(np.sqrt((d ** 2).mean())))
            G[m]["gbias"].append(float(np.mean((grad_mag(fl) - gt_gm)[em])) * 1000)
            G[m]["psd"].append(E.radial_psd(fl[y0:y0 + S, x0:x0 + S], hann)[hik].sum() / ref_psd[hik].sum())
            G[m]["ssim"].append(float(ssim(np.nan_to_num(gtile), np.nan_to_num(fl[y0:y0 + S, x0:x0 + S]), data_range=dr)))
            for bi, (lo, hi) in enumerate(BINS):
                sel = em & (dist >= lo) & (dist < hi)
                if sel.sum() > 30:
                    v = ae[sel]; depth[m][bi].append(float(np.nanmean(v[~np.isnan(v)])))

    n = len(G["FNO_realmask"]["mae"])
    print(f"\n=== CORRECTED: reconstruct from SPARSE real obs, series8, {n} frames, holes ~26% ===")
    print(f"{'method':<16}{'MAE':>7}{'RMSE':>7}{'gradBias':>9}{'PSDratio':>9}{'SSIM':>7}")
    for m in methods:
        r = {k: np.mean(G[m][k]) for k in G[m]}
        print(f"{m:<16}{r['mae']:>7.3f}{r['rmse']:>7.3f}{r['gbias']:>9.1f}{r['psd']:>9.2f}{r['ssim']:>7.3f}")
    print(f"\n--- MAE(K) by gap depth (px to nearest REAL obs) [n frames per bin in ()] ---")
    print(f"{'depth(px)':<12}" + "".join(f"{m:>16}" for m in methods))
    for bi, (lo, hi) in enumerate(BINS):
        lab = f"{lo}-{'inf' if hi>900 else hi}"
        row = "".join(f"{(np.mean(depth[m][bi]) if depth[m][bi] else float('nan')):>12.3f}({len(depth[m][bi]):>2})" for m in methods)
        print(f"{lab:<12}{row}")
    np.savez_compressed(DI/"visualization/paper_figs/cache/dineof_sparse.npz",
                        G={m:{k:np.array(v) for k,v in G[m].items()} for m in methods},
                        depth={m:[np.array(x) for x in depth[m]] for m in methods}, bins=BINS)
    print("saved cache/dineof_sparse.npz")


if __name__ == "__main__":
    main()
