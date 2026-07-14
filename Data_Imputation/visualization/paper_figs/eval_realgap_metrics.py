#!/usr/bin/env python3
"""NO-REFERENCE metrics on REAL JAXA cloud gaps (no GT): FNO vs DINEOF.
  (1) PSD hi-k ratio vs clear-sky climatology  (=1 ideal; structure realism)
  (2) temporal ratio = mean|Δt at filled| / mean|Δt at observed|  (=1 ideal)
  (3) seam gradient (mK/px) at obs<->fill boundary, OCEAN-ONLY ring
Deployment pipeline input. Series 8 real cloudy frames."""
import sys, argparse
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
import eval_realgap_noref as E
from dineof import dineof_fill
HR = DI / "experiments/hourly_data/h12"; OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)
CACHE = DI / "visualization/paper_figs/cache"; TILE = (112, 96, 192)


def fno_realgap(fno, ss, ob, ms, land, dev):
    nm, ns = fno[1], fno[2]; cloud = (ob[-1] == 0) & (land == 0)
    si = ss.copy(); keep = (ob[-1] > 0) & (land == 0); si[-1] = np.where(keep, ss[-1], nm)
    m = ms.copy().astype(np.float32); m[-1] = (~keep & (land == 0)).astype(np.float32)
    sn = np.nan_to_num((si - nm) / ns, nan=0.0)
    st = torch.from_numpy(sn).unsqueeze(0).float().to(dev); mt = torch.from_numpy(m).unsqueeze(0).to(dev)
    with torch.no_grad(): pr = fno[0](st, mt).squeeze().cpu().numpy() * ns + nm
    out = np.where(cloud, pr, ss[-1]); return E.gauss_keep_obs(np.where(land == 0, out, np.nan), land, cloud.astype(np.uint8))


def dineof_realgap(ss, ob, ocm, land, k=2):
    cloud = (ob[-1] == 0) & (ocm == 1); valid = (ob == 1) & (ocm[None] == 1)
    data = np.where(valid, ss, np.nan).astype(np.float32)
    filled = dineof_fill(data, valid.astype(np.uint8), k=k, max_iter=25, tol=5e-4, center=True)
    out = ss[-1].copy(); out[cloud] = filled[-1][cloud]; return E.gauss_keep_obs(np.where(ocm == 1, out, np.nan), land, cloud.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--k", type=int, default=2); ap.add_argument("--lo", type=float, default=0.2); ap.add_argument("--hi", type=float, default=0.95)
    args = ap.parse_args()
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]; miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = land == 0; T = sst.shape[0]; WIN = 30; ocm = ocean.astype(np.uint8)
    y0, x0, S = TILE; hann = np.outer(np.hanning(S), np.hanning(S)); hik = slice(S // 8, S // 2)
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    tile_cov = obs[:, y0:y0 + S, x0:x0 + S].reshape(T, -1).mean(1)
    ref_psd = np.mean([E.radial_psd(sst[i, y0:y0 + S, x0:x0 + S], hann) for i in range(T) if tile_cov[i] > 0.95][:30], 0)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fno = E.load_fno(E.NEW_MODEL, dev)
    ocean_core = binary_erosion(ocean, iterations=2)   # ocean pixels away from land
    cloudy = [i for i in range(WIN, T) if args.lo < cov[i] < args.hi][:args.n]
    print(f"real cloudy frames: {len(cloudy)}  (deployment input, DINEOF k={args.k})")

    def win(idx):
        return (np.stack([sst[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32),
                np.stack([miss[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32),
                np.stack([obs[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.uint8))
    M = {m: {"psd": [], "tratio": [], "seam": [], "curve": []} for m in ["DINEOF", "FNO"]}
    for idx in cloudy:
        ss, ms, ob = win(idx); ssp, msp, obp = win(idx - 1)
        cloud = (obs[idx] == 0) & ocean
        fills = {"FNO": fno_realgap(fno, ss, ob, ms, land, dev), "DINEOF": dineof_realgap(ss, ob, ocm, land, args.k)}
        fills_p = {"FNO": fno_realgap(fno, ssp, obp, msp, land, dev), "DINEOF": dineof_realgap(ssp, obp, ocm, land, args.k)}
        ring = binary_dilation(cloud, iterations=1) & (~cloud) & ocean_core   # ocean-only seam
        both_fill = cloud & ((obs[idx - 1] == 0) & ocean) & ocean_core
        both_obs = (obs[idx] > 0) & (obs[idx - 1] > 0) & ocean_core
        obs_dt = np.nanmean(np.abs(sst[idx] - sst[idx - 1])[both_obs]) if both_obs.sum() > 30 else np.nan
        for m in ["DINEOF", "FNO"]:
            fl = fills[m]; c = E.radial_psd(fl[y0:y0 + S, x0:x0 + S], hann)
            M[m]["curve"].append(c); M[m]["psd"].append(c[hik].sum() / ref_psd[hik].sum())
            gy, gx = np.gradient(np.where(ocean, fl, np.nan)); g = np.sqrt(gy ** 2 + gx ** 2)
            M[m]["seam"].append(float(np.nanmean(g[ring])) * 1000)
            if both_fill.sum() > 30 and obs_dt == obs_dt:
                fdt = np.nanmean(np.abs(fl - fills_p[m])[both_fill]); M[m]["tratio"].append(float(fdt / obs_dt))
    print(f"\nobserved-pixel |Δt| temporal ref (context): frame-to-frame real SST change")
    print(f"{'method':<9}{'PSD hi-k (=1)':>14}{'temporal ratio (=1)':>21}{'seam grad(mK/px)':>18}")
    for m in ["DINEOF", "FNO"]:
        print(f"{m:<9}{np.mean(M[m]['psd']):>14.2f}{np.mean(M[m]['tratio']):>21.2f}{np.mean(M[m]['seam']):>18.1f}")
    np.savez_compressed(CACHE / "realgap_metrics.npz",
                        M={m: {k: (np.mean(v, 0) if k == "curve" else np.array(v)) for k, v in M[m].items()} for m in M},
                        ref_psd=ref_psd, tile=TILE, n=len(cloudy))
    # figures: PSD curves + metric bars
    k = np.arange(len(ref_psd))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].loglog(k[1:S // 2], ref_psd[1:S // 2], "k-", lw=2.5, label="clear-sky climatology")
    for m, c in [("DINEOF", "C5"), ("FNO", "C4")]:
        ax[0].loglog(k[1:S // 2], np.mean(M[m]["curve"], 0)[1:S // 2], color=c, lw=2, label=f"{m} real-gap fill")
    ax[0].set_xlabel("wavenumber k"); ax[0].set_ylabel("radial PSD"); ax[0].set_title("Structure spectrum on real gaps\n(closest to black = most realistic)"); ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
    keys = ["psd", "tratio", "seam"]; labs = ["|PSD hi-k −1|↓", "|temporal−1|↓", "seam mK/px↓"]; x = np.arange(3); w = 0.35
    for i, m in enumerate(["DINEOF", "FNO"]):
        vals = [abs(np.mean(M[m]["psd"]) - 1), abs(np.mean(M[m]["tratio"]) - 1), np.mean(M[m]["seam"]) / 100]
        ax[1].bar(x + (i - 0.5) * w, vals, w, color=("C5" if m == "DINEOF" else "C4"), label=m)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["|PSD−1|", "|temporal−1|", "seam/100"]); ax[1].set_title("No-reference quality (lower = better)"); ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Real JAXA cloud-gap fill — no-reference metrics, series 8 ({len(cloudy)} cloudy frames)", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "12_realgap_noref_fno_vs_dineof.png", dpi=120, bbox_inches="tight")
    print(f"saved {OUT}/12_realgap_noref_fno_vs_dineof.png")


if __name__ == "__main__":
    main()
