#!/usr/bin/env python3
"""Paper figures for the MAIN contribution: filling REAL JAXA cloud gaps
(no ground truth) — FNO vs DINEOF, deployment pipeline input, series 8.

Produces two journal-quality figures (PNG + PDF):
  fig_realgap_fill     — qualitative fill gallery: real obs / DINEOF / FNO /
                         FNO gradient (structure), 3 representative cloudy frames.
  fig_realgap_metrics  — quantitative no-reference metrics (read from cache):
                         (a) radial PSD vs clear-sky climatology,
                         (b) normalized quality bars (|PSD-1|, |temporal-1|, seam).

Run: python3 gen_fig_realgap.py [--gpu 0]
"""
import sys, argparse
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation

DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
import eval_realgap_noref as E
from dineof import dineof_fill
from style import (apply_paper_style, save_fig, annotate_panel, CMAPS,
                   LAND_COLOR, MISSING_OVERLAY)

HR = DI / "experiments/hourly_data/h12"
OUT = DI / "visualization/paper_figs/output"; OUT.mkdir(exist_ok=True)
CACHE = DI / "visualization/paper_figs/cache"
TILE = (112, 96, 192)

# consistent method colors across both figures
C_FNO = "#1f6fb4"      # ours — blue
C_DINEOF = "#c1440e"   # baseline — rust
C_REF = "#111111"      # clear-sky climatology — near-black
CMAP_SST = CMAPS["sst"]
CMAP_GRAD = plt.get_cmap("magma")


# ----------------------------------------------------------------------------- fills
def fno_realgap(fno, ss, ob, ms, land, dev):
    """Deployment input: history KNN-filled, day30 = raw obs (clouds->mean).
    Fill ONLY the real cloud (obs==0 & ocean); real obs never overwritten."""
    nm, ns = fno[1], fno[2]; cloud = (ob[-1] == 0) & (land == 0)
    si = ss.copy(); keep = (ob[-1] > 0) & (land == 0); si[-1] = np.where(keep, ss[-1], nm)
    m = ms.copy().astype(np.float32); m[-1] = (~keep & (land == 0)).astype(np.float32)
    sn = np.nan_to_num((si - nm) / ns, nan=0.0)
    st = torch.from_numpy(sn).unsqueeze(0).float().to(dev)
    mt = torch.from_numpy(m).unsqueeze(0).to(dev)
    with torch.no_grad():
        pr = fno[0](st, mt).squeeze().cpu().numpy() * ns + nm
    out = np.where(cloud, pr, ss[-1])
    return E.gauss_keep_obs(np.where(land == 0, out, np.nan), land, cloud.astype(np.uint8))


def dineof_realgap(ss, ob, ocm, land, k=2):
    cloud = (ob[-1] == 0) & (ocm == 1); valid = (ob == 1) & (ocm[None] == 1)
    data = np.where(valid, ss, np.nan).astype(np.float32)
    filled = dineof_fill(data, valid.astype(np.uint8), k=k, max_iter=25, tol=5e-4, center=True)
    out = ss[-1].copy(); out[cloud] = filled[-1][cloud]
    return E.gauss_keep_obs(np.where(ocm == 1, out, np.nan), land, cloud.astype(np.uint8))


def grad_mag(field, ocean):
    gy, gx = np.gradient(np.nan_to_num(field))
    return np.where(ocean, np.sqrt(gy ** 2 + gx ** 2), np.nan)


# ----------------------------------------------------------------------------- gallery
def show_land(ax, land):
    ax.imshow(np.where(land == 1, 0.0, np.nan), origin="lower",
              cmap=ListedColormap([LAND_COLOR]), vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_edgecolor("#333333")


def make_gallery(sst, obs, miss, land, cov, dev, args):
    ocean = land == 0; ocm = ocean.astype(np.uint8); WIN = 30; T = sst.shape[0]
    fno = E.load_fno(E.NEW_MODEL, dev)
    # representative frames with LARGE real gaps (model's home turf): 40-70% missing
    cand = [i for i in range(WIN, T) if 0.30 < cov[i] < 0.60]
    if len(cand) < 3:
        cand = [i for i in range(WIN, T) if 0.20 < cov[i] < 0.70]
    picks = [cand[len(cand) // 6], cand[len(cand) // 2], cand[5 * len(cand) // 6]]

    fills = []
    for idx in picks:
        ss = np.stack([sst[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
        ms = np.stack([miss[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.float32)
        ob = np.stack([obs[max(0, idx - 29 + t)] for t in range(WIN)]).astype(np.uint8)
        dn = dineof_realgap(ss, ob, ocm, land, args.k)
        fn = fno_realgap(fno, ss, ob, ms, land, dev)
        fills.append((idx, ss[-1], ob[-1], dn, fn))

    # GLOBAL sst scale across frames (same region/season) -> single shared colorbar
    obs_vals = np.concatenate([f[1][(f[2] > 0) & ocean] for f in fills])
    vmin, vmax = np.percentile(obs_vals, [2, 98])
    gmax = np.nanpercentile(np.concatenate(
        [grad_mag(f[4], ocean)[ocean] for f in fills]), 98)

    apply_paper_style()
    nrow = len(picks)
    fig, axs = plt.subplots(nrow, 4, figsize=(11.0, 2.9 * nrow), constrained_layout=False)
    col_titles = ["Real JAXA observation", "DINEOF fill", "FNO fill (ours)",
                  "FNO fill — gradient magnitude"]
    letters = "abcdefghijklmnop"
    im_sst = im_grad = None
    for r, (idx, day, ob_t, dn, fn) in enumerate(fills):
        cloud = (ob_t == 0) & ocean
        obs_only = np.where((ob_t > 0) & ocean, day, np.nan)
        # (col 0) real obs with cloud overlay
        ax = axs[r, 0]; show_land(ax, land)
        im_sst = ax.imshow(obs_only, origin="lower", cmap=CMAP_SST, vmin=vmin, vmax=vmax)
        ax.imshow(np.where(cloud, 0.0, np.nan), origin="lower",
                  cmap=ListedColormap([MISSING_OVERLAY]), vmin=0, vmax=1)
        # (col 1) DINEOF, (col 2) FNO
        for c, fld in [(1, dn), (2, fn)]:
            ax = axs[r, c]; show_land(ax, land)
            ax.imshow(np.where(ocean, fld, np.nan), origin="lower",
                      cmap=CMAP_SST, vmin=vmin, vmax=vmax)
        # (col 3) FNO gradient (structure)
        ax = axs[r, 3]; show_land(ax, land)
        im_grad = ax.imshow(grad_mag(fn, ocean), origin="lower",
                            cmap=CMAP_GRAD, vmin=0, vmax=gmax)
        # labels
        for c in range(4):
            annotate_panel(axs[r, c], f"({letters[r * 4 + c]})", loc="upper left", fontsize=9)
            if r == 0:
                axs[r, c].set_title(col_titles[c], fontsize=9.5, pad=4)
        axs[r, 0].set_ylabel(f"frame {idx}\nmissing {1 - cov[idx]:.0%}",
                             fontsize=8.5, rotation=90, labelpad=6)

    # shared colorbars
    sst_cb = fig.colorbar(im_sst, ax=axs[:, 0:3], orientation="vertical",
                          fraction=0.020, pad=0.012, shrink=0.92)
    sst_cb.set_label("SST (K)", fontsize=9); sst_cb.ax.tick_params(labelsize=8)
    sst_cb.outline.set_linewidth(0.5)
    g_cb = fig.colorbar(im_grad, ax=axs[:, 3], orientation="vertical",
                        fraction=0.045, pad=0.012, shrink=0.92)
    g_cb.set_label("|∇SST| (K/px)", fontsize=9); g_cb.ax.tick_params(labelsize=8)
    g_cb.outline.set_linewidth(0.5)

    fig.suptitle("Reconstruction of real JAXA cloud gaps (no ground truth) — deployment pipeline",
                 fontsize=11, y=0.995)
    save_fig(fig, OUT / "fig_realgap_fill.png", also_pdf=True)
    plt.close(fig)
    print(f"saved {OUT}/fig_realgap_fill.png (+pdf)  frames={picks}")


# ----------------------------------------------------------------------------- metrics
def make_metrics():
    d = np.load(CACHE / "realgap_metrics.npz", allow_pickle=True)
    M = d["M"].item(); ref_psd = d["ref_psd"]; n = int(d["n"]); S = TILE[2]
    hik = slice(S // 8, S // 2); k = np.arange(len(ref_psd))
    stat = {m: dict(psd=float(np.mean(M[m]["psd"])),
                    tr=float(np.mean(M[m]["tratio"])),
                    seam=float(np.mean(M[m]["seam"]))) for m in ["DINEOF", "FNO"]}

    apply_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    # (a) PSD spectrum
    kk = k[1:S // 2]
    ax[0].loglog(kk, ref_psd[1:S // 2], color=C_REF, lw=2.2, label="clear-sky climatology (ref.)")
    ax[0].loglog(kk, M["DINEOF"]["curve"][1:S // 2], color=C_DINEOF, lw=1.8, label="DINEOF")
    ax[0].loglog(kk, M["FNO"]["curve"][1:S // 2], color=C_FNO, lw=1.8, label="FNO (ours)")
    ax[0].axvspan(S // 8, S // 2, color="0.85", alpha=0.35, lw=0, zorder=0)
    ax[0].set_xlabel("wavenumber $k$"); ax[0].set_ylabel("radial power spectral density")
    ax[0].set_title("Structure spectrum on real gaps", fontsize=10)
    ax[0].legend(frameon=False, fontsize=7.8, loc="lower left")
    ax[0].grid(True, which="both", alpha=0.25, lw=0.5)
    annotate_panel(ax[0], "(a)", loc="upper right", fontsize=10)

    # (b) normalized quality bars (lower = better; 0 = ideal)
    labs = ["|PSD$_{hi}$−1|", "|temporal−1|", "seam /100"]
    x = np.arange(3); w = 0.36
    vD = [abs(stat["DINEOF"]["psd"] - 1), abs(stat["DINEOF"]["tr"] - 1), stat["DINEOF"]["seam"] / 100]
    vF = [abs(stat["FNO"]["psd"] - 1), abs(stat["FNO"]["tr"] - 1), stat["FNO"]["seam"] / 100]
    bD = ax[1].bar(x - w / 2, vD, w, color=C_DINEOF, label="DINEOF", edgecolor="white", lw=0.5)
    bF = ax[1].bar(x + w / 2, vF, w, color=C_FNO, label="FNO (ours)", edgecolor="white", lw=0.5)
    for bars in (bD, bF):
        for b in bars:
            ax[1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                       f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, fontsize=8.5)
    ax[1].set_ylabel("deviation from ideal (lower = better)")
    ax[1].set_title("No-reference quality", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    ax[1].grid(axis="y", alpha=0.25, lw=0.5)
    ax[1].set_ylim(0, max(vD + vF) * 1.18)
    annotate_panel(ax[1], "(b)", loc="upper left", fontsize=10)

    fig.suptitle(f"Real JAXA cloud-gap fill — no-reference evaluation (series 8, {n} cloudy frames)",
                 fontsize=10.5, y=1.02)
    save_fig(fig, OUT / "fig_realgap_metrics.png", also_pdf=True)
    plt.close(fig)
    print(f"saved {OUT}/fig_realgap_metrics.png (+pdf)")
    print(f"  DINEOF: PSD={stat['DINEOF']['psd']:.2f} temporal={stat['DINEOF']['tr']:.2f} seam={stat['DINEOF']['seam']:.1f}")
    print(f"  FNO   : PSD={stat['FNO']['psd']:.2f} temporal={stat['FNO']['tr']:.2f} seam={stat['FNO']['seam']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--skip-gallery", action="store_true")
    args = ap.parse_args()
    make_metrics()
    if args.skip_gallery:
        return
    with h5py.File(HR / "jaxa_knn_filled_08.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    cov = obs.reshape(sst.shape[0], -1).sum(1) / (land == 0).sum()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    make_gallery(sst, obs, miss, land, cov, dev, args)


if __name__ == "__main__":
    main()
