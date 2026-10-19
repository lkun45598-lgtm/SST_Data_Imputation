#!/usr/bin/env python3
"""Controlled reconstruction experiment (WITH pseudo-GT).
Take high-coverage (clear) frames, punch REAL-CLOUD-SHAPED holes into the
observed field, fill with each method, compare to the stored true values.
-> exact MAE/RMSE on the held-out (masked) pixels. Figure = one frame; table
   = aggregate over N frames. Obs pixels outside the hole are never touched."""
import sys, argparse
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
from simple_interp import linear_interp_2d, cubic_interp_2d
from real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
import eval_realgap_noref as E

HR = DI / "experiments/hourly_data/h12"
OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)


def fill_all(ss, ms, gt, art, obs_t, land, models, dev):
    """art = artificial holes to reconstruct. Source = observed & ~art."""
    ocean = (land == 0).astype(np.uint8)
    src = (obs_t > 0) & (art == 0) & (ocean == 1) & ~np.isnan(gt)
    out = {}
    for nm, fn in (("linear", linear_interp_2d), ("cubic", cubic_interp_2d)):
        f = fn(np.where(src, gt, np.nan), src.astype(np.float32), ocean)
        c = np.where(art > 0, f, gt)
        out[nm] = E.gauss_keep_obs(np.where(ocean == 1, c, np.nan), land, art)
    obs_field = np.where(src, gt, np.nan)
    out["knn"] = E.gauss_keep_obs(E.knn_fill(obs_field, art, ocean), land, art)
    for name, (mdl, a, b) in models.items():
        out[name] = E.fill_model(mdl, ss, ms, art, land, a, b, dev)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=8)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    with h5py.File(HR / f"jaxa_knn_filled_{args.series:02d}.h5") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = land == 0; T = sst.shape[0]; WIN = 30
    cov = obs.reshape(T, -1).sum(1) / ocean.sum()
    clear = [i for i in range(WIN, T) if cov[i] > args.cov]
    print(f"clear frames (cov>{args.cov:.0%}): {len(clear)}")

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    models = {"FNO_square": E.load_fno(E.OLD_MODEL, dev),
              "FNO_realmask": E.load_fno(E.NEW_MODEL, dev)}
    methods = ["linear", "cubic", "knn", "FNO_square", "FNO_realmask"]
    bank = build_cloud_bank([str(HR / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]],
                            max_donors=400, seed=7)
    gen = RealCloudMaskGenerator(bank, seed=42, ratio_range=(0.2, 0.35))

    res = {m: {"mae": [], "rmse": []} for m in methods}
    saved_fig = False
    for idx in clear[:args.n]:
        ss = np.zeros((WIN, *sst.shape[1:]), np.float32); ms = np.zeros_like(ss)
        for t in range(WIN):
            s = max(0, idx - (WIN - 1) + t); ss[t] = sst[s]; ms[t] = miss[s]
        gt = ss[-1].copy()
        elig = ((obs[idx] > 0) & (ocean == 1)).astype(np.float32)
        art = gen.generate(elig).astype(np.uint8)
        em = (art > 0) & (obs[idx] > 0) & (ocean == 1) & ~np.isnan(gt)
        if em.sum() < 200: continue
        fills = fill_all(ss, ms, gt, art, obs[idx], land, models, dev)
        for m in methods:
            d = (fills[m] - gt)[em]; d = d[~np.isnan(d)]
            res[m]["mae"].append(float(np.abs(d).mean()))
            res[m]["rmse"].append(float(np.sqrt((d ** 2).mean())))
        if not saved_fig and 0.25 < art.sum()/elig.sum() < 0.40:
            make_fig(idx, gt, art, em, fills, methods, ocean, res)
            saved_fig = True

    print(f"\n=== Controlled real-cloud-mask reconstruction, series{args.series}, "
          f"{len(res['linear']['mae'])} clear frames ===")
    print(f"{'method':<14}{'MAE(K)':>9}{'RMSE(K)':>9}")
    order = sorted(methods, key=lambda m: np.mean(res[m]["mae"]))
    for m in order:
        print(f"{m:<14}{np.mean(res[m]['mae']):>9.4f}{np.mean(res[m]['rmse']):>9.4f}")


def make_fig(idx, gt, art, em, fills, methods, ocean, res):
    vmin, vmax = np.percentile(gt[ocean], [2, 98])
    fig, axs = plt.subplots(2, 4, figsize=(19, 9))
    axs = axs.ravel()
    g = np.where(ocean, gt, np.nan)
    axs[0].imshow(g, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Ground truth (clear frame {idx})", fontsize=12)
    masked = np.where(ocean & (art == 0), gt, np.nan)
    axs[1].imshow(np.where(ocean, 0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=1)
    im = axs[1].imshow(masked, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Input: real-cloud holes punched\n(hole={art.sum()/ (ocean.sum()):.0%} of ocean)", fontsize=12)
    for ax, m in zip(axs[2:], methods):
        f = np.where(ocean, fills[m], np.nan)
        im = ax.imshow(f, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        mae = np.mean(res[m]["mae"])
        ax.set_title(f"{m}   MAE={mae:.3f}K", fontsize=12,
                     color=("crimson" if m == "FNO_realmask" else "black"))
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=axs.tolist(), shrink=0.7, label="SST (K)")
    fig.suptitle("Controlled reconstruction: fill real-cloud-shaped holes on a clear frame "
                 "(pseudo-GT available)  — MAE on held-out masked pixels", fontsize=13)
    plt.savefig(OUT / "07_controlled_recon.png", dpi=115, bbox_inches="tight")
    print(f"saved {OUT}/07_controlled_recon.png")


if __name__ == "__main__":
    main()
