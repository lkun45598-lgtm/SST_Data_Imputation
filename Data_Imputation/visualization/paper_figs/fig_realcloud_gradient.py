#!/usr/bin/env python3
"""Real JAXA cloud-gap reconstruction — filled SST + gradient magnitude.

Single representative scene (a large real cloud gap, no ground truth).
Row 1: each method's filled SST field (shared SST scale), white contour = cloud edge.
Row 2: gradient magnitude |grad SST| of each filled field (shared scale).

Because natural cloud gaps have NO collocated reference, this figure is a
qualitative plausibility check (cf. Discussion): the gradient row exposes which
methods preserve fronts / mesoscale texture and which over-smooth the gap into a
featureless patch, without needing pixelwise error.

Representative scene: SERIES_ID=8, frame 179 (obs 17.7%, cloud px 108698) —
a large real cloud gap under very sparse observations.

Usage:
  python fig_realcloud_gradient.py --gpu 3 --frame 179
  python fig_realcloud_gradient.py --gpu 3 --frame 179 --no-gaussian
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

DI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DI))
sys.path.insert(0, str(DI / "baselines"))
sys.path.insert(0, str(DI / "visualization/paper_figs"))

from baselines.eval_all_methods import (
    load_fno, load_deep, run_torch_model, FNO_MEAN, FNO_STD, W, GAUSS_SIGMA,
    linear_interp_2d, cubic_interp_2d)
from baselines.optimal_interpolation import oi_reconstruct
from gen_eval_data import knn_inpaint_2d, gauss_filter
from style import apply_paper_style, CMAPS, setup_geo_ax, LAND_COLOR, MISSING_OVERLAY

SST_CMAP = CMAPS["sst"]
GRAD_CMAP = mpl.colormaps["magma"]
H12 = DI / "experiments/hourly_data/h12"
OUT = DI / "visualization/paper_figs/output"
OUT.mkdir(parents=True, exist_ok=True)
SERIES_ID = 8

METHODS = ["obs", "fno", "dincae", "unet", "convlstm", "oi", "knn", "linear", "cubic"]
TITLES = ["JAXA obs\n(cloud gaps)", "FNO-CBAM\n(Ours)", "DINCAE", "U-Net",
          "ConvLSTM", "OI / Kriging", "KNN / IDW", "Linear", "Cubic"]

def gradmag(field, ocean):
    """Gradient magnitude of a 2D field; NaN outside ocean."""
    f = np.where(ocean, field, np.nan)
    gy, gx = np.gradient(f)
    g = np.sqrt(gy ** 2 + gx ** 2)
    return np.where(ocean, g, np.nan)


def build_input(sst_all, miss_all, idx):
    sst_seq = np.stack([sst_all[max(0, idx - (W - 1) + t)] for t in range(W)])
    miss_seq = np.stack([miss_all[max(0, idx - (W - 1) + t)] for t in range(W)])
    return sst_seq, miss_seq


def reconstruct_all(idx, sst_all, obs_all, miss_all, land, ocean, models, device,
                    apply_gaussian=True):
    """Return dict method -> filled full-ocean SST field (K), NaN on land.

    Mirrors the DEPLOYED inference (inference/fill_jaxa.py):
      * history frames (t-29..t-1): the KNN-filled proxy field (sst_all), giving
        the model dense temporal context;
      * target frame t: the RAW JAXA observation — every non-observed ocean pixel
        (`~obs`, i.e. the whole real cloud gap) is the region to reconstruct, and
        the model only ever sees observed pixels there (the gap is set to the
        normalization mean inside run_torch_model before inference).
    So `cloud` here is the *entire real cloud gap*, not the sparse residual
    `original_missing_mask`, and baselines fill from genuine observations only.
    """
    sst_seq, miss_seq = build_input(sst_all, miss_all, idx)
    # real cloud gap on the target frame = all non-observed ocean pixels
    cloud = ((obs_all[idx] == 0) & ocean).astype(np.float32)
    # target-frame background = raw observation (gap = NaN); model input's last
    # frame gap is overwritten with the mean inside run_torch_model, so the KNN
    # proxy value never leaks into the reconstruction.
    obs_bg = np.where(cloud > 0, np.nan, sst_all[idx]).astype(np.float32)
    seq_in = sst_seq.copy()
    seq_in[-1] = obs_bg
    ms_in = miss_seq.copy().astype(np.float32)
    ms_in[-1] = cloud
    preds = {}
    land = (~ocean).astype(np.uint8)
    for nm in ("fno", "dincae", "unet", "convlstm"):
        out = run_torch_model(models[nm], seq_in, ms_in, cloud,
                              FNO_MEAN, FNO_STD, device)
        # NOTE: deployed inference does NOT low-pass the model output, so we
        # splice the raw prediction into the gap and keep raw observations
        # elsewhere (exactly fill_jaxa.py's output composition).
        full = np.where(cloud > 0, out, obs_bg)
        # Smoothing reproduces the earlier qualitative figure; --no-gaussian
        # exposes the deployed raw model output for a direct comparison.
        if apply_gaussian and nm == "fno":
            full = gauss_filter(full, land, sigma=1.0, fill_region=(cloud > 0))
        preds[nm] = np.where(ocean, full, np.nan)
    oi_out, _ = oi_reconstruct(sst_all[idx], cloud, obs_all[idx].astype(np.uint8))
    preds["oi"] = np.where(ocean, np.where(cloud > 0, oi_out, obs_bg), np.nan)
    ki = sst_all[idx].copy(); ki[cloud > 0] = np.nan
    knn_out = knn_inpaint_2d(ki, cloud, obs_all[idx].astype(np.uint8), k=20)
    preds["knn"] = np.where(ocean, np.where(cloud > 0, knn_out, obs_bg), np.nan)
    # classical griddata interpolants (same sparse-obs input as KNN): the field
    # has the cloud gap set to NaN; fill only within the gap, keep obs elsewhere.
    lin_out = linear_interp_2d(ki, cloud > 0, land, obs_all[idx].astype(np.uint8))
    preds["linear"] = np.where(ocean, np.where(cloud > 0, lin_out, obs_bg), np.nan)
    cub_out = cubic_interp_2d(ki, cloud > 0, land, obs_all[idx].astype(np.uint8))
    preds["cubic"] = np.where(ocean, np.where(cloud > 0, cub_out, obs_bg), np.nan)
    return preds, cloud, obs_bg
def k_to_c(a):
    return a - 273.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frame", type=int, default=179)
    ap.add_argument("--out-name", type=str, default="fig_realcloud_gradient")
    ap.add_argument("--no-gaussian", action="store_true",
                    help="show raw FNO output without Gaussian post-processing")
    args = ap.parse_args()
    if args.no_gaussian and args.out_name == "fig_realcloud_gradient":
        args.out_name = "fig_realcloud_gradient_nogauss"
    import torch
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    apply_paper_style()

    path = H12 / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:].astype(np.float32)
        obs_all = f["original_obs_mask"][:].astype(np.uint8)
        miss_all = f["original_missing_mask"][:].astype(np.uint8)
        land = f["land_mask"][:].astype(np.uint8)
        lat = f["latitude"][:]
        lon = f["longitude"][:]
    ocean = (land == 0)

    print("loading models...")
    models = {"fno": load_fno(device), "dincae": load_deep("dincae", device),
              "unet": load_deep("unet", device), "convlstm": load_deep("convlstm", device)}

    idx = args.frame
    preds, cloud, obs_bg = reconstruct_all(
        idx, sst_all, obs_all, miss_all, land, ocean, models, device,
        apply_gaussian=not args.no_gaussian,
    )
    obs_ratio = (obs_all[idx] * ocean).sum() / ocean.sum()
    print(f"frame {idx}: obs {obs_ratio*100:.1f}%, cloud px {int(cloud.sum())}")

    # flip to ascending lat for origin='lower' + geo extent
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        flip = lambda a: a[::-1, :]
        ocean = flip(ocean); cloud = flip(cloud)
        obs_bg = flip(obs_bg); obs_idx = flip(obs_all[idx])
        preds = {k: flip(v) for k, v in preds.items()}
    else:
        obs_idx = obs_all[idx]
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # observed-only field (row-1 first column) = raw JAXA observation, gap blank
    obs_field = np.where(ocean, k_to_c(obs_bg), np.nan)
    fields_c = {"obs": obs_field}
    for m in METHODS[1:]:
        fields_c[m] = np.where(ocean, k_to_c(preds[m]), np.nan)

    # shared SST scale from observed pixels
    vmin, vmax = np.percentile(obs_field[np.isfinite(obs_field)], [2, 98])
    # gradient fields (skip obs col — has holes)
    grads = {m: gradmag(fields_c[m], ocean) for m in METHODS[1:]}
    gall = np.concatenate([grads[m][np.isfinite(grads[m])] for m in METHODS[1:]])
    gmax = np.percentile(gall, 98)
    _render(args, extent, lon, lat, ocean, cloud, fields_c, grads,
            vmin, vmax, gmax, obs_ratio, idx)
def _draw(ax, img, extent, ocean, cmap, vmin, vmax, lon, lat,
          show_x, show_y, cloud=None):
    ax.imshow(np.where(~ocean, 1.0, np.nan), extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]), vmin=0, vmax=1,
              aspect="equal", interpolation="nearest", zorder=1)
    im = ax.imshow(img, extent=extent, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest", zorder=2)
    # NOTE: no cloud-boundary contour — the deployed inference visualization
    # (inference/fill_jaxa.py, Panel 3) shows the reconstructed ocean field
    # solid with no outline; the cloud gap is marked only on the obs column.
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    return im


def _render(args, extent, lon, lat, ocean, cloud, fields_c, grads,
            vmin, vmax, gmax, obs_ratio, idx):
    nc = len(METHODS)
    fig, axes = plt.subplots(2, nc, figsize=(2.35 * nc, 5.6),
                             layout="constrained")
    # row 1: filled SST
    im_sst = None
    for c, m in enumerate(METHODS):
        # obs column: draw raw obs, then grey out the cloud gap (fig1b style),
        # and skip the white cloud contour (the grey patch already marks it).
        im_sst = _draw(axes[0, c], fields_c[m], extent, ocean, SST_CMAP,
                       vmin, vmax, lon, lat, show_x=False, show_y=(c == 0),
                       cloud=(None if m == "obs" else cloud))
        if m == "obs":
            axes[0, c].imshow(np.where(cloud > 0, 1.0, np.nan), extent=extent,
                              origin="lower",
                              cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                              vmin=0, vmax=1, aspect="equal",
                              interpolation="nearest", zorder=2.5)
        axes[0, c].set_title(TITLES[c], fontsize=10.5, fontweight="bold")
    # row 2: gradient magnitude (obs col left blank -> gray)
    im_g = None
    for c, m in enumerate(METHODS):
        if m == "obs":
            axes[1, c].imshow(np.where(ocean, 0.92, np.nan), extent=extent,
                              origin="lower", cmap="Greys", vmin=0, vmax=1,
                              aspect="equal", zorder=1)
            axes[1, c].imshow(np.where(~ocean, 1.0, np.nan), extent=extent,
                              origin="lower",
                              cmap=mpl.colors.ListedColormap([LAND_COLOR]),
                              vmin=0, vmax=1, aspect="equal", zorder=2)
            axes[1, c].text(0.5, 0.5, "(no reference\nin gap)",
                            transform=axes[1, c].transAxes, ha="center",
                            va="center", fontsize=9, color="0.45", style="italic")
            setup_geo_ax(axes[1, c], lon, lat, draw_xlabel=True,
                         draw_ylabel=True, step=2)
            continue
        im_g = _draw(axes[1, c], grads[m], extent, ocean, GRAD_CMAP,
                     0, gmax, lon, lat, show_x=True, show_y=(c == 0),
                     cloud=cloud)

    cb1 = fig.colorbar(im_sst, ax=axes[0, :].tolist(), location="right",
                       shrink=0.82, pad=0.008, aspect=18)
    cb1.set_label("SST (°C)", fontsize=11)
    cb2 = fig.colorbar(im_g, ax=axes[1, :].tolist(), location="right",
                       shrink=0.82, pad=0.008, aspect=18)
    cb2.set_label(r"$|\nabla \mathrm{SST}|$ (K/pixel)", fontsize=11)

    axes[0, 0].text(-0.55, 0.5, "Filled SST", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=12,
                    fontweight="bold")
    axes[1, 0].text(-0.55, 0.5, "Gradient magnitude",
                    transform=axes[1, 0].transAxes, rotation=90, va="center",
                    ha="center", fontsize=12, fontweight="bold")
    fig.suptitle(f"Reconstruction of a real Himawari-8 cloud gap "
                 f"(no ground truth; observed {obs_ratio*100:.0f}%). "
                 f"Grey overlay = cloud gap on the observation column.",
                 fontsize=12.5, fontweight="bold")
    out = OUT / f"{args.out_name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"saved -> {out} (+.pdf)")



if __name__ == "__main__":
    main()
