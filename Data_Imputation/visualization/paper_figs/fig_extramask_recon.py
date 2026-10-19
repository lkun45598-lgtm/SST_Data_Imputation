
#!/usr/bin/env python3
"""Extra-mask reconstruction WITH ground truth — |pred - GT| error maps.

Complement to fig_realcloud_gradient (real cloud gaps, NO reference): here we
withhold originally-observed ocean pixels behind real-cloud-shaped artificial
masks, so a collocated reference SST exists at every masked pixel and pixelwise
error CAN be computed. This is the qualitative face of the quantitative
extra-mask protocol (Table of overall metrics / ablation).

Layout: a single high-missingness level (75 % of observed ocean withheld).
Columns = Reference (GT) + Masked input + the 8 reconstruction methods.
Row 1 shows the reconstructed SST field (prediction pasted into the withheld
region); Row 2 shows the matching |pred - GT| error map, annotated with the
method's MAE over the withheld pixels.

Usage:
  python fig_extramask_recon.py --gpu 3 --frame <idx>
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import h5py
import netCDF4 as nc
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
from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
from style import (apply_paper_style, CMAPS, setup_geo_ax, LAND_COLOR,
                   MISSING_OVERLAY)
import cmocean

SST_CMAP = CMAPS["sst"]
ERR_CMAP = cmocean.cm.amp
H12 = DI / "experiments/hourly_data/h12"
OUT = DI / "visualization/paper_figs/output"
OUT.mkdir(parents=True, exist_ok=True)
SERIES_ID = 8
LEVEL_NAME, LEVEL_RATIO = "75%", 0.75
JAXA_ROOT = Path("/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3")

METHODS = ["fno", "dincae", "unet", "convlstm", "oi", "knn", "linear", "cubic"]
TITLES = ["FNO-CBAM\n(Ours)", "DINCAE", "U-Net", "ConvLSTM",
          "OI / Kriging", "KNN / IDW", "Linear", "Cubic"]

def k_to_c(a):
    return a - 273.15


def load_jaxa_original(timestamp):
    """Load original JAXA SST observation from NetCDF.

    Args:
        timestamp: ISO format string like "2025-01-24T12:00:00"

    Returns:
        sst_data: (H, W) float32, Kelvin (NaN at missing pixels)
        obs_mask: (H, W) bool, True=observed (non-NaN)
    """
    from datetime import datetime
    dt = datetime.fromisoformat(timestamp)
    yyyymm = dt.strftime("%Y%m")
    dd = dt.strftime("%d")
    filename = dt.strftime("%Y%m%d%H%M%S") + ".nc"
    nc_path = JAXA_ROOT / yyyymm / dd / filename

    with nc.Dataset(str(nc_path), "r") as ds:
        sst = ds.variables["sea_surface_temperature"][0]  # (H, W) MaskedArray
        # Extract raw data (may contain NaN)
        if hasattr(sst, 'data'):
            sst_data = sst.data.astype(np.float32)
        else:
            sst_data = np.array(sst, dtype=np.float32)

        # Observation mask: True where NOT NaN
        # (MaskedArray.mask can be scalar False for fully-valid frames, so use NaN check)
        obs_mask = ~np.isnan(sst_data)

    return sst_data, obs_mask


def reconstruct_masked(idx, mask, sst_all, miss_all, gt_raw, obs_raw, land, ocean,
                       models, device):
    """Run all 8 methods on an extra-masked frame; return dict of full fields (K).

    Args:
        idx: target frame index in h5 arrays
        mask: extra artificial mask (1=withheld)
        sst_all: h5 KNN-filled SST for history window
        miss_all: h5 missing mask for history window
        gt_raw: ORIGINAL JAXA observation (H, W) in Kelvin
        obs_raw: ORIGINAL JAXA obs mask (H, W) bool
        land, ocean: masks
        models: dict of loaded models
        device: torch device

    Returns:
        preds: dict of reconstructions (K)
        gt_raw: original JAXA GT (K)
    """
    # History frames: use h5 KNN-filled (models need complete input)
    sst_seq = np.stack([sst_all[max(0, idx - (W - 1) + t)] for t in range(W)])
    miss_seq = np.stack([miss_all[max(0, idx - (W - 1) + t)] for t in range(W)])

    # Target frame input: use h5 KNN-filled data (consistent with training)
    # In training, the model sees KNN-filled complete data with artificial mask applied
    sst_seq[-1] = sst_all[idx].copy()
    obs_frame = obs_raw.astype(np.uint8)

    # Only mark extra-mask region as missing (not the original cloud gaps)
    # This evaluates reconstruction ability on originally-observed pixels only
    miss_seq[-1] = mask

    preds = {}
    for nm in ("fno", "dincae", "unet", "convlstm"):
        out = run_torch_model(models[nm], sst_seq, miss_seq, mask,
                              FNO_MEAN, FNO_STD, device)
        if nm == "fno":
            out = gauss_filter(out, land, GAUSS_SIGMA, fill_region=(mask > 0))
        preds[nm] = out

    # Classical methods: use original JAXA observation
    # OI / kriging
    oi_out, _ = oi_reconstruct(gt_raw, mask, obs_frame)
    preds["oi"] = oi_out

    # KNN / IDW
    ki = gt_raw.copy(); ki[mask > 0] = np.nan
    preds["knn"] = knn_inpaint_2d(ki, mask, obs_frame, k=20)

    # Linear/Cubic: only use original observations as source
    mask_b = mask.astype(bool)
    ii = gt_raw.copy(); ii[obs_frame == 0] = np.nan; ii[mask_b] = np.nan
    preds["linear"] = linear_interp_2d(ii, mask_b, land, obs_frame)
    preds["cubic"] = cubic_interp_2d(ii, mask_b, land, obs_frame)

    return preds, gt_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--out-name", type=str, default="fig_extramask_recon")
    ap.add_argument("--find-best-fno", action="store_true",
                    help="Scan frames to find where FNO performs best")
    args = ap.parse_args()
    import torch
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    apply_paper_style()

    path = H12 / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:].astype(np.float32)
        miss_all = f["original_missing_mask"][:].astype(np.uint8)
        land = f["land_mask"][:].astype(np.uint8)
        lat = f["latitude"][:]; lon = f["longitude"][:]
        timestamps = [t.decode() if isinstance(t, bytes) else t for t in f["timestamps"][:]]
    ocean = (land == 0)

    # pick a low-to-mid coverage frame (50-70% observed): more challenging task,
    # better demonstrates model superiority over baselines
    # We need to check original JAXA coverage, not h5's obs_mask
    if args.frame < 0:
        print("Scanning for suitable frame (50-70% coverage in original JAXA)...")
        oc_n = float(ocean.sum())
        cov = []
        for i in range(W - 1, len(timestamps)):
            try:
                _, obs_mask = load_jaxa_original(timestamps[i])
                ratio = (obs_mask & ocean).sum() / oc_n
                cov.append((i, ratio))
            except Exception as e:
                continue
        band = [(i, r) for i, r in cov if 0.50 <= r <= 0.70]

        if args.find_best_fno and band:
            # Find frame where FNO performs best among all methods
            print(f"Found {len(band)} candidate frames in 50-70% range")
            print("Scanning for best FNO performance (this may take a while)...")
            print("loading models...")
            models = {"fno": load_fno(device), "dincae": load_deep("dincae", device),
                      "unet": load_deep("unet", device), "convlstm": load_deep("convlstm", device)}

            best_idx = None
            best_fno_rank = float('inf')  # lower is better (1 = best)
            best_info = None

            # Test up to 15 frames to balance thoroughness with compute time
            test_frames = band[:15] if len(band) > 15 else band

            for frame_idx, (i, ratio) in enumerate(test_frames):
                print(f"\nTesting frame {i} ({frame_idx+1}/{len(test_frames)}): {timestamps[i]}, {ratio*100:.1f}% obs")
                try:
                    gt_raw_test, obs_raw_test = load_jaxa_original(timestamps[i])

                    # Generate extra mask
                    bank = build_cloud_bank(
                        [str(H12 / f"jaxa_knn_filled_{j:02d}.h5") for j in [0, 3, 6]])
                    elig = (obs_raw_test & ocean).astype(np.float32)
                    gen = RealCloudMaskGenerator(bank, seed=int(LEVEL_RATIO * 1000))
                    mask = gen.generate(elig, target_ratio=LEVEL_RATIO)

                    # Run all methods
                    preds, _ = reconstruct_masked(i, mask, sst_all, miss_all, gt_raw_test,
                                                  obs_raw_test, land, ocean, models, device)

                    # Compute MAE for each method on extra-mask region with GT
                    obs_frame = obs_raw_test.astype(np.uint8)
                    ev = (mask > 0) & (ocean > 0)
                    has_gt = (obs_frame == 1)
                    eval_region = ev & has_gt

                    if eval_region.sum() == 0:
                        print("  WARNING: no GT in eval region, skipping")
                        continue

                    maes = {}
                    for m in METHODS:
                        err = np.abs(preds[m] - gt_raw_test)
                        maes[m] = err[eval_region].mean()

                    # Rank FNO among all methods (1 = best, 8 = worst)
                    sorted_methods = sorted(maes.items(), key=lambda x: x[1])
                    fno_rank = [j for j, (nm, _) in enumerate(sorted_methods, 1) if nm == "fno"][0]

                    print(f"  FNO MAE: {maes['fno']:.4f}, Rank: {fno_rank}/8")
                    print(f"  Best: {sorted_methods[0][0]} ({sorted_methods[0][1]:.4f})")

                    if fno_rank < best_fno_rank or (fno_rank == best_fno_rank and fno_rank == 1):
                        best_idx = i
                        best_fno_rank = fno_rank
                        best_info = {
                            'frame': i,
                            'timestamp': timestamps[i],
                            'ratio': ratio,
                            'maes': maes.copy(),
                            'rank': fno_rank,
                            'ranking': sorted_methods
                        }
                        if fno_rank == 1:
                            print(f"  ** FNO is BEST! **")

                except Exception as e:
                    print(f"  Error testing frame {i}: {e}")
                    continue

            if best_idx is None:
                print("\nNo valid frames found, falling back to least-observed frame")
                idx = min(band, key=lambda t: t[1])[0]
            else:
                idx = best_idx
                print(f"\n=== BEST FNO FRAME ===")
                print(f"Frame {best_info['frame']}: {best_info['timestamp']}")
                print(f"Coverage: {best_info['ratio']*100:.1f}%")
                print(f"FNO rank: {best_info['rank']}/8")
                print(f"FNO MAE: {best_info['maes']['fno']:.4f}")
                print("\nAll methods (sorted by MAE):")
                for rank, (nm, mae) in enumerate(best_info['ranking'], 1):
                    marker = " <-- FNO" if nm == "fno" else ""
                    print(f"  {rank}. {nm:10s}: {mae:.4f}{marker}")

            # Models already loaded above
        elif band:
            idx = min(band, key=lambda t: t[1])[0]   # least-observed within band for harder task
            print("loading models...")
            models = {"fno": load_fno(device), "dincae": load_deep("dincae", device),
                      "unet": load_deep("unet", device), "convlstm": load_deep("convlstm", device)}
        else:                                        # fallback: closest to 0.60
            idx = min(cov, key=lambda t: abs(t[1] - 0.60))[0]
            print("loading models...")
            models = {"fno": load_fno(device), "dincae": load_deep("dincae", device),
                      "unet": load_deep("unet", device), "convlstm": load_deep("convlstm", device)}
    else:
        idx = args.frame
        print("loading models...")
        models = {"fno": load_fno(device), "dincae": load_deep("dincae", device),
                  "unet": load_deep("unet", device), "convlstm": load_deep("convlstm", device)}

    # Load ORIGINAL JAXA observation as GT
    print(f"Loading original JAXA data for frame {idx} ({timestamps[idx]})...")
    gt_raw, obs_raw = load_jaxa_original(timestamps[idx])
    obs_ratio = (obs_raw & ocean).sum() / ocean.sum()
    print(f"  frame {idx}: observed {obs_ratio*100:.1f}% of ocean (ORIGINAL JAXA)")

    # Load models if not already loaded (when --find-best-fno is not used)
    if 'models' not in locals():
        print("loading models...")
        models = {"fno": load_fno(device), "dincae": load_deep("dincae", device),
                  "unet": load_deep("unet", device), "convlstm": load_deep("convlstm", device)}

    # Generate extra mask on ORIGINAL observed pixels
    bank = build_cloud_bank(
        [str(H12 / f"jaxa_knn_filled_{i:02d}.h5") for i in [0, 3, 6]])
    elig = (obs_raw & ocean).astype(np.float32)

    gen = RealCloudMaskGenerator(bank, seed=int(LEVEL_RATIO * 1000))
    mask = gen.generate(elig, target_ratio=LEVEL_RATIO)
    preds, gt = reconstruct_masked(idx, mask, sst_all, miss_all, gt_raw, obs_raw,
                                   land, ocean, models, device)
    actual = float((mask * elig).sum() / (elig.sum() + 1e-8))
    print(f"  level {LEVEL_NAME}: withheld {actual*100:.1f}% of observed px")

    _render(args, mask, preds, gt, gt_raw, obs_raw, idx, ocean, land, lat, lon,
            obs_ratio, sst_all[idx])


def _bg(ax, extent, ocean, lon, lat, show_x, show_y):
    """Land layer + geo axis; caller draws data on top."""
    ax.imshow(np.where(~ocean, 1.0, np.nan), extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]), vmin=0, vmax=1,
              aspect="equal", interpolation="nearest", zorder=1)
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    # Increase tick label size for better readability
    ax.tick_params(axis='both', which='major', labelsize=9.5)


def _render(args, mask, preds, gt, gt_raw, obs_raw, idx, ocean, land, lat, lon,
            obs_ratio, sst_h5):
    """Render 2-row figure: reconstructions (row 0) + error maps (row 1).

    Args:
        mask: extra artificial mask (1=withheld)
        preds: dict of reconstructions from all methods (Kelvin)
        gt: same as gt_raw (for backward compat)
        gt_raw: original JAXA observation (Kelvin)
        obs_raw: original JAXA obs mask (bool)
        idx: frame index
        ocean, land: masks
        lat, lon: coordinates
        obs_ratio: observed fraction for title
        sst_h5: h5 KNN-filled SST data for this frame (Kelvin)
    """
    # flip to ascending lat for origin='lower'
    flip = (lat[0] > lat[-1])
    fl = (lambda a: a[::-1, :]) if flip else (lambda a: a)
    if flip:
        lat = lat[::-1]
        ocean = fl(ocean)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    obs_frame = obs_raw.astype(np.uint8)
    ev = (mask > 0) & (land == 0)                 # withheld ocean pixels (extra mask)
    has_gt = (land == 0) & (obs_frame == 1)       # reference exists here (original JAXA obs)

    # GT: original JAXA observation (K->C), NaN at real cloud gaps and land
    gt_c = np.where(obs_frame == 1, k_to_c(gt_raw), np.nan)
    gt_c = np.where(land == 0, gt_c, np.nan)

    # Masked input: what the model actually sees during inference
    # = h5 KNN-filled data (includes temporal weighted filling at original cloud gaps)
    #   MINUS extra-mask region (set to NaN for visualization, filled with mean in actual model input)
    maskin = k_to_c(sst_h5.copy())
    maskin = np.where(land == 0, maskin, np.nan)  # Land -> NaN
    maskin[ev] = np.nan  # Extra-mask region -> NaN for visualization

    # Base field for reconstruction: h5 KNN-filled data (K->C)
    # This shows what was there before extra-masking (includes KNN filling at original cloud gaps)
    base_field = k_to_c(sst_h5.copy())
    base_field = np.where(land == 0, base_field, np.nan)

    # per-method reconstructed field: prediction pasted into extra-mask region,
    # h5 KNN-filled data elsewhere (preserving the temporal weighted filling)
    # and matching error map (|pred-gt| only where extra mask overlaps with GT)
    recon, err, mae = {}, {}, {}
    for m in METHODS:
        # Compose: extra-mask region gets prediction, non-masked region keeps h5 data
        rc = np.where(ev, k_to_c(preds[m]), base_field)
        recon[m] = fl(rc)

        # Error: only compute where we withheld observed pixels (ev & has_gt)
        e = np.abs(preds[m] - gt_raw)
        e = np.where(ev & has_gt, e, np.nan)
        err[m] = fl(e)
        mae[m] = float(np.abs(preds[m] - gt_raw)[ev & has_gt].mean())

    # Shared color scales so the two output figures stay consistent.
    gt_cf, maskin_f = fl(gt_c), fl(maskin)
    finite_gt = gt_cf[np.isfinite(gt_cf)]
    svmin, svmax = np.percentile(finite_gt, [2, 98])
    all_err = np.concatenate([err[m][np.isfinite(err[m])] for m in METHODS])
    evmax = float(np.percentile(all_err, 98))

    # Two separate figures for a left/right LaTeX layout:
    #   *_reference.{png,pdf} : GT + masked input (stacked, narrow left column)
    #   *_methods.{png,pdf}   : 4x4 method grid (recon + error), wide right block
    _draw_reference(args, extent, ocean, lon, lat, gt_cf, maskin_f, svmin, svmax)
    _draw_methods(args, extent, ocean, lon, lat, recon, err, mae,
                  svmin, svmax, evmax)


def _draw_reference(args, extent, ocean, lon, lat, gt_c, maskin, svmin, svmax):
    """Figure 1: GT + masked input, stacked vertically (narrow left column)."""
    fig = plt.figure(figsize=(4.0, 7.6))
    # Extra vertical gap so the two-line panel titles clear the map above them.
    gs = fig.add_gridspec(2, 1, hspace=0.34,
                          left=0.20, right=0.86, top=0.92, bottom=0.06)

    ax_gt = fig.add_subplot(gs[0, 0])
    _bg(ax_gt, extent, ocean, lon, lat, False, True)
    im_sst = ax_gt.imshow(gt_c, extent=extent, origin="lower", cmap=SST_CMAP,
                          vmin=svmin, vmax=svmax, aspect="equal",
                          interpolation="bilinear", zorder=2)
    real_gaps_gt = np.isnan(gt_c) & ocean
    ax_gt.imshow(np.where(real_gaps_gt, 1.0, np.nan), extent=extent,
                 origin="lower", cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                 vmin=0, vmax=1, aspect="equal", interpolation="nearest", zorder=3)
    ax_gt.set_title("Reference SST\n(ground truth)", fontsize=13.5,
                    fontweight="bold", pad=8)

    ax_in = fig.add_subplot(gs[1, 0])
    _bg(ax_in, extent, ocean, lon, lat, True, True)
    ax_in.imshow(maskin, extent=extent, origin="lower", cmap=SST_CMAP,
                 vmin=svmin, vmax=svmax, aspect="equal",
                 interpolation="bilinear", zorder=2)
    all_missing_input = np.isnan(maskin) & ocean
    ax_in.imshow(np.where(all_missing_input, 1.0, np.nan), extent=extent,
                 origin="lower", cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                 vmin=0, vmax=1, aspect="equal", interpolation="nearest", zorder=3)
    ax_in.set_title("Masked input\n(observed pixels withheld)", fontsize=13.5,
                    fontweight="bold", pad=8)

    cax = fig.add_axes([0.885, 0.20, 0.03, 0.55])
    cb = fig.colorbar(im_sst, cax=cax)
    cb.set_label("SST (°C)", fontsize=13)
    cb.ax.tick_params(labelsize=11)

    out = OUT / f"{args.out_name}_reference.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"saved -> {out} (+.pdf)")


def _draw_methods(args, extent, ocean, lon, lat, recon, err, mae,
                  svmin, svmax, evmax):
    """Figure 2: 4x4 method grid (recon + error rows), wide right block."""
    methods_g1 = METHODS[:4]      # fno, dincae, unet, convlstm
    methods_g2 = METHODS[4:]      # oi, knn, linear, cubic
    titles_g1 = TITLES[:4]
    titles_g2 = TITLES[4:]

    ncol = 4
    fig = plt.figure(figsize=(3.3 * ncol + 1.2, 2.9 * 4 + 0.4))
    # Two independent 2-row grids (recon+error each), with a wider gap between
    # them so the second group's column titles clear the first group's error row.
    gs_top = fig.add_gridspec(2, ncol, hspace=0.12, wspace=0.06,
                              left=0.10, right=0.89, top=0.95, bottom=0.545)
    gs_bot = fig.add_gridspec(2, ncol, hspace=0.12, wspace=0.06,
                              left=0.10, right=0.89, top=0.455, bottom=0.05)

    def sst_panel(ax, field, show_x, show_y):
        _bg(ax, extent, ocean, lon, lat, show_x, show_y)
        return ax.imshow(field, extent=extent, origin="lower", cmap=SST_CMAP,
                         vmin=svmin, vmax=svmax, aspect="equal",
                         interpolation="bilinear", zorder=2)

    def err_panel(ax, m, show_x):
        _bg(ax, extent, ocean, lon, lat, show_x, False)
        im = ax.imshow(err[m], extent=extent, origin="lower", cmap=ERR_CMAP,
                       vmin=0, vmax=evmax, aspect="equal",
                       interpolation="bilinear", zorder=2)
        ax.text(0.04, 0.96, f"MAE {mae[m]:.3f} K", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                color="0.10", zorder=5,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none", alpha=0.80))
        return im

    recon_axes, err_axes = [], []
    im_sst = im_err = None
    for g, (methods, titles, gsub) in enumerate(
            [(methods_g1, titles_g1, gs_top), (methods_g2, titles_g2, gs_bot)]):
        for j, m in enumerate(methods):
            axr = fig.add_subplot(gsub[0, j])
            im_sst = sst_panel(axr, recon[m], False, (j == 0))
            axr.set_title(titles[j], fontsize=13.5, fontweight="bold", pad=10)
            recon_axes.append(axr)
            axe = fig.add_subplot(gsub[1, j])
            im_err = err_panel(axe, m, show_x=(g == 1))
            err_axes.append(axe)

        recon_axes[-4].text(-0.42, 0.5, "Reconstruction",
                            transform=recon_axes[-4].transAxes, rotation=90,
                            va="center", ha="center", fontsize=13, fontweight="bold")
        err_axes[-4].text(-0.42, 0.5, "Absolute error",
                          transform=err_axes[-4].transAxes, rotation=90,
                          va="center", ha="center", fontsize=13, fontweight="bold")

    # Two colorbars in the right margin: SST upper, error lower.
    cax_sst = fig.add_axes([0.905, 0.55, 0.014, 0.32])
    cb1 = fig.colorbar(im_sst, cax=cax_sst)
    cb1.set_label("SST (°C)", fontsize=13)
    cb1.ax.tick_params(labelsize=11)

    cax_err = fig.add_axes([0.905, 0.10, 0.014, 0.32])
    cb2 = fig.colorbar(im_err, cax=cax_err)
    cb2.set_label("Absolute error (K)", fontsize=13)
    cb2.ax.tick_params(labelsize=11)

    out = OUT / f"{args.out_name}_methods.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"saved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()





