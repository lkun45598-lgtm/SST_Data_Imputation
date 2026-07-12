"""Figure 4 (overview) — single clean row of 5 reconstruction panels.

Masked input / Ground truth / KNN / FNO-CBAM / |FNO-GT|.
A zoom box on the KNN and FNO panels marks the region detailed in Figure 5.
Shared SST colorbar for the four SST panels; separate error colorbar.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from style import (apply_paper_style, CMAPS, setup_geo_ax, annotate_metric,
                   LAND_COLOR, MISSING_OVERLAY, PALETTE)

CACHE = Path(__file__).parent / "cache" / "eval_cache.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZOOM_SIZE_DEG = 3.5


def k_to_c(a):
    return a - 273.15


def flip_lat_if_descending(case):
    lat = case["lat"]
    if lat[0] > lat[-1]:
        case["lat"] = lat[::-1]
        for k in ("gt", "fno", "knn", "mask", "eval_mask", "land", "obs_mask"):
            if k in case:
                case[k] = case[k][::-1, :]
    return case


def find_best_zoom(diff_field, obs_mask, ocean, lat, lon, zoom_deg=ZOOM_SIZE_DEG):
    """Zoom window centered on the largest smoothed |FNO-KNN| disagreement over
    observed ocean. Deterministic — matches fig5_zoom.py exactly."""
    score = np.abs(diff_field).astype(np.float32)
    score = np.where((ocean == 1) & (obs_mask == 1), score, 0)
    score = gaussian_filter(score, sigma=6)
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    half_y = int(round(zoom_deg / 2 / dlat))
    half_x = int(round(zoom_deg / 2 / dlon))
    H, W = score.shape
    score[:half_y, :] = 0
    score[H - half_y:, :] = 0
    score[:, :half_x] = 0
    score[:, W - half_x:] = 0
    iy, ix = np.unravel_index(np.argmax(score), score.shape)
    return int(iy), int(ix), half_y, half_x


def render_sst_panel(ax, sst_c, land, ocean, lat, lon, vmin, vmax,
                     extra_mask=None, extra_color=None,
                     show_y=False, show_x=True, panel_title=None):
    cmap = CMAPS["sst"]
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(np.where(land == 1, 1.0, np.nan), extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]),
              aspect="equal", interpolation="nearest")
    if extra_mask is not None:
        ax.imshow(np.where(extra_mask, 1.0, np.nan), extent=extent,
                  origin="lower", cmap=mpl.colors.ListedColormap([extra_color]),
                  aspect="equal", interpolation="nearest")
    disp = np.where(ocean == 1, sst_c, np.nan)
    if extra_mask is not None:
        disp = np.where(extra_mask, np.nan, disp)
    im = ax.imshow(disp, extent=extent, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=9)
    if panel_title:
        ax.set_title(panel_title, fontsize=13, pad=6)
    return im


def main():
    apply_paper_style()
    raw = np.load(CACHE, allow_pickle=True)
    case = dict(raw["cases"].item()["mid"])
    flip_lat_if_descending(case)

    lat, lon = case["lat"], case["lon"]
    land = case["land"]
    ocean = 1 - land
    obs_mask = case["obs_mask"]
    art_mask = case["mask"]
    gt_c = k_to_c(case["gt"])
    fno_c = k_to_c(case["fno"])
    knn_c = k_to_c(case["knn"])
    err_fno = fno_c - gt_c
    diff_fno_knn = fno_c - knn_c

    finite = gt_c[(ocean == 1) & (obs_mask == 1)]
    vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    cloud = (obs_mask == 0) & (ocean == 1)
    err_vals = np.abs(err_fno)[(obs_mask == 1) & (ocean == 1)]
    err_vmax = max(0.3, float(np.percentile(err_vals, 99)))

    iy, ix, half_y, half_x = find_best_zoom(diff_fno_knn, obs_mask, ocean, lat, lon)
    iy0, iy1 = max(0, iy - half_y), min(len(lat), iy + half_y + 1)
    ix0, ix1 = max(0, ix - half_x), min(len(lon), ix + half_x + 1)
    z_lat0, z_lat1 = float(lat[iy0]), float(lat[iy1 - 1])
    z_lon0, z_lon1 = float(lon[ix0]), float(lon[ix1 - 1])
    zc = PALETTE["model"]

    fig = plt.figure(figsize=(18, 4.7), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.04, wspace=0.02)
    ax = fig.subplots(1, 5)

    im_sst = render_sst_panel(
        ax[0], gt_c, land, ocean, lat, lon, vmin, vmax,
        extra_mask=(art_mask > 0) & (ocean == 1), extra_color=MISSING_OVERLAY,
        show_y=True, panel_title="Masked input")

    sparse_gt = np.where((obs_mask == 1) & (ocean == 1), gt_c, np.nan)
    render_sst_panel(ax[1], sparse_gt, land, ocean, lat, lon, vmin, vmax,
                     extra_mask=cloud, extra_color=MISSING_OVERLAY,
                     panel_title="Ground truth")
    obs_pct = float(((obs_mask == 1) & (ocean == 1)).sum() / max(ocean.sum(), 1) * 100)
    annotate_metric(ax[1], [f"obs: {obs_pct:.1f}%"], loc="lower left", fontsize=10)

    render_sst_panel(ax[2], knn_c, land, ocean, lat, lon, vmin, vmax,
                     panel_title="KNN reconstruction")
    annotate_metric(ax[2], [f"MAE: {case['knn_metrics']['mae']:.3f} K"],
                    loc="lower left", fontsize=10)

    render_sst_panel(ax[3], fno_c, land, ocean, lat, lon, vmin, vmax,
                     panel_title="FNO-CBAM (ours)")
    annotate_metric(ax[3], [f"MAE: {case['fno_metrics']['mae']:.3f} K"],
                    loc="lower left", fontsize=10)

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax[4].imshow(np.where(land == 1, 1.0, np.nan), extent=extent, origin="lower",
                 cmap=mpl.colors.ListedColormap([LAND_COLOR]), aspect="equal",
                 interpolation="nearest")
    ax[4].imshow(np.where(cloud, 1.0, np.nan), extent=extent, origin="lower",
                 cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                 aspect="equal", interpolation="nearest")
    err_disp = np.where((obs_mask == 1) & (ocean == 1), np.abs(err_fno), np.nan)
    im_err = ax[4].imshow(err_disp, extent=extent, origin="lower",
                          cmap=cmocean.cm.amp, vmin=0, vmax=err_vmax,
                          aspect="equal", interpolation="nearest")
    setup_geo_ax(ax[4], lon, lat, draw_xlabel=True, draw_ylabel=False, step=2)
    ax[4].set_yticklabels([])
    ax[4].tick_params(labelsize=9)
    ax[4].set_title("|FNO − GT|", fontsize=13, pad=6)

    for a in (ax[2], ax[3]):
        a.add_patch(Rectangle((z_lon0, z_lat0), z_lon1 - z_lon0, z_lat1 - z_lat0,
                              fill=False, edgecolor=zc, linewidth=2.4))
    ax[3].text(z_lon0 + 0.1, z_lat1 - 0.35, "Fig. 5", fontsize=11, color=zc,
               fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                         edgecolor=zc, linewidth=1.0))

    cb_sst = fig.colorbar(im_sst, ax=ax[:4].tolist(), location="right",
                          shrink=0.9, pad=0.012, aspect=24)
    cb_sst.set_label("SST (°C)", fontsize=12)
    cb_sst.ax.tick_params(labelsize=10)
    cb_err = fig.colorbar(im_err, ax=ax[4], location="right",
                          shrink=0.9, pad=0.02, aspect=24)
    cb_err.set_label("|FNO − GT|  (K)", fontsize=11)
    cb_err.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Reconstruction overview — MID mask ({case['actual_ratio']*100:.0f}%), "
        f"{case['ts'][:10]}", fontsize=15, fontweight="bold")

    out = OUT_DIR / "fig4_overview.png"
    fig.savefig(out, dpi=360)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
