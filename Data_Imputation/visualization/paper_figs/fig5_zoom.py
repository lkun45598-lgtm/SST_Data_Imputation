"""Figure 5 (zoom detail) — the region boxed in Figure 4.

Top row: KNN / FNO-CBAM / Ground truth zoom maps (shared SST colorbar).
Bottom: a full-width SST cross-section profile through the zoom center,
showing GT (points), KNN (dashed) and FNO-CBAM (solid).
The zoom region is selected identically to fig4_overview.py.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
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
    """Identical to fig4_overview.find_best_zoom (deterministic)."""
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
                     show_y=False, panel_title=None):
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
    setup_geo_ax(ax, lon, lat, draw_xlabel=True, draw_ylabel=show_y, step=1)
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
    gt_c = k_to_c(case["gt"])
    fno_c = k_to_c(case["fno"])
    knn_c = k_to_c(case["knn"])
    diff_fno_knn = fno_c - knn_c

    finite = gt_c[(ocean == 1) & (obs_mask == 1)]
    vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))

    iy, ix, half_y, half_x = find_best_zoom(diff_fno_knn, obs_mask, ocean, lat, lon)
    iy0, iy1 = max(0, iy - half_y), min(len(lat), iy + half_y + 1)
    ix0, ix1 = max(0, ix - half_x), min(len(lon), ix + half_x + 1)
    z_lat0, z_lat1 = float(lat[iy0]), float(lat[iy1 - 1])
    z_lon0, z_lon1 = float(lon[ix0]), float(lon[ix1 - 1])
    zc = PALETTE["model"]

    lat_z, lon_z = lat[iy0:iy1], lon[ix0:ix1]
    land_z = land[iy0:iy1, ix0:ix1]
    ocean_z = 1 - land_z
    gt_z = gt_c[iy0:iy1, ix0:ix1]
    fno_z = fno_c[iy0:iy1, ix0:ix1]
    knn_z = knn_c[iy0:iy1, ix0:ix1]
    obs_z = obs_mask[iy0:iy1, ix0:ix1]
    cloud_z = (obs_z == 0) & (ocean_z == 1)
    sparse_gt_z = np.where((obs_z == 1) & (ocean_z == 1), gt_z, np.nan)

    fig = plt.figure(figsize=(13, 9.5), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.06)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78])
    ax_knn = fig.add_subplot(gs[0, 0])
    ax_fno = fig.add_subplot(gs[0, 1])
    ax_gt = fig.add_subplot(gs[0, 2])
    ax_prof = fig.add_subplot(gs[1, :])

    render_sst_panel(ax_knn, knn_z, land_z, ocean_z, lat_z, lon_z, vmin, vmax,
                     show_y=True, panel_title="KNN")
    im = render_sst_panel(ax_fno, fno_z, land_z, ocean_z, lat_z, lon_z, vmin, vmax,
                          panel_title="FNO-CBAM (ours)")
    render_sst_panel(ax_gt, sparse_gt_z, land_z, ocean_z, lat_z, lon_z, vmin, vmax,
                     extra_mask=cloud_z, extra_color=MISSING_OVERLAY,
                     panel_title="Ground truth")
    for a in (ax_knn, ax_fno, ax_gt):
        for spine in a.spines.values():
            spine.set_color(zc)
            spine.set_linewidth(2.0)

    cb = fig.colorbar(im, ax=[ax_knn, ax_fno, ax_gt], location="right",
                      shrink=0.9, pad=0.012, aspect=22)
    cb.set_label("SST (°C)", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    # full-width profile along zoom-center latitude
    center_lat = (z_lat0 + z_lat1) / 2
    iy_c = int(np.argmin(np.abs(lat - center_lat)))
    gt_line = gt_c[iy_c, ix0:ix1]
    knn_line = knn_c[iy_c, ix0:ix1]
    fno_line = fno_c[iy_c, ix0:ix1]
    valid = (obs_mask[iy_c, ix0:ix1] == 1) & (ocean[iy_c, ix0:ix1] == 1)
    gt_plot = np.where(valid, gt_line, np.nan)

    ax_prof.plot(lon_z, knn_line, color="#8c8c8c", lw=1.8, linestyle="--",
                 label="KNN baseline")
    ax_prof.plot(lon_z, fno_line, color=PALETTE["model"], lw=2.4,
                 label="FNO-CBAM (ours)")
    ax_prof.plot(lon_z, gt_plot, color=PALETTE["gt"], lw=0, marker="o",
                 markersize=4.0, alpha=0.95, label="GT (original obs)")
    ax_prof.set_xlim(z_lon0, z_lon1)
    ax_prof.set_xlabel("Longitude (°E)", fontsize=12)
    ax_prof.set_ylabel("SST (°C)", fontsize=12)
    ax_prof.set_title(f"SST profile along {center_lat:.1f}°N", fontsize=13,
                      pad=6, color=zc)
    ax_prof.tick_params(labelsize=10)
    ax_prof.grid(alpha=0.30, linestyle="--", linewidth=0.5)
    ax_prof.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.95)
    ax_prof.spines["top"].set_visible(False)
    ax_prof.spines["right"].set_visible(False)

    z_eval = (obs_z == 1) & (ocean_z == 1)
    if z_eval.sum() > 0:
        zerr_fno = np.abs(fno_z[z_eval] - gt_z[z_eval]).mean()
        zerr_knn = np.abs(knn_z[z_eval] - gt_z[z_eval]).mean()
        annotate_metric(ax_prof, ["MAE in zoom:",
                                  f"  KNN = {zerr_knn:.3f} K",
                                  f"  FNO = {zerr_fno:.3f} K",
                                  f"  Δ  = {zerr_knn - zerr_fno:+.3f} K"],
                        loc="upper left", fontsize=11)

    fig.suptitle(
        f"Zoom detail of the boxed region in Fig. 4  "
        f"({z_lat0:.1f}–{z_lat1:.1f}°N, {z_lon0:.1f}–{z_lon1:.1f}°E)",
        fontsize=15, fontweight="bold")

    out = OUT_DIR / "fig5_zoom.png"
    fig.savefig(out, dpi=360)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
