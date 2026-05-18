"""Figure 4: Reconstruction results — 3 cases (low/mid/high mask) × 5 columns.

Columns:
  (1) Masked input (GT with masked pixels rendered as soft gray)
  (2) Ground truth (complete SST)
  (3) KNN baseline (2D IDW reconstruction)
  (4) FNO-CBAM reconstruction (with Gaussian σ=1.0)
  (5) |Error| map for FNO-CBAM (only at masked pixels)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from style import (apply_paper_style, CMAPS, save_fig,
                   setup_geo_ax, annotate_metric,
                   LAND_COLOR, MISSING_OVERLAY)

CACHE = Path(__file__).parent / "cache" / "eval_cache.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def k_to_c(arr):
    return arr - 273.15


def flip_lat_if_descending(case):
    lat = case["lat"]
    if lat[0] > lat[-1]:
        case["lat"] = lat[::-1]
        for k in ("gt", "fno", "knn", "mask", "eval_mask", "land"):
            case[k] = case[k][::-1, :]
    return case


def render_panel(ax, sst_c, land, ocean_mask, lat, lon, vmin, vmax,
                 cmap, extra_overlay_mask=None, extra_overlay_color=None,
                 show_y=False, show_x=False, panel_title=None):
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    # land
    land_layer = np.where(land == 1, 1.0, np.nan)
    ax.imshow(land_layer, extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]),
              aspect="equal", interpolation="nearest")
    if extra_overlay_mask is not None:
        ov_layer = np.where(extra_overlay_mask, 1.0, np.nan)
        ax.imshow(ov_layer, extent=extent, origin="lower",
                  cmap=mpl.colors.ListedColormap([extra_overlay_color]),
                  aspect="equal", interpolation="nearest")
    disp = np.where(ocean_mask == 1, sst_c, np.nan)
    if extra_overlay_mask is not None:
        disp = np.where(extra_overlay_mask, np.nan, disp)
    im = ax.imshow(disp, extent=extent, origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    if panel_title:
        ax.set_title(panel_title, fontsize=13, pad=6)
    return im


def render_error_panel(ax, err, ocean, land, mask, lat, lon, vmax_err,
                       show_y=False, show_x=False, panel_title=None):
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    # land
    land_layer = np.where(land == 1, 1.0, np.nan)
    ax.imshow(land_layer, extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]),
              aspect="equal", interpolation="nearest")
    # ocean (white background outside mask region)
    bg = np.where((ocean == 1) & (mask == 0), 1.0, np.nan)
    ax.imshow(bg, extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap(["#fafafa"]),
              aspect="equal", interpolation="nearest")
    # error
    disp = np.where(mask > 0, np.abs(err), np.nan)
    im = ax.imshow(disp, extent=extent, origin="lower",
                   cmap=CMAPS["error"], vmin=0, vmax=vmax_err,
                   aspect="equal", interpolation="nearest")
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    if panel_title:
        ax.set_title(panel_title, fontsize=13, pad=6)
    return im


def main():
    apply_paper_style()
    raw = np.load(CACHE, allow_pickle=True)
    cases_arr = raw["cases"].item()  # dict (object array)

    order = ["low", "mid", "high"]
    rows = []
    for k in order:
        c = dict(cases_arr[k])
        flip_lat_if_descending(c)
        rows.append((k, c))

    # ----- Determine SST color range from union of GT values across cases -----
    all_gt_c = []
    for _, c in rows:
        ocean = 1 - c["land"]
        all_gt_c.append(k_to_c(c["gt"])[ocean == 1])
    all_concat = np.concatenate(all_gt_c)
    vmin = float(np.percentile(all_concat, 1))
    vmax = float(np.percentile(all_concat, 99))
    # Error range: shared across cases
    err_vmax = 0.6

    # ===== Figure =====
    # 3 rows × 7 cols layout:
    #   col0..3 = 4 SST panels (Masked, GT, KNN, FNO)
    #   col4    = SST colorbar gutter
    #   col5    = Error map
    #   col6    = Error colorbar gutter
    fig = plt.figure(figsize=(22, 13.5))
    gs = fig.add_gridspec(
        3, 7,
        width_ratios=[1, 1, 1, 1, 0.12, 1, 0.12],
        height_ratios=[1, 1, 1],
        hspace=0.10, wspace=0.10,
        left=0.08, right=0.97, top=0.94, bottom=0.05,
    )

    last_sst_im = None
    last_err_im = None
    col_titles = [
        "Masked Input",
        "Ground Truth",
        "KNN baseline",
        "FNO-CBAM (ours)",
        "Absolute Error (FNO-CBAM)",
    ]

    for r, (level_name, c) in enumerate(rows):
        ocean = 1 - c["land"]
        gt_c = k_to_c(c["gt"])
        fno_c = k_to_c(c["fno"])
        knn_c = k_to_c(c["knn"])
        mask = c["mask"]
        is_top = (r == 0)
        is_bot = (r == len(rows) - 1)
        # ===== Col 1: Masked Input =====
        ax = fig.add_subplot(gs[r, 0])
        last_sst_im = render_panel(
            ax, gt_c, c["land"], ocean, c["lat"], c["lon"], vmin, vmax,
            cmap=CMAPS["sst"],
            extra_overlay_mask=(mask > 0) & (ocean == 1),
            extra_overlay_color=MISSING_OVERLAY,
            show_y=True, show_x=is_bot,
            panel_title=col_titles[0] if is_top else None,
        )
        # Left row label — use the y-axis annotation channel (text-only via supylabel-like)
        label_text = f"{level_name.upper()} mask  ({c['actual_ratio']*100:.0f}%)"
        # Place via figure-level transform so it doesn't fight with the y-tick labels of col 0
        ax_pos = ax.get_position()
        fig.text(
            ax_pos.x0 - 0.040,                # left of the panel + room for y-tick labels
            ax_pos.y0 + ax_pos.height / 2,
            label_text,
            ha="center", va="center", rotation=90,
            fontsize=15, fontweight="bold", color="#222",
        )

        # ===== Col 2: GT =====
        ax = fig.add_subplot(gs[r, 1])
        render_panel(ax, gt_c, c["land"], ocean, c["lat"], c["lon"], vmin, vmax,
                     cmap=CMAPS["sst"],
                     show_y=False, show_x=is_bot,
                     panel_title=col_titles[1] if is_top else None)

        # ===== Col 3: KNN baseline =====
        ax = fig.add_subplot(gs[r, 2])
        render_panel(ax, knn_c, c["land"], ocean, c["lat"], c["lon"], vmin, vmax,
                     cmap=CMAPS["sst"],
                     show_y=False, show_x=is_bot,
                     panel_title=col_titles[2] if is_top else None)
        annotate_metric(
            ax,
            [f"MAE: {c['knn_metrics']['mae']:.3f} K",
             f"RMSE: {c['knn_metrics']['rmse']:.3f} K"],
            loc="lower left", fontsize=12,
        )

        # ===== Col 4: FNO-CBAM =====
        ax = fig.add_subplot(gs[r, 3])
        render_panel(ax, fno_c, c["land"], ocean, c["lat"], c["lon"], vmin, vmax,
                     cmap=CMAPS["sst"],
                     show_y=False, show_x=is_bot,
                     panel_title=col_titles[3] if is_top else None)
        annotate_metric(
            ax,
            [f"MAE: {c['fno_metrics']['mae']:.3f} K",
             f"RMSE: {c['fno_metrics']['rmse']:.3f} K"],
            loc="lower left", fontsize=12,
        )

        # ===== Col 5: error map (gs col 5; col 4 is colorbar gutter) =====
        ax = fig.add_subplot(gs[r, 5])
        err = fno_c - gt_c
        last_err_im = render_error_panel(
            ax, err, ocean, c["land"], (mask * ocean).astype(np.uint8),
            c["lat"], c["lon"], err_vmax,
            show_y=False, show_x=is_bot,
            panel_title=col_titles[4] if is_top else None,
        )
        annotate_metric(
            ax,
            [f"Max: {c['fno_metrics']['maxv']:.3f} K"],
            loc="lower left", fontsize=12,
        )

    # SST colorbar — placed in gutter column 4 (between FNO panel and Error panel)
    cax_sst = fig.add_subplot(gs[:, 4])
    cax_sst.axis("off")
    pos = cax_sst.get_position()
    sst_axes = fig.add_axes([pos.x0 + pos.width * 0.05,
                             pos.y0 + pos.height * 0.06,
                             pos.width * 0.35, pos.height * 0.88])
    cb_sst = fig.colorbar(last_sst_im, cax=sst_axes, orientation="vertical")
    cb_sst.set_label("SST (°C)", fontsize=13, labelpad=8)
    cb_sst.ax.tick_params(labelsize=12)
    cb_sst.outline.set_linewidth(0.6)

    # Error colorbar — placed in gutter column 6 (after error panel)
    cax_err = fig.add_subplot(gs[:, 6])
    cax_err.axis("off")
    pos = cax_err.get_position()
    err_axes = fig.add_axes([pos.x0 + pos.width * 0.05,
                             pos.y0 + pos.height * 0.06,
                             pos.width * 0.35, pos.height * 0.88])
    cb_err = fig.colorbar(last_err_im, cax=err_axes, orientation="vertical")
    cb_err.set_label("|Error| (K)", fontsize=13, labelpad=8)
    cb_err.ax.tick_params(labelsize=12)
    cb_err.outline.set_linewidth(0.6)

    fig.suptitle(
        "Reconstruction comparison: KNN baseline vs FNO-CBAM "
        "(low / mid / high masking)",
        fontsize=15, fontweight="bold", y=0.985,
    )
    out = OUT_DIR / "fig4_reconstruction.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
