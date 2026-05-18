"""Figure 2: Three-stage preprocessing pipeline visualization.

2 rows x 4 cols:
  Row 1 (SST): Raw JAXA hourly | Temporal weighted | Low-pass filtered | 3D KNN filled
  Row 2 (Mask): observed / filled-by-temporal / filled-by-KNN / land / missing

Frame: 2018-03-09 (series_00 frame 241), a high-cloud-cover day.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from style import (apply_paper_style, CMAPS, save_fig,
                   annotate_panel, annotate_metric, setup_geo_ax,
                   LAND_COLOR, MASKED_OVERLAY, MISSING_OVERLAY,
                   CAT_OBSERVED, CAT_TEMPORAL, CAT_KNN, CAT_MISSING, CAT_LAND)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline data paths
DATA_ROOT = Path("/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM")
RAW_NC = Path("/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/201803/09/20180309230000.nc")

SERIES_ID = 0
FRAME_IDX = 241  # 2018-03-09


def _flip_lat(arr, lat):
    """If latitude is descending (h5 default), flip the lat-axis so that
    row 0 = southernmost. With origin='lower' this gives correct geo orientation."""
    if lat[0] > lat[-1]:
        return arr[..., ::-1, :]
    return arr


def load_pipeline_frame():
    weighted_h5 = DATA_ROOT / "jaxa_weighted_aligned" / f"jaxa_weighted_series_{SERIES_ID:02d}.h5"
    filtered_h5 = DATA_ROOT / "jaxa_filtered" / f"jaxa_filtered_{SERIES_ID:02d}.h5"
    knn_h5 = DATA_ROOT / "jaxa_knn_filled" / f"jaxa_knn_filled_{SERIES_ID:02d}.h5"

    with h5py.File(weighted_h5, "r") as f:
        sst_w = f["sst_data"][FRAME_IDX]
        miss_w = f["missing_mask"][FRAME_IDX]
        fill_w = f["fill_mask"][FRAME_IDX]
        ts = f["timestamps"][FRAME_IDX]
        ts = ts.decode() if isinstance(ts, bytes) else ts
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        sst_full = f["sst_data"][:]
        land = np.all(np.isnan(sst_full), axis=0)

    with h5py.File(filtered_h5, "r") as f:
        sst_f = f["sst_data"][FRAME_IDX]

    with h5py.File(knn_h5, "r") as f:
        sst_k = f["sst_data"][FRAME_IDX]
        knn_filled = (
            (f["original_missing_mask"][FRAME_IDX] == 1)
            & (f["temporal_fill_mask"][FRAME_IDX] == 0)
        )

    with nc.Dataset(RAW_NC) as ds:
        raw = ds.variables["sea_surface_temperature"][:]
        raw = raw[0]
        if hasattr(raw, "mask"):
            raw = np.where(raw.mask, np.nan, raw.filled())
        raw = np.array(raw, dtype=np.float32)

    # Flip lat-axis if descending so origin='lower' shows north on top
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        sst_w = sst_w[::-1, :]
        miss_w = miss_w[::-1, :]
        fill_w = fill_w[::-1, :]
        sst_f = sst_f[::-1, :]
        sst_k = sst_k[::-1, :]
        knn_filled = knn_filled[::-1, :]
        land = land[::-1, :]
        raw = raw[::-1, :]

    ocean = ~land
    return dict(
        ts=ts, lat=lat, lon=lon, land=land, ocean=ocean,
        raw=raw,
        sst_w=sst_w, miss_w=miss_w, fill_w=fill_w,
        sst_f=sst_f,
        sst_k=sst_k, knn_filled=knn_filled,
    )


def k_to_c(arr):
    """Convert Kelvin to Celsius for nicer colorbar."""
    return arr - 273.15


def main():
    apply_paper_style()
    d = load_pipeline_frame()

    # Convert to Celsius
    raw_c = k_to_c(d["raw"])
    sst_w_c = k_to_c(d["sst_w"])
    sst_f_c = k_to_c(d["sst_f"])
    sst_k_c = k_to_c(d["sst_k"])

    ocean = d["ocean"]
    # Determine consistent SST color range from filled stage (all ocean covered)
    finite_values = sst_k_c[ocean]
    vmin = np.percentile(finite_values, 1)
    vmax = np.percentile(finite_values, 99)

    # Missing rates within ocean
    def missing_rate(field):
        return float((np.isnan(field) & ocean).sum() / ocean.sum() * 100)

    rate_raw = missing_rate(raw_c)
    rate_w = missing_rate(sst_w_c)
    rate_f = missing_rate(sst_f_c)
    rate_k = missing_rate(sst_k_c)

    # Data aspect = 9°lat / 7°lon ≈ 1.29 (taller than wide)
    fig = plt.figure(figsize=(18, 12.5))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1, 1, 1, 1, 0.20],
        height_ratios=[1, 1],
        hspace=0.06, wspace=0.10,
        left=0.05, right=0.97, top=0.93, bottom=0.06,
    )

    lon = d["lon"]
    lat = d["lat"]
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    titles = [
        "(a) Raw JAXA hourly\n2018-03-09 23:00",
        "(b) Temporal-weighted fill",
        "(c) Gaussian low-pass filter ($\\sigma$=1.5)",
        "(d) 3D progressive KNN fill",
    ]
    rate_strs = [
        f"missing: {rate_raw:.1f}%",
        f"missing: {rate_w:.1f}%",
        f"missing: {rate_f:.1f}%",
        f"missing: {rate_k:.1f}%",
    ]

    sst_panels = [raw_c, sst_w_c, sst_f_c, sst_k_c]

    # Render ordering per panel:
    #   1) land (tan) underlay
    #   2) ocean-missing (sky blue) layer
    #   3) SST colormap on valid pixels
    cmap_sst = CMAPS["sst"]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    last_im = None
    for i, (panel, title, rate_s) in enumerate(zip(sst_panels, titles, rate_strs)):
        ax = fig.add_subplot(gs[0, i])

        # Layer 1: tan land
        land_layer = np.where(d["land"], 1.0, np.nan)
        ax.imshow(land_layer, extent=extent, origin="lower",
                  cmap=mpl.colors.ListedColormap([LAND_COLOR]),
                  aspect="equal", interpolation="nearest")

        # Layer 2: neutral gray ocean-missing (off the SST colormap palette)
        missing_layer = np.where(np.isnan(panel) & d["ocean"], 1.0, np.nan)
        ax.imshow(missing_layer, extent=extent, origin="lower",
                  cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                  aspect="equal", interpolation="nearest")

        # Layer 3: SST values
        disp = np.where(d["ocean"] & ~np.isnan(panel), panel, np.nan)
        im = ax.imshow(disp, extent=extent, origin="lower",
                       cmap=cmap_sst, norm=norm, aspect="equal",
                       interpolation="nearest")
        last_im = im

        ax.set_title(title, fontsize=14, pad=8)
        setup_geo_ax(ax, lon, lat, draw_xlabel=False, draw_ylabel=(i == 0))
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        # Hide x-tick labels on row 1 (row 2 has the same range)
        ax.set_xticklabels([])
        if i != 0:
            ax.set_yticklabels([])
        annotate_metric(ax, rate_s, loc="upper right", fontsize=12)

    # Shared SST colorbar — narrow cax inside the wide right column
    cbar_holder = fig.add_subplot(gs[0, 4])
    cbar_holder.axis("off")
    # Make a slim inset colorbar inside the holder
    pos = cbar_holder.get_position()
    cbar_ax = fig.add_axes([pos.x0 + 0.005, pos.y0,
                            pos.width * 0.28, pos.height])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("SST (°C)", fontsize=13, labelpad=8)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(0.6)

    # ===== Row 2: Mask composition per stage =====
    # Categories per pixel:
    #   0 = land (gray)
    #   1 = missing (dark)
    #   2 = filled by KNN (orange)
    #   3 = filled by temporal weighting (light blue)
    #   4 = original observation (white)

    def build_mask_panel(stage):
        """Category encoding:
          0=land(gray) 1=missing(black) 2=KNN-filled(orange)
          3=temporal-filled(blue) 4=observed(white)
        Semantics:
          miss_w == 1  -> still NaN AFTER temporal fill (= still missing at stage 2)
          fill_w == 1  -> filled by temporal weighting
          observed_orig= NOT(miss_w==1) AND NOT(fill_w==1)
        """
        m = np.zeros(d["land"].shape, dtype=np.uint8)
        ocean = d["ocean"]
        observed = ((d["miss_w"] == 0) & (d["fill_w"] == 0)) & ocean
        filled_temp = (d["fill_w"] == 1) & ocean
        still_missing_after_temp = (d["miss_w"] == 1) & ocean

        if stage == "raw":
            # Raw hourly NC: only original observations exist
            raw_missing = np.isnan(d["raw"])
            m[ocean & raw_missing] = 1
            m[ocean & ~raw_missing] = 4
        elif stage == "weighted":
            m[still_missing_after_temp] = 1
            m[filled_temp] = 3
            m[observed] = 4
        elif stage == "filtered":
            m[still_missing_after_temp] = 1
            m[filled_temp] = 3
            m[observed] = 4
        elif stage == "knn":
            # KNN fills ALL still-missing pixels in ocean
            knn_filled = still_missing_after_temp  # everything that was missing -> KNN
            m[ocean] = 1  # init (will be overwritten)
            m[knn_filled] = 2
            m[filled_temp] = 3
            m[observed] = 4
        m[d["land"]] = 0
        return m

    # Categorical mask panel palette — harmonious earth tones
    cat_colors = [CAT_LAND, CAT_MISSING, CAT_KNN, CAT_TEMPORAL, CAT_OBSERVED]
    cat_cmap = mpl.colors.ListedColormap(cat_colors)
    cat_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

    stage_keys = ["raw", "weighted", "filtered", "knn"]
    row2_titles = [
        "(e) Raw observation mask",
        "(f) After temporal fill",
        "(g) After low-pass filter",
        "(h) After 3D KNN fill",
    ]

    for i, (sk, title) in enumerate(zip(stage_keys, row2_titles)):
        ax = fig.add_subplot(gs[1, i])
        mask_arr = build_mask_panel(sk)
        ax.imshow(mask_arr, extent=extent, origin="lower",
                  cmap=cat_cmap, norm=cat_norm, aspect="equal",
                  interpolation="nearest")
        ax.set_title(title, fontsize=14, pad=8)
        setup_geo_ax(ax, lon, lat, draw_xlabel=True, draw_ylabel=(i == 0))
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        if i != 0:
            ax.set_yticklabels([])

    # Legend for mask categories — placed in dedicated right column, large readable
    legend_ax = fig.add_subplot(gs[1, 4])
    legend_ax.axis("off")
    legend_patches = [
        Patch(facecolor=CAT_OBSERVED, edgecolor="#666", linewidth=0.7, label="Observed"),
        Patch(facecolor=CAT_TEMPORAL, edgecolor="#666", linewidth=0.7, label="Temporal-filled"),
        Patch(facecolor=CAT_KNN, edgecolor="#666", linewidth=0.7, label="KNN-filled"),
        Patch(facecolor=CAT_MISSING, edgecolor="#666", linewidth=0.7, label="Missing"),
        Patch(facecolor=CAT_LAND, edgecolor="#666", linewidth=0.7, label="Land"),
    ]
    legend_ax.legend(handles=legend_patches, loc="center left", frameon=False,
                     fontsize=13, handlelength=1.8, handleheight=1.4,
                     borderpad=0.6, labelspacing=1.0,
                     bbox_to_anchor=(0.0, 0.5))

    fig.suptitle("Three-stage preprocessing pipeline (frame: 2018-03-09)",
                 fontsize=16, y=0.985, fontweight="bold")

    out = OUT_DIR / "fig2_preprocessing.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
