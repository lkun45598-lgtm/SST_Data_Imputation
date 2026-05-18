"""Figure 6: Error spatial distribution and statistics (2×2 layout).

(a) Mean |error| spatial map across samples
(b) Error histogram with mean / std annotation
(c) Predicted vs ground truth scatter (with 1:1 line and R²)
(d) Per-sample MAE vs mask ratio (binned box/violin)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress, gaussian_kde
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from style import (apply_paper_style, CMAPS, save_fig,
                   setup_geo_ax, annotate_panel, annotate_metric,
                   LAND_COLOR, MISSING_OVERLAY,
                   PALETTE)

CACHE = Path(__file__).parent / "cache" / "fig6_stats.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def k_to_c(arr):
    return arr - 273.15


def main():
    apply_paper_style()
    data = np.load(CACHE, allow_pickle=True)
    spatial_err_sum = data["spatial_err_sum"]
    spatial_err_count = data["spatial_err_count"]
    err_flat = data["err_flat"]            # in Kelvin (= °C diff)
    pred_flat = data["pred_flat"]          # in Kelvin
    gt_flat = data["gt_flat"]              # in Kelvin
    sample_records = data["sample_records"]
    land = data["land_mask"]
    lat = data["lat"]
    lon = data["lon"]

    # Flip lat-axis if descending so origin='lower' renders correctly
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        spatial_err_sum = spatial_err_sum[::-1, :]
        spatial_err_count = spatial_err_count[::-1, :]
        land = land[::-1, :]

    # Mean |error| per pixel (only where count > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_err = np.where(spatial_err_count > 0,
                            spatial_err_sum / np.maximum(spatial_err_count, 1),
                            np.nan)

    # Smooth the spatial error map: 60 samples is sparse per-pixel coverage
    # so each pixel is noisy. A small Gaussian smooth reveals the macro pattern.
    valid_mask = ~np.isnan(mean_err)
    fill_value = np.nanmean(mean_err)
    smoothed = mean_err.copy()
    smoothed[~valid_mask] = fill_value
    smoothed = gaussian_filter(smoothed, sigma=3.0)
    mean_err_smooth = np.where(valid_mask, smoothed, np.nan)

    # ---- Figure ----
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.05, 1],
        hspace=0.30, wspace=0.24,
        left=0.07, right=0.96, top=0.92, bottom=0.07,
    )

    # ===== (a) Spatial mean |error| map =====
    ax_a = fig.add_subplot(gs[0, 0])
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    land_layer = np.where(land == 1, 1.0, np.nan)
    ax_a.imshow(land_layer, extent=extent, origin="lower",
                cmap=mpl.colors.ListedColormap([LAND_COLOR]),
                aspect="equal", interpolation="nearest")
    # Background for ocean pixels not covered
    no_data = (spatial_err_count == 0) & (land == 0)
    bg = np.where(no_data, 1.0, np.nan)
    ax_a.imshow(bg, extent=extent, origin="lower",
                cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                aspect="equal", interpolation="nearest")
    vmax_err = float(np.nanpercentile(mean_err_smooth, 98))
    im_a = ax_a.imshow(mean_err_smooth, extent=extent, origin="lower",
                       cmap=CMAPS["error"], vmin=0, vmax=vmax_err,
                       aspect="equal", interpolation="bilinear")
    setup_geo_ax(ax_a, lon, lat, draw_xlabel=True, draw_ylabel=True, step=2)
    ax_a.tick_params(labelsize=12)
    ax_a.xaxis.label.set_size(13)
    ax_a.yaxis.label.set_size(13)
    ax_a.set_title("(a) Mean |error| spatial distribution",
                   fontsize=15, pad=8, fontweight="bold")
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.040, pad=0.025)
    cb_a.set_label("Mean |error| (K)", fontsize=13)
    cb_a.ax.tick_params(labelsize=12)
    cb_a.outline.set_linewidth(0.6)

    # ===== (b) Error histogram with KDE overlay =====
    ax_b = fig.add_subplot(gs[0, 1])
    err = err_flat  # signed error in K
    mean_e = float(err.mean())
    std_e = float(err.std())
    median_e = float(np.median(err))
    # Robust scale: IQR / 1.349 ≈ std for normal data
    q25, q75 = np.percentile(err, [25, 75])
    iqr = q75 - q25
    # Clip extreme tails for plotting
    lim = float(np.percentile(np.abs(err), 99))
    bins = np.linspace(-lim, lim, 80)
    counts, edges, _ = ax_b.hist(err, bins=bins, color=PALETTE["model"],
                                 edgecolor="#22425a", alpha=0.85, linewidth=0.4,
                                 density=False, label="Observed")
    # KDE fit (much better than normal for heavy-tailed data)
    # Subsample to keep KDE fast
    rng = np.random.default_rng(0)
    sub = rng.choice(err, size=min(50000, len(err)), replace=False)
    kde = gaussian_kde(sub, bw_method=0.06)
    xs = np.linspace(-lim, lim, 400)
    pdf = kde(xs)
    bin_width = edges[1] - edges[0]
    pdf_scaled = pdf * len(err) * bin_width
    ax_b.plot(xs, pdf_scaled, color="#a83333", lw=2.2, label="KDE fit")
    ax_b.axvline(0, color="#444", lw=1, linestyle="--", alpha=0.6)
    ax_b.axvline(mean_e, color="#a83333", lw=1.2, linestyle=":", alpha=0.85,
                 label=f"mean={mean_e:+.3f} K")
    ax_b.set_xlim(-lim, lim)
    ax_b.set_xlabel("Error (K)", fontsize=13)
    ax_b.set_ylabel("Pixel count", fontsize=13)
    ax_b.set_title("(b) Error distribution", fontsize=15, pad=8, fontweight="bold")
    ax_b.tick_params(labelsize=12)
    annotate_metric(ax_b, [
        f"N = {len(err):,}",
        f"mean = {mean_e:+.4f} K",
        f"median = {median_e:+.4f} K",
        f"std = {std_e:.4f} K",
        f"IQR = {iqr:.4f} K",
    ], loc="upper right", fontsize=11)
    ax_b.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)

    # ===== (c) Pred vs GT scatter =====
    ax_c = fig.add_subplot(gs[1, 0])
    gt_c = k_to_c(gt_flat)
    pred_c = k_to_c(pred_flat)
    rng = np.random.default_rng(0)
    # Subsample for plotting if too many
    if len(gt_c) > 40000:
        idx = rng.choice(len(gt_c), 40000, replace=False)
        gt_plot = gt_c[idx]
        pred_plot = pred_c[idx]
    else:
        gt_plot = gt_c
        pred_plot = pred_c

    # 2D histogram for density coloring
    hb = ax_c.hexbin(gt_plot, pred_plot, gridsize=80, cmap="Blues", mincnt=1,
                    bins="log", linewidths=0)
    # 1:1 line
    lo = min(gt_c.min(), pred_c.min())
    hi = max(gt_c.max(), pred_c.max())
    ax_c.plot([lo, hi], [lo, hi], color="#a83333", lw=1.6,
              linestyle="--", label="1:1 line", zorder=3)
    # Regression
    slope, intercept, r_value, _, _ = linregress(gt_c, pred_c)
    xs = np.array([lo, hi])
    ax_c.plot(xs, slope * xs + intercept, color="#333", lw=1.3,
              linestyle="-", alpha=0.85,
              label=f"linear fit: y={slope:.3f}x{intercept:+.2f}", zorder=3)
    ax_c.set_xlim(lo, hi)
    ax_c.set_ylim(lo, hi)
    ax_c.set_xlabel("Ground truth SST (°C)", fontsize=13)
    ax_c.set_ylabel("Predicted SST (°C)", fontsize=13)
    ax_c.set_title("(c) Predicted vs ground truth",
                   fontsize=15, pad=8, fontweight="bold")
    ax_c.tick_params(labelsize=12)
    ax_c.set_aspect("equal")
    annotate_metric(ax_c, [
        f"R² = {r_value**2:.4f}",
        f"r  = {r_value:.4f}",
        f"slope = {slope:.4f}",
    ], loc="lower right", fontsize=12)
    ax_c.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95)
    ax_c.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    cb_c = fig.colorbar(hb, ax=ax_c, fraction=0.040, pad=0.025)
    cb_c.set_label("Pixel count (log scale)", fontsize=12)
    cb_c.ax.tick_params(labelsize=11)
    cb_c.outline.set_linewidth(0.6)

    # ===== (d) Error vs mask ratio (boxplot) =====
    ax_d = fig.add_subplot(gs[1, 1])
    # Bin samples by actual_ratio
    records = list(sample_records)
    ratios = np.array([r["actual_ratio"] for r in records])
    maes = np.array([r["mae"] for r in records])
    levels = np.array([r["level"] for r in records])

    bin_edges = [0.0, 0.40, 0.65, 1.0]
    bin_labels = ["≤40%", "40-65%", ">65%"]
    boxes_data = []
    for lo_, hi_ in zip(bin_edges[:-1], bin_edges[1:]):
        sel = (ratios >= lo_) & (ratios < hi_)
        boxes_data.append(maes[sel])
    positions = np.arange(len(boxes_data))

    bp = ax_d.boxplot(boxes_data, positions=positions, widths=0.58,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="#222", lw=2.0),
                      whiskerprops=dict(color="#444", lw=1.2),
                      capprops=dict(color="#444", lw=1.2),
                      boxprops=dict(linewidth=1.0))
    box_colors = ["#a8c7e0", "#e0b48a", "#d09090"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("#333")
        patch.set_alpha(0.88)
    # Overlay all individual sample points (jittered)
    for i, vals in enumerate(boxes_data):
        if len(vals) == 0:
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.22
        ax_d.scatter(np.full(len(vals), i) + jitter, vals,
                     color="#222", edgecolor="white", linewidth=0.5,
                     alpha=0.70, s=22, zorder=3)
    # Annotated medians on top of each box
    medians = [float(np.median(v)) if len(v) > 0 else np.nan for v in boxes_data]
    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(bin_labels, fontsize=13)
    ax_d.set_xlabel("Mask ratio", fontsize=13)
    ax_d.set_ylabel("Per-sample MAE (K)", fontsize=13)
    ax_d.set_title("(d) Error vs mask ratio",
                   fontsize=15, pad=8, fontweight="bold")
    ax_d.tick_params(labelsize=12)
    ax_d.grid(axis="y", alpha=0.30, linestyle="--", linewidth=0.5)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    # Pad ylim to leave room for annotations
    cur_lim = ax_d.get_ylim()
    ax_d.set_ylim(cur_lim[0], cur_lim[1] * 1.10)
    # Annotate sample counts and median values per bin
    yt = ax_d.get_ylim()[1]
    for i, (vals, med) in enumerate(zip(boxes_data, medians)):
        ax_d.text(i, yt * 0.96, f"n={len(vals)}",
                  ha="center", va="top", fontsize=12, color="#333",
                  fontweight="bold")
        if not np.isnan(med):
            ax_d.text(i, yt * 0.88, f"med={med:.3f} K",
                      ha="center", va="top", fontsize=10, color="#555")

    fig.suptitle("Reconstruction error statistics (60 samples · 3 mask levels)",
                 fontsize=17, fontweight="bold", y=0.97)

    out = OUT_DIR / "fig6_error_stats.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")
    # Print quick summary
    print(f"  Pixels in (b)/(c): {len(err_flat):,}")
    print(f"  Mean signed error: {mean_e:+.4f} K, std: {std_e:.4f} K")
    print(f"  R² = {r_value**2:.4f}, slope = {slope:.4f}")


if __name__ == "__main__":
    main()
