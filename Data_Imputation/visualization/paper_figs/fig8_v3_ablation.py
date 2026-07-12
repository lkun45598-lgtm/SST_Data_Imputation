"""Figure 8 v3 — publication-style method comparison.

Three complementary views of the same data:
  Row 1: per-sample MAE DISTRIBUTION (box+violin combined) for each method,
         split by mask condition. Shows median/IQR/outliers, not just mean.

  Row 2: MAE degradation CURVES (mask ratio -> MAE) per method, with
         error bars (std across samples). Distinguishes small (dashed) vs
         large-blob (solid) mask conditions.

  Row 3: HEATMAP of relative improvement of each DL method over the best
         classical baseline (Cubic), per (mask-type × mask-ratio) bin.
         Green = DL wins, red = classical wins.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style, save_fig

CACHE_SMALL = Path(__file__).parent / "cache" / "fig8_stats.npz"
CACHE_SMALL_BL = Path(__file__).parent / "cache" / "baseline_stats.npz"
CACHE_LARGE = Path(__file__).parent / "cache" / "fig8_largeblob_stats.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "linear_2d", "cubic_2d", "dineof", "knn",
    "fno_only", "cbam_basic", "cbam_boundary", "cbam_full", "main",
]
METHOD_LABELS = {
    "linear_2d":     "Linear",
    "cubic_2d":      "Cubic",
    "dineof":        "DINEOF",
    "knn":           "KNN-IDW",
    "fno_only":      "FNO only",
    "cbam_basic":    "+ CBAM (+grad)",
    "cbam_boundary": "+ boundary",
    "cbam_full":     "+ temporal",
    "main":          "Main (ours)",
}
METHOD_COLORS = {
    "linear_2d":     "#c8c8c8",
    "cubic_2d":      "#a8a8a8",
    "dineof":        "#7a7a7a",
    "knn":           "#9a9a9a",
    "fno_only":      "#d4956b",
    "cbam_basic":    "#e0b48a",
    "cbam_boundary": "#7fa7c4",
    "cbam_full":     "#3b6e8f",
    "main":          "#9b5d6e",
}
# Mark DL vs classical for differentiating in plots
IS_DL = {m: m not in {"linear_2d", "cubic_2d", "dineof", "knn"}
         for m in METHOD_ORDER}


def load_records():
    small = list(np.load(CACHE_SMALL, allow_pickle=True)["records"])
    if CACHE_SMALL_BL.exists():
        small.extend(list(np.load(CACHE_SMALL_BL, allow_pickle=True)["records"]))
    large = list(np.load(CACHE_LARGE, allow_pickle=True)["records"])
    return small, large


def get_mae_arr(records, method, level=None):
    vals = [r["mae"] for r in records
            if r["method"] == method
            and (level is None or r["level"] == level)
            and np.isfinite(r.get("mae", np.nan))]
    return np.array(vals)


def main():
    apply_paper_style()
    small, large = load_records()
    methods_present = [m for m in METHOD_ORDER
                       if any(r["method"] == m for r in small + large)]
    print(f"Methods to plot: {methods_present}")

    # ===== Figure layout =====
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.7],
        hspace=0.42, wspace=0.16,
        left=0.06, right=0.97, top=0.95, bottom=0.06,
    )

    LEVELS = [("low", 0.30), ("mid", 0.55), ("high", 0.75)]

    # ----- Row 1: Box+strip distribution per method, split by mask type -----
    for col, (records, title) in enumerate([(small, "Small mask (10-50 px)"),
                                             (large, "Large-blob mask (80-200 px)")]):
        ax = fig.add_subplot(gs[0, col])
        positions = np.arange(len(methods_present))
        all_data = []
        for m in methods_present:
            arr = get_mae_arr(records, m)
            all_data.append(arr if len(arr) > 0 else [np.nan])

        bp = ax.boxplot(all_data, positions=positions, widths=0.58,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#222", lw=1.8),
                        whiskerprops=dict(color="#444", lw=1.0),
                        capprops=dict(color="#444", lw=1.0),
                        boxprops=dict(linewidth=0.8))
        for patch, m in zip(bp["boxes"], methods_present):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_edgecolor("#333")
            patch.set_alpha(0.85)
        # Overlay individual sample points
        rng = np.random.default_rng(0)
        for i, arr in enumerate(all_data):
            if len(arr) == 0:
                continue
            jitter = (rng.random(len(arr)) - 0.5) * 0.30
            ax.scatter(np.full(len(arr), i) + jitter, arr,
                       color="#222", alpha=0.40, s=14, zorder=3,
                       edgecolor="white", linewidth=0.4)

        ax.set_xticks(positions)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods_present],
                           fontsize=10, rotation=30, ha="right")
        ax.set_ylabel("MAE per sample (K)", fontsize=12)
        ax.set_title(f"({chr(97 + col)}) Per-sample MAE distribution — {title}",
                     fontsize=13, pad=8, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.30, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Cap y-axis so DINEOF doesn't dominate
        ax.set_ylim(0, 0.45)
        # Mark DINEOF off-scale if needed
        for i, m in enumerate(methods_present):
            if m == "dineof":
                median = float(np.median(all_data[i]))
                if median > 0.45:
                    ax.text(i, 0.43, f"med={median:.2f}↑",
                            ha="center", va="top", fontsize=9,
                            color="#a83333", fontweight="bold")

    # ----- Row 2: MAE vs mask-ratio CURVES, split by mask type -----
    for col, (records, title) in enumerate([(small, "Small mask"),
                                             (large, "Large-blob mask")]):
        ax = fig.add_subplot(gs[1, col])
        for m in methods_present:
            if m == "dineof":
                continue  # off-scale, plot separately if needed
            means = []
            stds = []
            for lv, _ in LEVELS:
                arr = get_mae_arr(records, m, lv)
                means.append(float(np.mean(arr)) if len(arr) > 0 else np.nan)
                stds.append(float(np.std(arr)) if len(arr) > 0 else 0)
            xs = [r for _, r in LEVELS]
            ls = "-" if IS_DL[m] else "--"
            lw = 2.2 if IS_DL[m] else 1.4
            mk = "o" if IS_DL[m] else "s"
            ax.errorbar(xs, means, yerr=stds,
                        marker=mk, markersize=7, linestyle=ls, linewidth=lw,
                        color=METHOD_COLORS[m], capsize=3.5,
                        ecolor=METHOD_COLORS[m], elinewidth=0.8,
                        label=METHOD_LABELS[m],
                        markeredgecolor="#222", markeredgewidth=0.5)
        ax.set_xlabel("Artificial mask ratio (of observed pixels)", fontsize=12)
        ax.set_ylabel("MAE (K)", fontsize=12)
        ax.set_title(f"({chr(99 + col)}) MAE vs mask ratio — {title}",
                     fontsize=13, pad=8, fontweight="bold")
        ax.set_xticks([r for _, r in LEVELS])
        ax.set_xticklabels([f"{r:.0%}" for _, r in LEVELS])
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.30, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 0.35)
        ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95,
                  ncol=2)

    # ----- Row 3: Heatmap of relative improvement of each method over Cubic -----
    ax_h = fig.add_subplot(gs[2, :])
    # Build (method × condition) matrix of MAE
    cols = []
    col_labels = []
    for mtype, recs in [("Small", small), ("Large", large)]:
        for lv_name, lv_val in LEVELS:
            col_labels.append(f"{mtype}\n{lv_val:.0%}")
            col = []
            for m in methods_present:
                arr = get_mae_arr(recs, m, lv_name)
                col.append(float(np.mean(arr)) if len(arr) > 0 else np.nan)
            cols.append(col)
    M = np.array(cols).T   # (n_methods, n_conditions)

    # Reference = Cubic per column; compute relative reduction (%) over Cubic
    cubic_idx = methods_present.index("cubic_2d")
    ref = M[cubic_idx]
    rel = (ref[None, :] - M) / ref[None, :] * 100.0  # positive = better than cubic
    # Mask Cubic's own row (always 0) for clarity
    rel_disp = rel.copy()

    cmap = plt.get_cmap("RdYlGn")
    vmin, vmax = -50, 50  # ±50% improvement scale
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax_h.imshow(rel_disp, aspect="auto", cmap=cmap, norm=norm,
                     interpolation="nearest")
    ax_h.set_xticks(range(M.shape[1]))
    ax_h.set_xticklabels(col_labels, fontsize=11)
    ax_h.set_yticks(range(M.shape[0]))
    ax_h.set_yticklabels([METHOD_LABELS[m] for m in methods_present], fontsize=10)
    ax_h.set_title("(e) Relative MAE reduction vs Cubic baseline (%) "
                   "— green: method is better, red: worse",
                   fontsize=13, pad=8, fontweight="bold")
    # Annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = rel_disp[i, j]
            mae = M[i, j]
            if np.isnan(val):
                continue
            color = "#000" if abs(val) < 25 else "#fff"
            ax_h.text(j, i, f"{val:+.1f}%\n({mae:.3f})",
                      ha="center", va="center", fontsize=9, color=color)
    # Vertical separator between mask types
    ax_h.axvline(2.5, color="#222", linewidth=1.5)
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.015)
    cbar.set_label("MAE reduction vs Cubic (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Method comparison — distribution, degradation curves, "
        "and relative improvement heatmap",
        fontsize=15, fontweight="bold", y=0.985,
    )

    out = OUT_DIR / "fig8_v3_ablation.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
