"""Figure 8 v2 — Method comparison under TWO mask conditions side-by-side.

Left column:  small scattered mask (10-50 px squares) — easier case
Right column: large-blob mask  (80-200 px squares) — realistic cloud cover

Shows that:
  - DL methods stay roughly stable
  - Cubic / Linear / KNN degrade significantly on large blobs
  - The DL advantage GROWS with mask size

Layout: 2 rows × 2 cols
  (a) Small mask, MAE         (b) Large-blob mask, MAE
  (c) Small mask, RMSE        (d) Large-blob mask, RMSE
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style, save_fig

CACHE_SMALL = Path(__file__).parent / "cache" / "fig8_stats.npz"
CACHE_SMALL_BL = Path(__file__).parent / "cache" / "baseline_stats.npz"
CACHE_LARGE = Path(__file__).parent / "cache" / "fig8_largeblob_stats.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "linear_2d", "cubic_2d", "dineof", "knn",
    "fno_only", "cbam_basic", "cbam_boundary", "cbam_full",
    "main",
]
METHOD_LABELS = {
    "linear_2d":     "Linear",
    "cubic_2d":      "Cubic",
    "dineof":        "DINEOF",
    "knn":           "KNN-IDW",
    "fno_only":      "FNO\nonly",
    "cbam_basic":    "+ CBAM\n(+grad)",
    "cbam_boundary": "+ bnd",
    "cbam_full":     "+ temp",
    "main":          "Main\n(ours)",
}
METHOD_COLORS = {
    "linear_2d":     "#c8c8c8",
    "cubic_2d":      "#a8a8a8",
    "dineof":        "#888888",
    "knn":           "#9a9a9a",
    "fno_only":      "#d4956b",
    "cbam_basic":    "#e0b48a",
    "cbam_boundary": "#7fa7c4",
    "cbam_full":     "#3b6e8f",
    "main":          "#9b5d6e",
}


def load_records():
    """Return (records_small, records_large)."""
    small = list(np.load(CACHE_SMALL, allow_pickle=True)["records"])
    if CACHE_SMALL_BL.exists():
        small.extend(list(np.load(CACHE_SMALL_BL, allow_pickle=True)["records"]))
    large = list(np.load(CACHE_LARGE, allow_pickle=True)["records"])
    return small, large


def aggregate(records, metric):
    out = {}
    for m in METHOD_ORDER:
        rs = [r[metric] for r in records
              if r["method"] == m and np.isfinite(r.get(metric, np.nan))]
        out[m] = (float(np.mean(rs)) if rs else np.nan,
                  float(np.std(rs)) if rs else np.nan,
                  len(rs))
    return out


def bar_panel(ax, methods, values, errs, title, ylabel, ymax=None):
    xs = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, "#888") for m in methods]
    v = np.array(values, dtype=float)
    e = np.array(errs, dtype=float)
    if ymax is None:
        sorted_v = np.sort(v[np.isfinite(v)])
        ymax = float(sorted_v[-2]) * 1.55 if len(sorted_v) >= 2 else float(np.nanmax(v) * 1.1)
        ymax = max(ymax, 0.05)
    plot_v = np.minimum(v, ymax)
    plot_e = np.where(v > ymax, 0, e)
    over = v > ymax

    ax.bar(xs, plot_v, yerr=plot_e, width=0.66,
           color=colors, edgecolor="#333", linewidth=0.8,
           capsize=3.5, error_kw=dict(ecolor="#444", lw=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=8, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.30, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ymax * 1.20)

    for x, val, capped in zip(xs, v, over):
        if not np.isfinite(val):
            continue
        if capped:
            ax.text(x, ymax * 1.04, f"{val:.3f}↑",
                    ha="center", va="bottom", fontsize=9,
                    color="#a83333", fontweight="bold")
        else:
            ax.text(x, val + ymax * 0.02, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    color="#222", fontweight="bold")


def main():
    apply_paper_style()
    small, large = load_records()
    methods_present = [m for m in METHOD_ORDER
                       if any(r["method"] == m for r in small + large)]
    print(f"Methods to plot: {methods_present}")

    # Aggregate per condition
    agg_mae_s = aggregate(small, "mae")
    agg_mae_l = aggregate(large, "mae")
    agg_rmse_s = aggregate(small, "rmse")
    agg_rmse_l = aggregate(large, "rmse")

    # Use a SHARED ymax per metric. DINEOF is treated as a far outlier
    # — we cap the axis to the max of NON-DINEOF methods so the relevant
    # contrasts are visually resolved.
    non_dineof = [m for m in methods_present if m != "dineof"]
    all_mae_nd = ([agg_mae_s[m][0] for m in non_dineof] +
                  [agg_mae_l[m][0] for m in non_dineof])
    all_rmse_nd = ([agg_rmse_s[m][0] for m in non_dineof] +
                   [agg_rmse_l[m][0] for m in non_dineof])
    max_mae_nd = max(v for v in all_mae_nd if np.isfinite(v))
    max_rmse_nd = max(v for v in all_rmse_nd if np.isfinite(v))
    ymax_mae = max_mae_nd * 1.25
    ymax_rmse = max_rmse_nd * 1.25

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.40, wspace=0.18,
        left=0.06, right=0.97, top=0.92, bottom=0.09,
    )

    ax = fig.add_subplot(gs[0, 0])
    v = [agg_mae_s[m][0] for m in methods_present]
    e = [agg_mae_s[m][1] for m in methods_present]
    bar_panel(ax, methods_present, v, e,
              "(a) MAE — Small mask (10-50 px squares)",
              "MAE (K)", ymax=ymax_mae)

    ax = fig.add_subplot(gs[0, 1])
    v = [agg_mae_l[m][0] for m in methods_present]
    e = [agg_mae_l[m][1] for m in methods_present]
    bar_panel(ax, methods_present, v, e,
              "(b) MAE — Large-blob mask (80-200 px squares)",
              "MAE (K)", ymax=ymax_mae)

    ax = fig.add_subplot(gs[1, 0])
    v = [agg_rmse_s[m][0] for m in methods_present]
    e = [agg_rmse_s[m][1] for m in methods_present]
    bar_panel(ax, methods_present, v, e,
              "(c) RMSE — Small mask",
              "RMSE (K)", ymax=ymax_rmse)

    ax = fig.add_subplot(gs[1, 1])
    v = [agg_rmse_l[m][0] for m in methods_present]
    e = [agg_rmse_l[m][1] for m in methods_present]
    bar_panel(ax, methods_present, v, e,
              "(d) RMSE — Large-blob mask",
              "RMSE (K)", ymax=ymax_rmse)

    n_small = agg_mae_s[methods_present[0]][2] if methods_present else 0
    n_large = agg_mae_l[methods_present[0]][2] if methods_present else 0
    fig.suptitle(
        f"Method comparison across mask conditions  "
        f"(small mask n={n_small}, large-blob n={n_large} per method)",
        fontsize=15, fontweight="bold", y=0.985,
    )

    out = OUT_DIR / "fig8_v2_ablation.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")

    # Summary
    print(f"\n{'method':<15}{'small MAE':>12}{'large MAE':>12}{'degrad %':>10}")
    for m in methods_present:
        s = agg_mae_s[m][0]
        l = agg_mae_l[m][0]
        if np.isfinite(s) and np.isfinite(l):
            deg = (l - s) / s * 100
            print(f"{METHOD_LABELS[m].replace(chr(10),' '):<15}{s:>12.4f}{l:>12.4f}{deg:>9.1f}%")


if __name__ == "__main__":
    main()
