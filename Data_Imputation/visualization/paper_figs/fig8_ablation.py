"""Figure 8: Ablation study bar charts (2x2 layout).

(a) MAE per method (overall)
(b) RMSE per method (overall)
(c) Boundary-region MAE per method (tests boundary loss effect)
(d) High-mask (>65%) MAE per method (tests robustness under heavy missing)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style, save_fig

CACHE = Path(__file__).parent / "cache" / "fig8_stats.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Display labels and colors (left-to-right = simpler-to-richer)
METHOD_ORDER = ["knn", "fno_only", "cbam_basic", "cbam_boundary", "cbam_full", "main"]
METHOD_LABELS = {
    "knn":           "KNN baseline",
    "fno_only":      "FNO only",
    "cbam_basic":    "+ CBAM\n(+grad)",
    "cbam_boundary": "+ boundary\nloss",
    "cbam_full":     "+ temporal\nloss",
    "main":          "main\n(100 ep)",
}
METHOD_COLORS = {
    "knn":           "#9a9a9a",
    "fno_only":      "#d4956b",
    "cbam_basic":    "#e0b48a",
    "cbam_boundary": "#7fa7c4",
    "cbam_full":     "#3b6e8f",
    "main":          "#9b5d6e",
}


def _bar_panel(ax, methods, values, errs, title, ylabel,
               show_legend=False, annotate_value=True):
    xs = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, "#888") for m in methods]
    bars = ax.bar(xs, values, yerr=errs, width=0.66,
                  color=colors, edgecolor="#333", linewidth=0.8,
                  capsize=4, error_kw=dict(ecolor="#444", lw=1.0))
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=8, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.30, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if annotate_value:
        for x, v in zip(xs, values):
            if not np.isfinite(v):
                continue
            ax.text(x, v + (max(values) * 0.02), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, color="#333",
                    fontweight="bold")


def aggregate(records, metric, *, level=None, ratio_min=None, ratio_max=None):
    """Return per-method (mean, std, n) of `metric`, filtered by level / ratio."""
    out = {}
    for m in METHOD_ORDER:
        rs = [r for r in records if r["method"] == m]
        if level:
            rs = [r for r in rs if r["level"] == level]
        if ratio_min is not None:
            rs = [r for r in rs if r["actual_ratio"] >= ratio_min]
        if ratio_max is not None:
            rs = [r for r in rs if r["actual_ratio"] <= ratio_max]
        vals = [r[metric] for r in rs if np.isfinite(r.get(metric, np.nan))]
        out[m] = (float(np.mean(vals)) if vals else np.nan,
                  float(np.std(vals)) if vals else np.nan,
                  len(vals))
    return out


def main():
    apply_paper_style()
    raw = np.load(CACHE, allow_pickle=True)
    records = list(raw["records"])

    methods = [m for m in METHOD_ORDER
               if any(r["method"] == m for r in records)]
    print(f"Methods found in cache: {methods}")

    # ---- Aggregates ----
    agg_mae = aggregate(records, "mae")
    agg_rmse = aggregate(records, "rmse")
    agg_bnd = aggregate(records, "bnd_mae")
    agg_high = aggregate(records, "mae", ratio_min=0.65)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.36, wspace=0.20,
        left=0.07, right=0.97, top=0.93, bottom=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    v = [agg_mae[m][0] for m in methods]
    e = [agg_mae[m][1] for m in methods]
    _bar_panel(ax, methods, v, e, "(a) MAE (all samples)", "MAE (K)")

    ax = fig.add_subplot(gs[0, 1])
    v = [agg_rmse[m][0] for m in methods]
    e = [agg_rmse[m][1] for m in methods]
    _bar_panel(ax, methods, v, e, "(b) RMSE (all samples)", "RMSE (K)")

    ax = fig.add_subplot(gs[1, 0])
    v = [agg_bnd[m][0] for m in methods]
    e = [agg_bnd[m][1] for m in methods]
    _bar_panel(ax, methods, v, e,
               "(c) Boundary-region MAE", "MAE @ mask boundary (K)")

    ax = fig.add_subplot(gs[1, 1])
    v = [agg_high[m][0] for m in methods]
    e = [agg_high[m][1] for m in methods]
    _bar_panel(ax, methods, v, e,
               "(d) MAE under heavy masking (>65%)", "MAE (K)")

    n_per = agg_mae[methods[0]][2] if methods else 0
    fig.suptitle(
        f"Ablation study — incremental contribution of CBAM, boundary, and "
        f"temporal losses  (n={n_per} samples/method)",
        fontsize=14, fontweight="bold", y=0.985,
    )

    out = OUT_DIR / "fig8_ablation.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'method':<18}{'MAE':>10}{'RMSE':>10}{'bnd_MAE':>12}{'MAE>65%':>12}")
    for m in methods:
        print(f"{METHOD_LABELS.get(m, m).replace(chr(10), ' '):<18}"
              f"{agg_mae[m][0]:>10.4f}{agg_rmse[m][0]:>10.4f}"
              f"{agg_bnd[m][0]:>12.4f}{agg_high[m][0]:>12.4f}")


if __name__ == "__main__":
    main()
