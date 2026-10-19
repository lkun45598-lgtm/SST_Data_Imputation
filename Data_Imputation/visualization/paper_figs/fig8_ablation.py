"""Figure 8: Method comparison bar charts (2x2 layout).

Compares FNO-CBAM (ours) against traditional and deep-learning baselines on
held-out JAXA series 8 (same frames + same real-cloud masks per method).

(a) MAE (all samples)
(b) RMSE (all samples)
(c) Deep-gap MAE (withheld pixels >=30 km from any real observation)
(d) MAE under heavy masking (>65%)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style, save_fig

CACHE = Path(__file__).parent / "cache" / "all_methods_full.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Methods ordered: classical baselines (gray) -> DL baselines (cool) -> ours (warm)
METHOD_ORDER = [
    "linear", "cubic", "knn", "oi", "dineof",  # traditional / classical baselines
    "dincae", "unet", "convlstm",              # deep-learning baselines
    "fno",                                     # ours
]
METHOD_LABELS = {
    "linear":   "Linear",
    "cubic":    "Cubic",
    "knn":      "KNN-IDW",
    "oi":       "OI /\nKriging",
    "dineof":   "DINEOF",
    "dincae":   "DINCAE",
    "unet":     "U-Net",
    "convlstm": "ConvLSTM",
    "fno":      "FNO-CBAM\n(ours)",
}
METHOD_COLORS = {
    "linear":   "#d8d8d8",
    "cubic":    "#c8c8c8",
    "knn":      "#b8b8b8",
    "oi":       "#a0a0a0",
    "dineof":   "#888888",
    "dincae":   "#8fb0cc",
    "unet":     "#6f97bd",
    "convlstm": "#4f7ba6",
    "fno":      "#9b5d6e",
}


def _bar_panel(ax, methods, values, errs, title, ylabel, ymax=None):
    """Plot bars; values above `ymax` are clipped and labelled 'off scale'."""
    xs = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, "#888") for m in methods]
    values_arr = np.array(values, dtype=float)
    errs_arr = np.array(errs, dtype=float)

    # Soft cap so far-outlier bars (e.g. DINEOF) don't dominate the y-axis
    if ymax is None:
        sorted_v = np.sort(values_arr[np.isfinite(values_arr)])
        if len(sorted_v) >= 2:
            ymax = max(float(sorted_v[-2]) * 1.5, 0.05)
        else:
            ymax = float(np.nanmax(values_arr) * 1.1)

    plot_values = np.minimum(values_arr, ymax)
    over_cap = values_arr > ymax

    # Asymmetric whiskers: clip the upper end to the visible panel top so that
    # high-variance methods (Cubic, DINEOF) don't shoot a whisker off-scale and
    # visually dominate; lower end is clipped at 0. Bar heights / y-limits stay
    # unchanged — only the drawn whisker length is capped.
    top = ymax * 1.16
    lo_err = np.where(over_cap, 0.0, np.minimum(errs_arr, plot_values))
    hi_err = np.where(over_cap, 0.0,
                      np.minimum(errs_arr, np.maximum(top - plot_values, 0.0)))
    plot_errs = np.vstack([lo_err, hi_err])

    ax.bar(xs, plot_values, yerr=plot_errs, width=0.66,
           color=colors, edgecolor="#333", linewidth=0.8,
           capsize=4, error_kw=dict(ecolor="#444", lw=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       fontsize=9.5, rotation=20, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=8, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.30, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ymax * 1.18)

    for x, v, capped in zip(xs, values_arr, over_cap):
        if not np.isfinite(v):
            continue
        if capped:
            ax.text(x, ymax * 1.02, f"{v:.3f}↑",
                    ha="center", va="bottom", fontsize=10,
                    color="#a83333", fontweight="bold")
            ax.text(x, ymax * 0.97, "off",
                    ha="center", va="top", fontsize=8,
                    color="#a83333", style="italic")
        else:
            ax.text(x, v + ymax * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10,
                    color="#333", fontweight="bold")


def aggregate(records, metric, *, ratio_min=None, ratio_max=None):
    """Return per-method (mean, std, n) of `metric`, optionally filtered by ratio."""
    out = {}
    for m in METHOD_ORDER:
        rs = [r for r in records if r["method"] == m]
        if ratio_min is not None:
            rs = [r for r in rs if r.get("target_ratio", 0) >= ratio_min]
        if ratio_max is not None:
            rs = [r for r in rs if r.get("target_ratio", 1) <= ratio_max]
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
    print(f"Methods to plot: {methods}")

    # ---- Aggregates ----
    agg_mae = aggregate(records, "mae")
    agg_rmse = aggregate(records, "rmse")
    agg_deep = aggregate(records, "mae_deep")            # deep-gap (>=30km) MAE
    agg_high = aggregate(records, "mae", ratio_min=0.65)  # heavy masking

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.38, wspace=0.20,
        left=0.07, right=0.97, top=0.92, bottom=0.09,
    )

    ax = fig.add_subplot(gs[0, 0])
    _bar_panel(ax, methods,
               [agg_mae[m][0] for m in methods],
               [agg_mae[m][1] for m in methods],
               "(a) MAE (all samples)", "MAE (K)")

    ax = fig.add_subplot(gs[0, 1])
    _bar_panel(ax, methods,
               [agg_rmse[m][0] for m in methods],
               [agg_rmse[m][1] for m in methods],
               "(b) RMSE (all samples)", "RMSE (K)")

    ax = fig.add_subplot(gs[1, 0])
    _bar_panel(ax, methods,
               [agg_deep[m][0] for m in methods],
               [agg_deep[m][1] for m in methods],
               "(c) Deep-gap MAE (≥30 km from observations)", "MAE (K)")

    ax = fig.add_subplot(gs[1, 1])
    _bar_panel(ax, methods,
               [agg_high[m][0] for m in methods],
               [agg_high[m][1] for m in methods],
               "(d) MAE under heavy masking (>65%)", "MAE (K)")

    n_per = agg_mae[methods[0]][2] if methods else 0
    fig.suptitle(
        f"Method Comparison: FNO-CBAM vs. Traditional and Deep-Learning "
        f"Baselines  (n={n_per} samples/method)",
        fontsize=15, fontweight="bold", y=0.985,
    )

    out = OUT_DIR / "fig8_ablation.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'method':<18}{'MAE':>10}{'RMSE':>10}{'deep_MAE':>12}{'MAE>65%':>12}")
    for m in methods:
        print(f"{METHOD_LABELS.get(m, m).replace(chr(10), ' '):<18}"
              f"{agg_mae[m][0]:>10.4f}{agg_rmse[m][0]:>10.4f}"
              f"{agg_deep[m][0]:>12.4f}{agg_high[m][0]:>12.4f}")


if __name__ == "__main__":
    main()
