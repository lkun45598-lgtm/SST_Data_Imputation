"""Figure 8 — baseline comparison as degradation curves.

MAE vs masking level for Ours (cbam_boundary) against classical baselines
(KNN-IDW, linear, cubic). DINEOF is kept as a failure case but clamped to the
top of the axis with its true value annotated (it is far off-scale).

Left panel:  small square-blob masks.
Right panel: large contiguous-blob masks (the realistic cloud-band regime).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style, save_fig, PALETTE

CACHE = Path(__file__).parent / "cache"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = ["low", "mid", "high"]
YMAX = 0.42

# method key -> (label, color, linestyle, marker)
METHODS = [
    ("main", "FNO-CBAM (ours)", PALETTE["model"], "-", "o"),
    ("knn",           "KNN-IDW",          "#8c8c8c",        "--", "s"),
    ("linear_2d",     "Linear interp",    "#4c9f70",        "--", "^"),
    ("cubic_2d",      "Cubic interp",     "#c9852b",        "--", "D"),
]
DINEOF = ("dineof", "DINEOF (fails)", "#c0392b")


def load(name):
    return list(np.load(CACHE / f"{name}.npz", allow_pickle=True)["records"])


def agg(records, method, metric="mae"):
    """mean, SEM and mean actual_ratio per level for one method."""
    mus, sems, ratios = [], [], []
    for lv in LEVELS:
        vals = [r[metric] for r in records
                if r["method"] == method and r["level"] == lv]
        rr = [r["actual_ratio"] for r in records
              if r["method"] == method and r["level"] == lv]
        if vals:
            mus.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
            ratios.append(np.mean(rr) * 100)
        else:
            mus.append(np.nan); sems.append(0); ratios.append(np.nan)
    return np.array(ratios), np.array(mus), np.array(sems)


def draw_panel(ax, records, title):
    for key, label, color, ls, mk in METHODS:
        x, mu, sem = agg(records, key)
        lw = 2.6 if key == "cbam_boundary" else 1.8
        z = 5 if key == "cbam_boundary" else 3
        ax.plot(x, mu, ls, color=color, marker=mk, markersize=7,
                lw=lw, label=label, zorder=z,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.fill_between(x, mu - sem, mu + sem, color=color, alpha=0.15, zorder=1)

    # DINEOF — off-scale failure, clamp to top with value labels
    key, label, color = DINEOF
    x, mu, _ = agg(records, key)
    if not np.all(np.isnan(mu)):
        yclip = np.full_like(mu, YMAX * 0.965)
        ax.plot(x, yclip, ":", color=color, marker="v", markersize=8,
                lw=1.6, label=label, zorder=4, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.8)
        for xi, mi in zip(x, mu):
            ax.annotate(f"{mi:.2f}", (xi, YMAX * 0.965),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_ylim(0, YMAX)
    ax.set_xlabel("Masked fraction of ocean (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.30, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def main():
    apply_paper_style()
    small = load("fig8_stats") + load("baseline_stats")  # DL+knn + interp/dineof
    large = load("fig8_largeblob_stats")                  # all methods

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True,
                                   layout="constrained")
    draw_panel(axL, small, "(a) Small scattered masks")
    draw_panel(axR, large, "(b) Large contiguous masks (cloud-band regime)")
    axL.set_ylabel("MAE (K)", fontsize=11)
    axR.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.95,
               ncol=1)

    fig.suptitle("Reconstruction accuracy vs. masking severity — "
                 "our model against classical baselines",
                 fontsize=14, fontweight="bold")

    out = OUT_DIR / "fig8_comparison.png"
    fig.savefig(out, dpi=360)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
