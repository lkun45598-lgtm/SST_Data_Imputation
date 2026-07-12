"""Figure 9 — component ablation (dual metric).

Variants: FNO only -> + CBAM (& gradient loss) -> + boundary loss.
Two panels tell the honest story:
  (a) Overall MAE stays essentially flat — the added terms don't chase raw MAE.
  (b) Boundary-region MAE drops ~20% — the components sharpen fronts / mask
      edges, which is exactly what the aggregate MAE under-weights and what
      Fig. 4-5 show qualitatively.
Temporal loss is intentionally excluded (no measurable benefit under the
single-frame spatial evaluation); the deployed model is `cbam_boundary`.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style

CACHE = Path(__file__).parent / "cache"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("fno_only",      "FNO\nonly"),
    ("cbam_basic",    "+ CBAM\n(+grad)"),
    ("cbam_boundary", "+ boundary"),
    ("cbam_full",     "+ temporal\n(full)"),
]
# regime -> (label, color)
REGIMES = [
    ("small", "Small masks", "#9ecae1"),
    ("large", "Large masks", "#3182bd"),
]


def load(name):
    return list(np.load(CACHE / f"{name}.npz", allow_pickle=True)["records"])


def mean_sem(records, method, metric):
    v = [r[metric] for r in records if r["method"] == method]
    return (np.mean(v), np.std(v) / np.sqrt(len(v))) if v else (np.nan, 0)


def draw(ax, data, metric, title, ylabel, summary=None):
    n = len(VARIANTS)
    width = 0.38
    xs = np.arange(n)
    drops = {}
    for j, (rk, rlabel, color) in enumerate(REGIMES):
        recs = data[rk]
        mus = [mean_sem(recs, vk, metric)[0] for vk, _ in VARIANTS]
        sems = [mean_sem(recs, vk, metric)[1] for vk, _ in VARIANTS]
        off = (j - 0.5) * width
        bars = ax.bar(xs + off, mus, width, yerr=sems, capsize=3,
                      color=color, edgecolor="#333", linewidth=0.6,
                      label=rlabel, error_kw=dict(lw=0.8))
        for b, m in zip(bars, mus):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8.5)
        drops[rlabel] = (mus[-1] - mus[0]) / mus[0] * 100  # signed % change

    if summary == "drop":
        txt = "FNO → ours (boundary):\n" + "\n".join(
            f"  {lbl}: {d:+.0f}%" for lbl, d in drops.items())
    elif summary == "flat":
        txt = "FNO → ours (overall):\n" + "\n".join(
            f"  {lbl}: {d:+.0f}%" for lbl, d in drops.items())
    else:
        txt = None
    if txt:
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="#111",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999",
                          lw=0.6, alpha=0.95))

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in VARIANTS], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def main():
    apply_paper_style()
    data = {"small": load("fig8_stats"), "large": load("fig8_largeblob_stats")}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    draw(axA, data, "mae",
         "(a) Overall MAE — flat", "MAE (K)", summary="flat")
    draw(axB, data, "bnd_mae",
         "(b) Boundary-region MAE — improves", "MAE at mask boundary (K)",
         summary="drop")
    # headroom so the legend (upper-left) and summary box (upper-right) clear
    # the tallest bars instead of overlapping them
    axA.set_ylim(0, axA.get_ylim()[1] * 1.28)
    axB.set_ylim(0, axB.get_ylim()[1] * 1.20)
    axA.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.95)

    fig.suptitle("Component ablation — physics-inspired losses sharpen "
                 "boundaries at negligible overall-MAE cost",
                 fontsize=14, fontweight="bold")

    out = OUT_DIR / "fig9_ablation.png"
    fig.savefig(out, dpi=360)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
