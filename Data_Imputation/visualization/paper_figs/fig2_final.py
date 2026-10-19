"""Figure 2 (publication-ready integrated version): proxy construction pipeline.

Reads the cached fields produced by fig2_v2_proxy.py (output/fig2_parts/
fig2_v2_data.npz) and renders one integrated, publication-grade figure:

  Top row    : same-UTC-hour causal support (t-3d, t-2d, t-1d) with weights,
               feeding (down-arrow) the pipeline.
  Bottom row : proxy construction flow  Target(raw) -> Stage 1 -> Stage 2
               -> Stage 3 (=proxy), with inter-stage arrows, plus the matched
               provenance mask that is generated alongside.
  Shared SST colorbar on the right; single categorical legend for the mask.

Output: output/fig2_final.png / .pdf
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from style import (apply_paper_style, CMAPS, setup_geo_ax,
                   LAND_COLOR, MISSING_OVERLAY,
                   CAT_OBSERVED, CAT_TEMPORAL, CAT_KNN, CAT_MISSING, CAT_LAND)

HERE = Path(__file__).parent
NPZ = HERE / "output" / "fig2_parts" / "fig2_v2_data.npz"
CAT_COLORS = [CAT_OBSERVED, CAT_TEMPORAL, CAT_KNN, CAT_MISSING, CAT_LAND]
CAT_LABELS = ["Observed", "Temporal-filled", "KNN-filled", "Missing", "Land"]


def k_to_c(a):
    return a - 273.15


def draw_sst(ax, field_c, land, lon, lat, vmin, vmax, title=None,
             show_y=True, show_x=True, miss_overlay=True, tag=None):
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(np.where(land, 1.0, np.nan), extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]),
              aspect="equal", interpolation="nearest", zorder=1)
    if miss_overlay:
        miss = np.isnan(field_c) & (~land)
        ax.imshow(np.where(miss, 1.0, np.nan), extent=extent, origin="lower",
                  cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                  aspect="equal", interpolation="nearest", zorder=2)
    disp = np.where(land, np.nan, field_c)
    im = ax.imshow(disp, extent=extent, origin="lower", cmap=CMAPS["sst"],
                   vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest", zorder=3)
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if title:
        ax.set_title(title, fontsize=12, pad=5)
    if tag:
        ax.text(0.035, 0.96, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top", ha="left", color="0.15",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.82), zorder=5)
    return im


def harrow(fig, a_left, a_right, color="0.25"):
    """Horizontal arrow from right edge of a_left to left edge of a_right."""
    pl = a_left.get_position(); pr = a_right.get_position()
    y = (pl.y0 + pl.y1) / 2
    x0 = pl.x1 + 0.004
    x1 = pr.x0 - 0.004
    fig.add_artist(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=18, lw=2.0, color=color, zorder=10))


def main():
    apply_paper_style()
    d = np.load(NPZ, allow_pickle=True)
    lat, lon, land, ocean = d["lat"], d["lon"], d["land"].astype(bool), d["ocean"].astype(bool)
    raw, s1, s2, s3 = (k_to_c(d[k]) for k in ("raw", "stage1", "stage2", "stage3"))
    h3, h2, h1 = (k_to_c(d[k]) for k in ("hist_t3", "hist_t2", "hist_t1"))
    cat, ts = d["cat"], str(d["ts"])

    vmin = float(np.percentile(s3[ocean], 1))
    vmax = float(np.percentile(s3[ocean], 99))

    def mr(f):
        return 100 * (np.isnan(f) & ocean).sum() / ocean.sum()

    fig = plt.figure(figsize=(19.5, 10.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1],
                          height_ratios=[1, 1], hspace=0.26, wspace=0.10,
                          left=0.045, right=0.915, top=0.885, bottom=0.075)
    fig.suptitle("Proxy construction from naturally gappy hourly SST  (target %s UTC)" % ts,
                 fontsize=16, fontweight="bold", y=0.965)

    # ---------- top row: same-UTC-hour historical support ----------
    top = [(h3, "$t-3$d   ($w_3=1/3$)", "a"),
           (h2, "$t-2$d   ($w_2=1/2$)", "b"),
           (h1, "$t-1$d   ($w_1=1$)", "c"),
           (raw, "Target $t$ (raw)   miss %.1f%%" % mr(raw), "d")]
    axes_top = []
    for i, (fld, title, tag) in enumerate(top):
        ax = fig.add_subplot(gs[0, i])
        draw_sst(ax, fld, land, lon, lat, vmin, vmax, title=title,
                 show_y=(i == 0), show_x=False, tag=tag)
        axes_top.append(ax)

    # role annotation (top-right cell)
    axr = fig.add_subplot(gs[0, 4]); axr.axis("off")
    axr.text(0.02, 0.62,
             "Same-UTC-hour\ncausal support\n($D=3$ days)\n\n"
             "inverse-lag weights\n$w_d = 1/d$\n\n"
             "$\\Rightarrow$ Stage 1\ntemporal fill",
             fontsize=12.5, va="center", ha="left", linespacing=1.5)
    _render_bottom(fig, gs, lon, lat, land, ocean, raw, s1, s2, s3, cat,
                   vmin, vmax, mr, axes_top)


def _render_bottom(fig, gs, lon, lat, land, ocean, raw, s1, s2, s3, cat,
                   vmin, vmax, mr, axes_top):
    bot = [(raw, "Target $t$ (raw)   %.1f%%" % mr(raw), "e", True),
           (s1,  "Stage 1: temporal fill   %.1f%%" % mr(s1), "f", True),
           (s2,  "Stage 2: Gaussian ($\\sigma$=1.5)   %.1f%%" % mr(s2), "g", True),
           (s3,  "Stage 3: 3D KNN = proxy   %.1f%%" % mr(s3), "h", True)]
    axes_bot = []
    im = None
    for i, (fld, title, tag, mo) in enumerate(bot):
        ax = fig.add_subplot(gs[1, i])
        im = draw_sst(ax, fld, land, lon, lat, vmin, vmax, title=title,
                      show_y=(i == 0), show_x=True, miss_overlay=mo, tag=tag)
        axes_bot.append(ax)

    # provenance mask panel (bottom-right)
    axc = fig.add_subplot(gs[1, 4])
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    axc.imshow(cat, extent=extent, origin="lower",
               cmap=mpl.colors.ListedColormap(CAT_COLORS),
               vmin=0, vmax=4, aspect="equal", interpolation="nearest")
    setup_geo_ax(axc, lon, lat, draw_xlabel=True, draw_ylabel=False, step=2)
    axc.set_title("Matched mask (provenance)", fontsize=12, pad=5)
    axc.text(0.035, 0.96, "i", transform=axc.transAxes, fontsize=12,
             fontweight="bold", va="top", ha="left", color="0.15",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82))
    axc.legend(handles=[Patch(fc=c, ec="0.5", label=l)
                        for c, l in zip(CAT_COLORS, CAT_LABELS)],
               loc="lower left", fontsize=8.5, framealpha=0.93)

    # shared SST colorbar (right of bottom-row axes, below the role text)
    pb = axes_bot[3].get_position()
    cax = fig.add_axes([0.925, pb.y0, 0.013, pb.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("SST (°C)", fontsize=12)

    fig.canvas.draw()  # positions finalized before drawing arrows
    # inter-stage flow arrows (bottom row): e->f->g->h->mask
    for a, b in zip(axes_bot, axes_bot[1:]):
        harrow(fig, a, b, color="0.25")
    harrow(fig, axes_bot[3], axc, color="#b5651d")  # proxy -> mask

    # support -> pipeline down-arrow: target(top d) into stage1(bottom f)
    pd = axes_top[3].get_position(); pf = axes_bot[1].get_position()
    x = (pf.x0 + pf.x1) / 2
    fig.add_artist(FancyArrowPatch(
        ((pd.x0 + pd.x1) / 2, pd.y0 - 0.004), (x, pf.y1 + 0.004),
        transform=fig.transFigure, connectionstyle="arc3,rad=-0.18",
        arrowstyle="-|>", mutation_scale=18, lw=2.0, color="0.4", zorder=10))
    fig.text((pd.x0 + x) / 2 + 0.01, (pd.y0 + pf.y1) / 2,
             "feeds Stage 1", fontsize=10.5, style="italic", color="0.35",
             ha="left", va="center")

    fig.savefig(HERE / "output" / "fig2_final.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / "output" / "fig2_final.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved output/fig2_final.png / .pdf")


if __name__ == "__main__":
    main()
