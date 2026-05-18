"""Scientific paper figure style + common utilities.

Provides:
- apply_paper_style(): matplotlib rcParams for journal-quality figures
- CMAPS: cmocean-based colormap dict (sst, mask, error, diff, anomaly)
- save_fig(): save with sensible defaults (300 dpi, tight bbox)
- add_scalebar(), annotate_metric(): light helpers
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean
import numpy as np

CMAPS = {
    "sst": plt.get_cmap("RdYlBu_r"),
    "sst_thermal": cmocean.cm.thermal,
    "sst_balance": cmocean.cm.balance,
    "mask": mpl.colors.ListedColormap(["#1a1a1a", "#f0f0f0"]),
    "error": plt.get_cmap("afmhot_r"),
    "error_seq": cmocean.cm.amp,
    "diff": plt.get_cmap("RdBu_r"),
}

# Harmonious earth-tone palette
LAND_COLOR = "#d8c7a3"            # warm sand tan
MISSING_OVERLAY = "#b8b8b8"       # soft warm gray — off the SST cmap
MASKED_OVERLAY = "#a8a8a8"        # slightly darker gray for artificial mask
COAST_BG = "#f4f4f4"
MISSING_COLOR = "#6a7079"         # muted slate (replaces pure black)

# Categorical mask palette (row 2 of fig2 etc.) — same family as land/SST cmap
CAT_OBSERVED = "#f8f1e3"          # warm cream off-white
CAT_TEMPORAL = "#7fa7c4"          # muted steel blue
CAT_KNN = "#d4956b"               # warm terracotta peach
CAT_MISSING = "#6a7079"           # muted slate
CAT_LAND = LAND_COLOR

# (kept for back-compat in other figs that don't overlap with SST cmap)
MASKED_COLOR = "#7fa7c4"

PALETTE = {
    "knn": "#8c8c8c",
    "model": "#1f77b4",
    "gt": "#d62728",
    "input": "#2ca02c",
    "boundary": "#9467bd",
    "fno_only": "#ff7f0e",
    "no_cbam": "#bcbd22",
    "no_bound": "#17becf",
    "no_temp": "#e377c2",
}


def apply_paper_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.dpi": 110,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, out_path, also_pdf=False):
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    if also_pdf and not str(out_path).endswith(".pdf"):
        pdf_path = str(out_path).rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)


def annotate_panel(ax, text, loc="upper left", fontsize=10, color="white",
                   bg="#222222", alpha=0.78, pad=0.04):
    """Top-left letter label, e.g. '(a)'."""
    transform = ax.transAxes
    if loc == "upper left":
        x, y, ha, va = pad, 1 - pad, "left", "top"
    elif loc == "upper right":
        x, y, ha, va = 1 - pad, 1 - pad, "right", "top"
    elif loc == "lower left":
        x, y, ha, va = pad, pad, "left", "bottom"
    elif loc == "lower right":
        x, y, ha, va = 1 - pad, pad, "right", "bottom"
    else:
        x, y, ha, va = pad, 1 - pad, "left", "top"
    ax.text(x, y, text, transform=transform, fontsize=fontsize,
            color=color, ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=bg,
                      edgecolor="none", alpha=alpha))


def annotate_metric(ax, lines, loc="lower right", fontsize=8):
    """Place a metric box like 'MAE: 0.12 K\nRMSE: 0.18 K'."""
    if isinstance(lines, (list, tuple)):
        text = "\n".join(lines)
    else:
        text = lines
    if loc == "lower right":
        x, y, ha, va = 0.97, 0.04, "right", "bottom"
    elif loc == "upper right":
        x, y, ha, va = 0.97, 0.96, "right", "top"
    else:
        x, y, ha, va = 0.03, 0.04, "left", "bottom"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            color="#111111", ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#999999", alpha=0.9, linewidth=0.5))


def setup_geo_ax(ax, lon, lat, draw_xlabel=True, draw_ylabel=True, step=2):
    """Lightweight geo formatting without cartopy dependency."""
    xticks = np.arange(np.ceil(lon.min()), np.floor(lon.max()) + 1, step)
    yticks = np.arange(np.ceil(lat.min()), np.floor(lat.max()) + 1, step)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{int(x)}°E" for x in xticks])
    ax.set_yticklabels([f"{int(y)}°N" for y in yticks])
    if draw_xlabel:
        ax.set_xlabel("Longitude")
    if draw_ylabel:
        ax.set_ylabel("Latitude")
    ax.tick_params(direction="out", length=3, width=0.7, color="#333333")


def shared_cbar(fig, mappable, axes, label, orientation="vertical",
                shrink=0.85, pad=0.02, fraction=0.025):
    cbar = fig.colorbar(mappable, ax=axes, orientation=orientation,
                        shrink=shrink, pad=pad, fraction=fraction)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)
    return cbar
