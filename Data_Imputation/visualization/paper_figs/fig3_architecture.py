"""Figure 3: FNO-CBAM-Temporal architecture diagram.

Layout:
  Top:    Overall pipeline (Input → Lifting → FNO×6 → Projection → Output)
  Bottom: A single FNO block expanded
            (SpectralConv2d ‖ LocalConv) → ⊕ → CBAM → LayerNorm → GELU + Residual
            CBAM expanded into Channel-Attention and Spatial-Attention sub-modules
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style, save_fig

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cohesive palette (matches fig2)
C_INPUT = "#d8c7a3"           # tan
C_LIFT = "#f8f1e3"            # cream
C_FNO = "#7fa7c4"             # steel blue
C_SPEC = "#a8c3da"            # lighter blue
C_LOCAL = "#d4956b"           # terracotta
C_CBAM = "#e0b48a"            # peach
C_NORM = "#c8c8c8"            # gray
C_OUTPUT = "#9bbf95"          # muted sage green
C_ARROW = "#444444"
C_TEXT = "#222222"
C_RESID = "#9b5d6e"           # muted plum for residual line


def rounded_box(ax, x, y, w, h, color, label, sub=None, fontsize=11,
                edgecolor="#555555", lw=1.2, fontweight="normal"):
    """Draw a rounded box centered at (x,y) with width w and height h."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.04",
                         facecolor=color, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(box)
    if sub is None:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=C_TEXT, fontweight=fontweight)
    else:
        ax.text(x, y + h*0.18, label, ha="center", va="center",
                fontsize=fontsize, color=C_TEXT, fontweight=fontweight)
        ax.text(x, y - h*0.22, sub, ha="center", va="center",
                fontsize=fontsize - 2, color="#555555", style="italic")


def arrow(ax, x0, y0, x1, y1, label=None, label_pos=0.5, label_offset=0.06,
          color=C_ARROW, lw=1.4, style="->", curve=0, fontsize=9):
    if curve == 0:
        connstyle = "arc3,rad=0"
    else:
        connstyle = f"arc3,rad={curve}"
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          arrowstyle=style, color=color, lw=lw,
                          mutation_scale=12, connectionstyle=connstyle)
    ax.add_patch(arr)
    if label:
        xm = x0 + (x1 - x0) * label_pos
        ym = y0 + (y1 - y0) * label_pos + label_offset
        ax.text(xm, ym, label, ha="center", va="bottom",
                fontsize=fontsize, color="#555555", style="italic")


def circle_node(ax, x, y, r, label, color="#ffffff", edgecolor="#333", fontsize=12):
    c = plt.Circle((x, y), r, facecolor=color, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=C_TEXT, fontweight="bold")


def draw_top_pipeline(ax):
    """Overall data flow from inputs to output."""
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Section title
    ax.text(7, 4.7, "(a) Overall FNO-CBAM-Temporal pipeline",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color=C_TEXT)

    # ===== Inputs (left) =====
    rounded_box(ax, 1.0, 3.3, 1.6, 0.85, C_INPUT,
                "SST sequence", sub="[B, 30, H, W]", fontsize=11)
    rounded_box(ax, 1.0, 1.7, 1.6, 0.85, C_INPUT,
                "Mask sequence", sub="[B, 30, H, W]", fontsize=11)

    # ===== Encoders =====
    rounded_box(ax, 3.0, 3.3, 1.4, 0.85, C_LIFT,
                "SST encoder", sub="Linear → W/2", fontsize=10)
    rounded_box(ax, 3.0, 1.7, 1.4, 0.85, C_LIFT,
                "Mask encoder", sub="Linear → W/2", fontsize=10)

    # ===== Concat node =====
    circle_node(ax, 4.6, 2.5, 0.18, "⊕", color="#ffffff", fontsize=14)
    ax.text(4.6, 2.18, "concat", ha="center", va="top", fontsize=8.5,
            color="#555555", style="italic")

    # ===== Lifted features =====
    rounded_box(ax, 5.8, 2.5, 1.3, 0.85, C_LIFT,
                "Lifted", sub="[B, 64, H, W]", fontsize=10)

    # ===== FNO blocks ×6 =====
    fno_box_x = 8.4
    rounded_box(ax, fno_box_x, 2.5, 2.4, 1.3, C_FNO,
                "FNO + CBAM Block",
                sub="×6 (depth=6)",
                fontsize=12, fontweight="bold")

    # ===== Projection =====
    rounded_box(ax, 11.2, 2.5, 1.3, 0.85, C_LIFT,
                "Projection", sub="Linear → 1ch", fontsize=10)

    # ===== Output =====
    rounded_box(ax, 13.0, 2.5, 1.4, 0.85, C_OUTPUT,
                "Output SST", sub="[B, 1, H, W]", fontsize=11, fontweight="bold")

    # ===== Arrows =====
    arrow(ax, 1.8, 3.3, 2.3, 3.3)
    arrow(ax, 1.8, 1.7, 2.3, 1.7)
    arrow(ax, 3.7, 3.3, 4.42, 2.62)
    arrow(ax, 3.7, 1.7, 4.42, 2.38)
    arrow(ax, 4.78, 2.5, 5.15, 2.5)
    arrow(ax, 6.45, 2.5, 7.2, 2.5, label="[B, 64, H, W]", label_offset=0.18, fontsize=9)
    arrow(ax, 9.6, 2.5, 10.55, 2.5)
    arrow(ax, 11.85, 2.5, 12.3, 2.5)

    # ===== Detail callout: link top FNO block to bottom-half expanded view =====
    ax.annotate("", xy=(fno_box_x, 1.70), xytext=(fno_box_x, 1.85),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0,
                                linestyle=(0, (3, 2))))
    ax.text(fno_box_x, 1.40, "expanded below ↓",
            ha="center", va="top", fontsize=10, color="#666666", style="italic")


def draw_bottom_block(ax):
    """Expanded single FNO+CBAM block."""
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Section title
    ax.text(7, 5.7, "(b) FNO + CBAM block (expanded)",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color=C_TEXT)

    # ===== Input feature =====
    rounded_box(ax, 0.8, 3.0, 1.3, 0.85, C_LIFT,
                "Input feature", sub="[B, W, H, W']", fontsize=10)

    # ===== Branch split node =====
    circle_node(ax, 2.3, 3.0, 0.10, "", color="#ffffff", fontsize=10)

    # ===== Two parallel branches =====
    # Top branch: Spectral
    rounded_box(ax, 4.0, 4.0, 1.8, 0.85, C_SPEC,
                "SpectralConv2d", sub="FFT → modes(80×64) → iFFT", fontsize=10)
    # Bottom branch: Local
    rounded_box(ax, 4.0, 2.0, 1.8, 0.85, C_LOCAL,
                "Local Conv 1×1", sub="pointwise", fontsize=10)

    # ===== Sum node =====
    circle_node(ax, 5.8, 3.0, 0.18, "+", color="#ffffff", fontsize=15)

    # ===== CBAM module (compound block) =====
    cbam_x, cbam_y, cbam_w, cbam_h = 8.4, 3.0, 3.2, 2.2
    cbam_outer = FancyBboxPatch(
        (cbam_x - cbam_w/2, cbam_y - cbam_h/2), cbam_w, cbam_h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        facecolor=C_CBAM, edgecolor="#8a6e4c", linewidth=1.4)
    ax.add_patch(cbam_outer)
    ax.text(cbam_x, cbam_y + cbam_h/2 - 0.18, "CBAM",
            ha="center", va="top", fontsize=12, fontweight="bold", color=C_TEXT)

    # Channel attention sub-block
    rounded_box(ax, cbam_x - 0.7, cbam_y - 0.05, 1.2, 0.7,
                "#f8e9d6", "Channel Att.",
                sub="AvgPool+MaxPool→MLP→σ", fontsize=9)
    # Spatial attention sub-block
    rounded_box(ax, cbam_x + 0.7, cbam_y - 0.05, 1.2, 0.7,
                "#f8e9d6", "Spatial Att.",
                sub="conv7×7→σ", fontsize=9)
    # Internal arrow between attentions
    arrow(ax, cbam_x - 0.10, cbam_y - 0.05, cbam_x + 0.10, cbam_y - 0.05, lw=1.0)

    # ===== LayerNorm =====
    rounded_box(ax, 11.0, 3.0, 1.2, 0.85, C_NORM,
                "LayerNorm", fontsize=10)

    # ===== Residual + GELU node =====
    circle_node(ax, 12.4, 3.0, 0.20, "+", color="#ffffff", fontsize=15)

    # ===== Output =====
    rounded_box(ax, 13.4, 3.0, 1.1, 0.85, C_LIFT,
                "GELU →", sub="next block", fontsize=10)

    # ===== Arrows =====
    arrow(ax, 1.45, 3.0, 2.2, 3.0)
    # split into two branches
    arrow(ax, 2.4, 3.05, 3.1, 4.0)
    arrow(ax, 2.4, 2.95, 3.1, 2.0)
    # branches join at sum
    arrow(ax, 4.9, 4.0, 5.65, 3.10)
    arrow(ax, 4.9, 2.0, 5.65, 2.90)
    # sum → CBAM
    arrow(ax, 5.98, 3.0, 6.78, 3.0)
    # CBAM → LayerNorm
    arrow(ax, 10.0, 3.0, 10.4, 3.0)
    # LayerNorm → +
    arrow(ax, 11.6, 3.0, 12.2, 3.0)
    # Output
    arrow(ax, 12.6, 3.0, 12.85, 3.0)

    # ===== Residual skip connection =====
    # From input → bypass → sum node
    arr = FancyArrowPatch((0.8, 2.55), (12.4, 2.55),
                          arrowstyle="-", color=C_RESID, lw=1.6,
                          linestyle=(0, (5, 3)),
                          connectionstyle="arc3,rad=-0.18")
    ax.add_patch(arr)
    arr_tip = FancyArrowPatch((12.40, 2.62), (12.40, 2.83),
                              arrowstyle="->", color=C_RESID, lw=1.6,
                              mutation_scale=12)
    ax.add_patch(arr_tip)
    ax.text(6.6, 1.45, "residual (skip connection)",
            ha="center", va="center", fontsize=10, color=C_RESID, style="italic")


def main():
    apply_paper_style()
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.08,
                          left=0.02, right=0.98, top=0.96, bottom=0.03)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])
    draw_top_pipeline(ax_top)
    draw_bottom_block(ax_bot)

    out = OUT_DIR / "fig3_architecture.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
