"""Figure 1 - study area + cloud-gap motivation.

(a) Regional setting: a dense (representative) SST field over the northern South
    China Sea with land, annotated with the study domain, the Kuroshio intrusion
    pathway (NE, via the Luzon Strait), and a small locator box.
(b) A representative raw hourly Himawari-8 scene: only originally-observed pixels
    are shown; extensive cloud-induced gaps appear in grey.
No cartopy dependency (matches the rest of the figure suite).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from style import (apply_paper_style, CMAPS, setup_geo_ax, annotate_metric,
                   LAND_COLOR, MISSING_OVERLAY)

H5 = Path("/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/"
          "jaxa_knn_filled/jaxa_knn_filled_00.h5")
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_SETTING = "2018-05-15T00:00:00"   # near-complete -> clean regional field
FRAME_CLOUD = "2017-12-04T00:00:00"     # heavy cloud -> gap illustration


def k_to_c(a):
    return a - 273.15


def load():
    import h5py
    with h5py.File(H5, "r") as f:
        sst = f["sst_data"][:]
        obs = f["original_obs_mask"][:]
        land = f["land_mask"][:]
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        ts = [t.decode() if isinstance(t, bytes) else t for t in f["timestamps"][:]]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        sst = sst[:, ::-1, :]; obs = obs[:, ::-1, :]; land = land[::-1, :]
    return sst, obs, land, lat, lon, ts


def draw_land(ax, land, extent):
    ax.imshow(np.where(land == 1, 1.0, np.nan), extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]), aspect="equal",
              interpolation="nearest", zorder=2)


def main():
    apply_paper_style()
    sst, obs, land, lat, lon, ts = load()
    ocean = 1 - land
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    i_set = ts.index(FRAME_SETTING)
    i_cld = ts.index(FRAME_CLOUD)

    field = k_to_c(sst[i_set])
    finite = field[ocean == 1]
    vmin, vmax = np.percentile(finite, 1), np.percentile(finite, 99)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 6.2),
                                   layout="constrained")

    # ---------- (a) regional setting ----------
    draw_land(axA, land, extent)
    disp = np.where(ocean == 1, field, np.nan)
    im = axA.imshow(disp, extent=extent, origin="lower", cmap=CMAPS["sst"],
                    vmin=vmin, vmax=vmax, aspect="equal",
                    interpolation="nearest", zorder=1)
    setup_geo_ax(axA, lon, lat, step=2)
    axA.set_title("(a) Study region — northern South China Sea",
                  fontsize=12, fontweight="bold")

    axA.text(114.2, 18.4, "South China\nSea", fontsize=12, style="italic",
             color="#12407a", ha="center", va="center", zorder=4,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                       alpha=0.55))
    # Kuroshio intrusion arrow from NE corner (via Luzon Strait, east of domain)
    arr = FancyArrowPatch((117.7, 22.6), (116.2, 21.2),
                          arrowstyle="-|>", mutation_scale=18, lw=2.4,
                          color="#7a1f1f", zorder=5,
                          connectionstyle="arc3,rad=-0.3")
    axA.add_patch(arr)
    axA.text(117.75, 23.0, "Kuroshio\nintrusion", fontsize=9.5, color="#7a1f1f",
             ha="right", va="center", fontweight="bold", zorder=5)
    axA.annotate("Luzon Strait", xy=(118.0, 20.6), xytext=(116.0, 19.9),
                 fontsize=9, color="#333", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.2), zorder=5)

    # small locator inset (schematic): SCS box within a coarse E-Asia frame
    axI = axA.inset_axes([0.02, 0.62, 0.30, 0.34])
    axI.set_xlim(105, 128); axI.set_ylim(5, 30)
    axI.add_patch(Rectangle((108, 8), 12, 14, fc="#dfe6ee", ec="#888",
                            lw=0.6))  # schematic landless context
    axI.add_patch(Rectangle((111, 15), 7, 9, fc="none", ec="#c0392b", lw=1.6))
    axI.text(122, 26, "E. Asia", fontsize=7, color="#555")
    axI.set_xticks([]); axI.set_yticks([])
    for s in axI.spines.values():
        s.set_linewidth(0.6)

    # ---------- (b) cloud-gap example ----------
    fld_c = k_to_c(sst[i_cld])
    draw_land(axB, land, extent)
    obs_c = (obs[i_cld] == 1) & (ocean == 1)
    gap = (obs[i_cld] == 0) & (ocean == 1)
    axB.imshow(np.where(gap, 1.0, np.nan), extent=extent, origin="lower",
               cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
               aspect="equal", interpolation="nearest", zorder=1)
    dispB = np.where(obs_c, fld_c, np.nan)
    axB.imshow(dispB, extent=extent, origin="lower", cmap=CMAPS["sst"],
               vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest",
               zorder=1)
    setup_geo_ax(axB, lon, lat, draw_ylabel=False, step=2)
    axB.set_yticklabels([])
    cov = obs_c.sum() / ocean.sum() * 100
    axB.set_title("(b) Raw Himawari-8 SST — cloud gaps",
                  fontsize=12, fontweight="bold")
    annotate_metric(axB, [f"missing: {100 - cov:.0f}%",
                          f"date: {ts[i_cld][:10]}"], loc="lower left",
                    fontsize=10)

    cb = fig.colorbar(im, ax=[axA, axB], location="right", shrink=0.85,
                      pad=0.015, aspect=30)
    cb.set_label("SST (°C)", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    # grey legend patch for "missing"
    axB.add_patch(Rectangle((0, 0), 0, 0, fc=MISSING_OVERLAY,
                            label="cloud / missing"))
    axB.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.95)

    out = OUT_DIR / "fig1_studyarea.png"
    fig.savefig(out, dpi=360)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out}  (setting={ts[i_set][:10]}, cloud={ts[i_cld][:10]}, "
          f"missing={100-cov:.0f}%)")


if __name__ == "__main__":
    main()
