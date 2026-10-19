"""Figure 2 supplement: bare map tiles for the KNN history support + mask branch.

Produces decoration-free map tiles (no titles, axes, colorbars, or annotation
boxes) so they can be assembled into the final Figure 2 composite, which adds all
text / weights / missing rates / colorbars itself. Every tile shares the same
lon/lat extent, pixel dimensions, and SST colour scale.

Exports (into output/fig2_parts/tiles/):
  proxy_tm1 .. proxy_tm7   7 completed historical proxy fields = Stage-3 output
                           of the SAME UTC hour on days t-1 .. t-7 (NOT the raw
                           gappy history). Displayed as 3 offset cards + "x7 days".
  obs_mask                 target-time observation mask (Observed/Missing/Land),
                           i.e. the network's M_t^obs, fixed thereafter.
  ftemp_mask               Stage-1 temporal-filled pixels (F_temp) only.
  fknn_mask                Stage-3 KNN-filled pixels (F_KNN) only.

Reuses the deployed pipeline exactly as fig2_v2_proxy.py (observation-preserving
Stage 1/2/3). Because the Stage-3 causal window is 7 days and WARMUP_DAYS >> 7,
the last 8 frames' proxy is identical to a full-series run.

Usage:
  python fig2_supplement_tiles.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).parent))
PRE = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation/preprocessing")
sys.path.insert(0, str(PRE))
import hourly_preprocess as hp
from knn_fill_3d import progressive_knn_fill_3d
from style import (apply_paper_style, CMAPS, LAND_COLOR, MISSING_OVERLAY,
                   CAT_OBSERVED, CAT_MISSING, CAT_LAND)

OUT_DIR = Path(__file__).parent / "output" / "fig2_parts" / "tiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match fig2_v2_proxy.py exactly.
TARGET = datetime(2018, 3, 9, 23, 0, 0)
HOUR = 23
WARMUP_DAYS = 40
N_HIST = 7          # KNN same-hour history depth (t-1 .. t-7)

FIGSIZE = (3.51, 4.51)   # data is 351(W) x 451(H); identical for every tile
DPI = 200


def k_to_c(a):
    return a - 273.15


def reproduce_pipeline():
    """Run the deployed observation-preserving pipeline on a WARMUP-day
    same-UTC-hour sub-stack ending at TARGET. Returns the full stage stacks +
    per-frame masks so the caller can slice the last N_HIST+1 frames."""
    start_time = (TARGET - timedelta(days=WARMUP_DAYS)).replace(hour=HOUR)

    sst_l, miss_l, fill_l = [], [], []
    lat = lon = None
    for day in range(WARMUP_DAYS + 1):
        tt = start_time + timedelta(days=day)
        filled, _orig, miss_before, _ = hp.temporal_weighted_fill_frame(
            tt, start_time, HOUR)
        if filled is None:
            continue
        if lat is None:
            _, lat, lon = hp.load_jaxa_frame(tt)
        miss_after = np.isnan(filled)
        fill_l.append((~miss_after & miss_before).astype(np.uint8))
        sst_l.append(filled)
        miss_l.append(miss_after.astype(np.uint8))

    sst = np.array(sst_l, np.float32)          # Stage 1 output (all frames)
    miss = np.array(miss_l, np.uint8)
    fillm = np.array(fill_l, np.uint8)
    land = np.all(np.isnan(sst), axis=0).astype(np.uint8)

    # Stage 2: mask-preserving Gaussian (only temporal-fill pixels rewritten)
    filt = np.zeros_like(sst)
    for i in range(len(sst)):
        filt[i] = hp.gaussian_filter_frame(sst[i], miss[i], fillm[i])

    # Stage 3: 3D causal progressive KNN over the sub-stack
    filled3d, _ = progressive_knn_fill_3d(
        filt, miss, land, k=hp.KNN_K, radius=hp.KNN_RADIUS, power=hp.KNN_POWER,
        time_weight=0.9, space_weight=1.0, window_size=7, batch_size=500)

    return dict(lat=lat, lon=lon, land=land.astype(bool),
                stage3=filled3d,                 # (T,H,W) completed proxy stack
                miss_after_s1=miss.astype(bool), # still-missing after Stage 1
                tfill=fillm.astype(bool))        # Stage-1 temporal-filled pixels


def _bare_ax(fig):
    """Axes filling the whole figure, no ticks/spines/labels."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def _save(fig, name):
    """Save png+pdf with NO tight bbox / padding, so the map fills the tile
    edge-to-edge (the paper style sets savefig.bbox='tight' by default)."""
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=DPI,
                    bbox_inches=None, pad_inches=0)
    plt.close(fig)


def save_sst_tile(name, field_c, land, extent, vmin, vmax):
    """Bare SST map tile: land in sand tan, ocean in the SST cmap. No decoration."""
    fig = plt.figure(figsize=FIGSIZE)
    ax = _bare_ax(fig)
    # figsize is set to the lon/lat degree ratio, so aspect="auto" fills the
    # frame edge-to-edge with no white margin and no geometric distortion.
    ax.imshow(np.where(land, 1.0, np.nan), extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]),
              vmin=0, vmax=1, aspect="auto", interpolation="nearest", zorder=1)
    disp = np.where(land, np.nan, field_c)
    ax.imshow(disp, extent=extent, origin="lower", cmap=CMAPS["sst"],
              vmin=vmin, vmax=vmax, aspect="auto",
              interpolation="nearest", zorder=2)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    _save(fig, name)


def save_mask_tile(name, cat, colors, extent):
    """Bare categorical mask tile (integer codes -> colors). No decoration."""
    fig = plt.figure(figsize=FIGSIZE)
    ax = _bare_ax(fig)
    ax.imshow(cat, extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap(colors),
              vmin=0, vmax=len(colors) - 1, aspect="auto",
              interpolation="nearest", zorder=1)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    _save(fig, name)


def main():
    apply_paper_style()
    R = reproduce_pipeline()
    lat, lon = R["lat"], R["lon"]
    land = R["land"]
    stage3 = R["stage3"]              # (T,H,W) completed proxy
    miss_s1 = R["miss_after_s1"]      # (T,H,W) still-missing after Stage 1
    tfill = R["tfill"]                # (T,H,W) Stage-1 temporal-filled

    # flip lat ascending for origin='lower' (match fig2_v2_proxy.py)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        flip = lambda a: a[..., ::-1, :]
        land = flip(land)
        stage3 = flip(stage3); miss_s1 = flip(miss_s1); tfill = flip(tfill)
    ocean = ~land
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    ti = stage3.shape[0] - 1          # target frame = last
    # shared SST scale from the target proxy (same convention as fig2_v2_proxy)
    s3_c_target = k_to_c(stage3[ti])
    vmin = float(np.percentile(s3_c_target[ocean], 1))
    vmax = float(np.percentile(s3_c_target[ocean], 99))

    # ---- 7 completed historical proxy tiles (t-1 .. t-7 same UTC hour) ----
    for d in range(1, N_HIST + 1):
        idx = ti - d
        if idx < 0:
            print(f"WARN: not enough warmup frames for t-{d}"); continue
        save_sst_tile(f"proxy_tm{d}", k_to_c(stage3[idx]), land, extent,
                      vmin, vmax)

    # ---- target-time observation mask (Observed / Missing / Land) ----
    # Observed at target = neither temporal- nor KNN-filled ocean pixel.
    knn_t = miss_s1[ti] & ocean
    filled_t = (tfill[ti] & ocean) | knn_t
    obs_cat = np.full(land.shape, 0, np.uint8)   # 0 observed
    obs_cat[filled_t] = 1                         # 1 missing (to be reconstructed)
    obs_cat[land] = 2                             # 2 land
    save_mask_tile("obs_mask", obs_cat,
                   [CAT_OBSERVED, CAT_MISSING, CAT_LAND], extent)

    # ---- F_temp: Stage-1 temporal-filled pixels only ----
    ftemp_cat = np.full(land.shape, 0, np.uint8)  # 0 background ocean
    ftemp_cat[tfill[ti] & ocean] = 1              # 1 temporal-filled
    ftemp_cat[land] = 2
    save_mask_tile("ftemp_mask", ftemp_cat,
                   [CAT_OBSERVED, MISSING_OVERLAY, CAT_LAND], extent)

    # ---- F_KNN: Stage-3 KNN-filled pixels only ----
    fknn_cat = np.full(land.shape, 0, np.uint8)
    fknn_cat[knn_t] = 1                            # 1 knn-filled
    fknn_cat[land] = 2
    save_mask_tile("fknn_mask", fknn_cat,
                   [CAT_OBSERVED, MISSING_OVERLAY, CAT_LAND], extent)

    # provenance / diagnostics
    obs_r = 100 * (obs_cat == 0).sum() / ocean.sum()
    print(f"target {TARGET.isoformat()}  ocean px={int(ocean.sum())}")
    print(f"  observed={obs_r:.1f}%  temporal-filled="
          f"{100*(tfill[ti]&ocean).sum()/ocean.sum():.1f}%  "
          f"knn-filled={100*knn_t.sum()/ocean.sum():.1f}%")
    print(f"  SST scale [{vmin:.2f}, {vmax:.2f}] degC, tile {FIGSIZE} @ {DPI}dpi")
    print(f"saved tiles -> {OUT_DIR}")


if __name__ == "__main__":
    main()
