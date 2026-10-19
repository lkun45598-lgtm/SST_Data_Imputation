"""Figure 2 (redesign): proxy construction from gappy hourly SST + matched mask.

Faithful to the CURRENT (fixed) deployed pipeline (hourly_preprocess.py +
knn_fill_3d.py), which preserves real observations across every stage:
  Stage 1  same-UTC-hour temporal-weighted fill, D=3 days, w_d = 1/d
  Stage 2  mask-preserving Gaussian low-pass filter, sigma=1.5 (fill region only)
  Stage 3  3D progressive causal KNN fill (k=20, 7-day same-hour window)

All three stages are recomputed on-the-fly by calling the deployed pipeline
functions on a WARMUP-day same-UTC-hour sub-stack ending at the target frame.
Because the Stage-3 causal window is only 7 days, the target frame's proxy is
identical to the full-series run.  We deliberately do NOT read the stale
h23/*.h5 (built 2026-04, before the 2026-07-12 observation-preservation fix),
whose stored "observations" were smoothed by up to ~10 degC.

Target frame: 2018-03-09 23:00 (series_00), a high-cloud day.

Outputs (into output/fig2_parts/):
  individual large panels  (hist_t3/t2/t1, target, stage1/2/3, category)
  a composite mockup       fig2_v2_composite.png/pdf
so the layout can be chosen before finalizing.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch, FancyArrowPatch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
PRE = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation/preprocessing")
sys.path.insert(0, str(PRE))
import hourly_preprocess as hp
from knn_fill_3d import progressive_knn_fill_3d
from style import (apply_paper_style, CMAPS, setup_geo_ax,
                   LAND_COLOR, MISSING_OVERLAY,
                   CAT_OBSERVED, CAT_TEMPORAL, CAT_KNN, CAT_MISSING, CAT_LAND)

OUT_DIR = Path(__file__).parent / "output" / "fig2_parts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JAXA_ROOT = Path("/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3")
TARGET = datetime(2018, 3, 9, 23, 0, 0)
HOUR = 23
WARMUP_DAYS = 40      # >> 7-day causal window => target-frame proxy is exact
D_DAYS = 3
SIGMA = 1.5


def k_to_c(a):
    return a - 273.15


def load_raw(t):
    f = JAXA_ROOT / t.strftime("%Y%m") / t.strftime("%d") / f"{t.strftime('%Y%m%d%H%M%S')}.nc"
    if not f.exists():
        return None
    with nc.Dataset(f) as ds:
        s = ds.variables["sea_surface_temperature"][:]
        s = s[0] if s.ndim == 3 else s
        if hasattr(s, "mask"):
            s = np.where(s.mask, np.nan, s.filled(np.nan))
        return np.array(s, dtype=np.float32)


def reproduce_pipeline():
    """Run the FIXED deployed pipeline on a WARMUP-day same-UTC-hour sub-stack
    ending at TARGET. Returns the three stage fields + masks for the last frame.
    Observations are preserved by construction (Stage 2/3 only touch fill pixels)."""
    start_time = (TARGET - timedelta(days=WARMUP_DAYS)).replace(hour=HOUR)

    sst_l, miss_l, fill_l = [], [], []
    lat = lon = None
    for day in range(WARMUP_DAYS + 1):
        tt = start_time + timedelta(days=day)
        filled, _orig, miss_before, _ = hp.temporal_weighted_fill_frame(tt, start_time, HOUR)
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

    ti = len(sst) - 1   # target frame = last
    return dict(lat=lat, lon=lon, land=land.astype(bool),
                stage1=sst[ti], stage2=filt[ti], stage3=filled3d[ti],
                miss_after_s1=miss[ti].astype(bool),
                tfill=fillm[ti].astype(bool))


def main():
    apply_paper_style()

    R = reproduce_pipeline()
    lat, lon = R["lat"], R["lon"]
    land = R["land"]
    stage1, stage2, stage3 = R["stage1"], R["stage2"], R["stage3"]
    tfill_dep = R["tfill"]              # temporal-filled pixels (Stage 1)
    miss_s1 = R["miss_after_s1"]        # still missing after Stage 1 (=> KNN-filled)
    ts = TARGET.isoformat()

    raw = load_raw(TARGET)
    hist = []
    for d in range(1, D_DAYS + 1):
        hs = load_raw(TARGET - timedelta(days=d))
        if hs is not None:
            hist.append((d, hs))

    # flip lat ascending for origin='lower'
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        flip = lambda a: a[::-1, :]
        land = flip(land); raw = flip(raw)
        stage1 = flip(stage1); stage2 = flip(stage2); stage3 = flip(stage3)
        tfill_dep = flip(tfill_dep); miss_s1 = flip(miss_s1)
        hist = [(d, flip(hs)) for d, hs in hist]

    ocean = ~land

    # sanity: observations preserved through all stages
    obs = (~np.isnan(raw)) & ocean
    dmax = float(np.nanmax(np.abs(stage3[obs] - raw[obs]))) if obs.any() else -1
    print("obs preserved: max|stage3-raw| on observed = %.5f K" % dmax)

    # category map (observed / temporal-filled / knn-filled / missing / land)
    knn_filled = miss_s1 & ocean
    cat = np.full(land.shape, 0, dtype=np.uint8)   # 0 observed
    cat[tfill_dep & ocean] = 1                     # temporal
    cat[knn_filled] = 2                            # knn
    cat[land] = 4                                  # land

    np.savez(OUT_DIR / "fig2_v2_data.npz",
             lat=lat, lon=lon, land=land, ocean=ocean,
             raw=raw, stage1=stage1, stage2=stage2, stage3=stage3,
             hist_t1=hist[0][1], hist_t2=hist[1][1], hist_t3=hist[2][1],
             cat=cat, ts=ts)

    print("Data prepared. ts=", ts)
    print("missing raw=%.1f%%  s1=%.1f%%  s2=%.1f%%  s3=%.1f%%" % (
        100*np.isnan(raw)[ocean].mean(), 100*np.isnan(stage1)[ocean].mean(),
        100*np.isnan(stage2)[ocean].mean(), 100*np.isnan(stage3)[ocean].mean()))
    render(lat, lon, land, ocean, raw, stage1, stage2, stage3, hist, cat, ts)


# ----------------------------------------------------------------------
# rendering helpers
# ----------------------------------------------------------------------
CAT_COLORS = [CAT_OBSERVED, CAT_TEMPORAL, CAT_KNN, CAT_MISSING, CAT_LAND]
CAT_LABELS = ["Observed", "Temporal-filled", "KNN-filled", "Missing", "Land"]


def _draw_sst(ax, field_c, land, lon, lat, vmin, vmax, title=None,
              show_y=True, show_x=True, miss_overlay=True):
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(np.where(land, 1.0, np.nan), extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap([LAND_COLOR]),
              aspect="equal", interpolation="nearest")
    if miss_overlay:
        miss = np.isnan(field_c) & (~land)
        ax.imshow(np.where(miss, 1.0, np.nan), extent=extent, origin="lower",
                  cmap=mpl.colors.ListedColormap([MISSING_OVERLAY]),
                  aspect="equal", interpolation="nearest")
    disp = np.where(land, np.nan, field_c)
    im = ax.imshow(disp, extent=extent, origin="lower", cmap=CMAPS["sst"],
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    setup_geo_ax(ax, lon, lat, draw_xlabel=show_x, draw_ylabel=show_y, step=2)
    if title:
        ax.set_title(title, fontsize=13, pad=6)
    return im


def render(lat, lon, land, ocean, raw, s1, s2, s3, hist, cat, ts):
    raw_c, s1_c, s2_c, s3_c = map(k_to_c, (raw, s1, s2, s3))
    h_c = [k_to_c(hs) for _, hs in hist]           # [t-1, t-2, t-3]
    vmin = float(np.percentile(s3_c[ocean], 1))
    vmax = float(np.percentile(s3_c[ocean], 99))

    def mr(f):
        return 100 * (np.isnan(f) & ocean).sum() / ocean.sum()

    # ---------- individual panels ----------
    panels = [
        ("hist_t3", h_c[2], "Same-hour history $t\\!-\\!3$d", "$w_3=1/3$"),
        ("hist_t2", h_c[1], "Same-hour history $t\\!-\\!2$d", "$w_2=1/2$"),
        ("hist_t1", h_c[0], "Same-hour history $t\\!-\\!1$d", "$w_1=1$"),
        ("target",  raw_c,  "Target $t$ (raw, gappy)", "missing %.1f%%" % mr(raw)),
        ("stage1",  s1_c,   "Stage 1: temporal fill", "missing %.1f%%" % mr(s1)),
        ("stage2",  s2_c,   "Stage 2: Gaussian ($\\sigma$=1.5)", "missing %.1f%%" % mr(s2)),
        ("stage3",  s3_c,   "Stage 3: 3D KNN fill (proxy)", "missing %.1f%%" % mr(s3)),
    ]
    for name, fld, title, note in panels:
        fig, ax = plt.subplots(figsize=(5.2, 5.6))
        im = _draw_sst(ax, fld, land, lon, lat, vmin, vmax, title=title)
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=11,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("SST (°C)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # category panel
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(cat, extent=extent, origin="lower",
              cmap=mpl.colors.ListedColormap(CAT_COLORS),
              vmin=0, vmax=4, aspect="equal", interpolation="nearest")
    setup_geo_ax(ax, lon, lat, step=2)
    ax.set_title("Pixel provenance (matched mask)", fontsize=13, pad=6)
    ax.legend(handles=[Patch(fc=c, ec="0.5", label=l)
                       for c, l in zip(CAT_COLORS, CAT_LABELS)],
              loc="lower left", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "category.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "category.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- composite mockup ----------
    _composite(lat, lon, land, ocean, raw_c, s1_c, s2_c, s3_c, h_c, cat,
               vmin, vmax, mr, raw, s1, s2, s3, ts)


def _composite(lat, lon, land, ocean, raw_c, s1_c, s2_c, s3_c, h_c, cat,
               vmin, vmax, mr, raw, s1, s2, s3, ts):
    fig = plt.figure(figsize=(20, 10.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1],
                          height_ratios=[1, 1], hspace=0.22, wspace=0.12,
                          left=0.05, right=0.94, top=0.90, bottom=0.07)
    fig.suptitle("Proxy construction from gappy hourly SST — target %s" % ts,
                 fontsize=16, fontweight="bold")

    # Top row: temporal support -> target
    top = [
        (h_c[2], "(a) $t\\!-\\!3$d  ($w{=}1/3$)"),
        (h_c[1], "(b) $t\\!-\\!2$d  ($w{=}1/2$)"),
        (h_c[0], "(c) $t\\!-\\!1$d  ($w{=}1$)"),
        (raw_c,  "(d) Target $t$ (raw)  miss %.1f%%" % mr(raw)),
    ]
    for i, (fld, title) in enumerate(top):
        ax = fig.add_subplot(gs[0, i])
        _draw_sst(ax, fld, land, lon, lat, vmin, vmax, title=title,
                  show_y=(i == 0), show_x=False)
    # annotate top row role
    axr = fig.add_subplot(gs[0, 4]); axr.axis("off")
    axr.text(0.0, 0.5,
             "Same-UTC-hour\ncausal support\n($D{=}3$ days)\n\nfeeds\nStage 1\ntemporal fill",
             fontsize=12, va="center", ha="left")

    # Bottom row: pipeline stages -> proxy, + category
    bot = [
        (raw_c, "(e) Raw (gappy)  %.1f%%" % mr(raw), True),
        (s1_c,  "(f) Stage 1 temporal  %.1f%%" % mr(s1), True),
        (s2_c,  "(g) Stage 2 Gaussian  %.1f%%" % mr(s2), True),
        (s3_c,  "(h) Stage 3 3D KNN = proxy  %.1f%%" % mr(s3), True),
    ]
    for i, (fld, title, mo) in enumerate(bot):
        ax = fig.add_subplot(gs[1, i])
        im = _draw_sst(ax, fld, land, lon, lat, vmin, vmax, title=title,
                       show_y=(i == 0), show_x=True, miss_overlay=mo)
        if i == 3:
            _last = im

    # category panel bottom-right
    axc = fig.add_subplot(gs[1, 4])
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    axc.imshow(cat, extent=extent, origin="lower",
               cmap=mpl.colors.ListedColormap(CAT_COLORS),
               vmin=0, vmax=4, aspect="equal", interpolation="nearest")
    setup_geo_ax(axc, lon, lat, draw_xlabel=True, draw_ylabel=False, step=2)
    axc.set_title("(i) Matched mask (provenance)", fontsize=13, pad=6)
    axc.legend(handles=[Patch(fc=c, ec="0.5", label=l)
                        for c, l in zip(CAT_COLORS, CAT_LABELS)],
               loc="lower left", fontsize=8, framealpha=0.92)

    fig.savefig(OUT_DIR.parent / "fig2_v2_composite.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR.parent / "fig2_v2_composite.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Composite + parts saved to", OUT_DIR)


if __name__ == "__main__":
    main()
