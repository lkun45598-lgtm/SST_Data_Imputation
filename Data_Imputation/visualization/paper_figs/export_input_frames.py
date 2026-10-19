"""Export raw single-panel PNGs for a hand-composed methodology figure.

The user assembles the final figure themselves; this script only dumps the
individual tiles, all sharing ONE SST colour scale so they stay coherent:

  output/input_panels/
    sst/sst_tNN.png       30-frame input window  (KNN-filled sst_data, ocean SST)
    mask/mask_tNN.png     matching mask_seq per frame (= original_missing_mask)
    day30_obs_sparse.png  day-30 raw observations only (non-obs = grey)
    day30_input_meanfill.png  actual model input @ t=30 (obs kept, non-obs -> norm_mean)
    day30_mask.png        day-30 reconstruction mask (non-obs & ocean = 1)
    jaxa_fill_h12.png      h12-realmask fill composed on the ORIGINAL JAXA frame
    jaxa_fill_modelonly.png pure model prediction everywhere (pre-composite)
    _colorbar_sst.png / _legend_mask.png   reference strips
    _meta.txt             idx / timestamp / vmin / vmax / coverage

Convention B (train-consistent):  day-30 input keeps real obs, fills non-obs
with norm_mean; mask[-1] marks the non-obs region; output = model at non-obs,
real obs preserved, Gaussian(sigma) applied only on the filled region.

Usage: python3 visualization/paper_figs/export_input_frames.py --idx 99 --gpu 7
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATA_IMPUTATION_DIR))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from visualization.paper_figs.style import (
    apply_paper_style, CMAPS, LAND_COLOR, CAT_OBSERVED, CAT_MISSING,
)

KNN_FILLED_DIR = DATA_IMPUTATION_DIR / "experiments/hourly_data/h12"
MODEL_PATH = DATA_IMPUTATION_DIR / "experiments/jaxa_finetune_h12_realmask_clean/best_model.pth"  # 新模型(正确OSTIA权重)
OUT_DIR = Path(__file__).parent / "output" / "input_panels"
WINDOW_SIZE = 30
GAUSSIAN_SIGMA = 1.0


def load_model(device):
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6,
        cbam_reduction_ratio=16,
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = float(ckpt.get("norm_mean", 299.9221))
    std = float(ckpt.get("norm_std", 2.6919))
    print(f"model epoch={ckpt.get('epoch', 'N/A')} mean={mean:.3f} std={std:.3f}")
    return model, mean, std


def gauss_fill_region(sst, land, sigma, fill_region):
    """Gaussian input uses real obs; write back only in fill_region, obs kept."""
    valid = ~np.isnan(sst) & (land == 0)
    if valid.sum() == 0:
        return sst
    mean_v = np.nanmean(sst)
    fill = sst.copy()
    fill[~valid] = mean_v
    out = gaussian_filter(fill, sigma=sigma)
    write = valid & (fill_region == 1)
    return np.where(write, out, np.where(valid, sst, np.nan))


def save_sst(arr, land, vmin, vmax, out_path):
    """One clean SST tile: ocean = RdYlBu_r, land = tan, NaN-ocean = slate."""
    H, W = arr.shape
    m = np.ma.masked_where((land == 1) | np.isnan(arr), arr)
    cmap = CMAPS["sst"].copy()
    cmap.set_bad(LAND_COLOR)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_sst_with_missing(arr, land, missing, vmin, vmax, out_path):
    """SST tile where `missing`==1 pixels are painted slate (non-obs)."""
    H, W = arr.shape
    disp = arr.copy().astype(np.float32)
    disp[missing == 1] = np.nan
    m = np.ma.masked_where((land == 1) | np.isnan(disp), disp)
    cmap = CMAPS["sst"].copy()
    cmap.set_bad(CAT_MISSING)          # non-obs -> slate
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    # land drawn on top in tan
    land_layer = np.ma.masked_where(land == 0, np.ones_like(arr))
    ax.imshow(land_layer, cmap=mpl.colors.ListedColormap([LAND_COLOR]),
              interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_mask(missing, land, out_path):
    """Mask tile: valid ocean = cream, missing = slate, land = tan."""
    H, W = missing.shape
    cat = np.zeros((H, W), dtype=np.int16)          # 0 = valid ocean
    cat[(missing == 1) & (land == 0)] = 1           # 1 = missing ocean
    cat[land == 1] = 2                              # 2 = land
    cmap = mpl.colors.ListedColormap([CAT_OBSERVED, CAT_MISSING, LAND_COLOR])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(cat, cmap=cmap, norm=norm, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_colorbar(vmin, vmax, out_path):
    fig, ax = plt.subplots(figsize=(1.1, 3.4))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=CMAPS["sst"], norm=norm,
                                   orientation="vertical")
    cb.set_label("SST (K)", fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_mask_legend(out_path):
    fig, ax = plt.subplots(figsize=(3.2, 0.9))
    ax.axis("off")
    handles = [
        mpl.patches.Patch(facecolor=CAT_OBSERVED, edgecolor="#999", label="valid / filled"),
        mpl.patches.Patch(facecolor=CAT_MISSING, label="missing (mask=1)"),
        mpl.patches.Patch(facecolor=LAND_COLOR, label="land"),
    ]
    ax.legend(handles=handles, loc="center", ncol=1, frameon=False, fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=8)
    ap.add_argument("--idx", type=int, default=99, help="day-30 (current) frame index")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--nframes", type=int, default=30, help="trailing frames to export")
    args = ap.parse_args()

    apply_paper_style()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model, norm_mean, norm_std = load_model(device)

    path = KNN_FILLED_DIR / f"jaxa_knn_filled_{args.series:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:].astype(np.float32)
        obs_all = f["original_obs_mask"][:].astype(np.uint8)
        miss_all = f["original_missing_mask"][:].astype(np.uint8)
        land = f["land_mask"][:].astype(np.uint8)
        ts = f["timestamps"][:]
        ts = [t.decode() if isinstance(t, bytes) else str(t) for t in ts]
    T, H, W = sst_all.shape
    ocean = (land == 0)
    idx = args.idx
    assert idx >= WINDOW_SIZE - 1, "idx must be >= 29"

    # --- build the 30-frame window ---
    sst_seq = np.stack([sst_all[max(0, idx - (WINDOW_SIZE - 1) + t)] for t in range(WINDOW_SIZE)])
    miss_seq = np.stack([miss_all[max(0, idx - (WINDOW_SIZE - 1) + t)] for t in range(WINDOW_SIZE)])

    # shared SST scale from the window's ocean pixels
    ocean_vals = sst_seq[:, ocean]
    vmin, vmax = np.percentile(ocean_vals, [2, 98])
    vmin, vmax = float(vmin), float(vmax)
    print(f"idx={idx} ts={ts[idx]} vmin={vmin:.3f} vmax={vmax:.3f}")
    obs30 = obs_all[idx]
    non_obs = (obs30 == 0) & ocean
    cov = obs30[ocean].mean()
    print(f"day-30 obs coverage on ocean = {cov:.3f}  (cloud/non-obs = {1 - cov:.3f})")

    # --- directories ---
    sst_dir = OUT_DIR / "sst"; mask_dir = OUT_DIR / "mask"
    sst_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # --- per-frame tiles (last nframes of the window) ---
    t0 = max(0, WINDOW_SIZE - args.nframes)
    for t in range(t0, WINDOW_SIZE):
        save_sst(sst_seq[t], land, vmin, vmax, sst_dir / f"sst_t{t:02d}.png")
        save_mask(miss_seq[t], land, mask_dir / f"mask_t{t:02d}.png")
    print(f"wrote {WINDOW_SIZE - t0} sst + mask tiles")

    # --- day-30 special panels ---
    # (a) sparse raw observations only
    save_sst_with_missing(sst_seq[-1], land, non_obs.astype(np.uint8), vmin, vmax,
                          OUT_DIR / "day30_obs_sparse.png")
    # (b) actual model input @ t=30: obs kept, non-obs -> norm_mean placeholder
    day30_input = np.where(non_obs, norm_mean, sst_seq[-1])
    save_sst(day30_input, land, vmin, vmax, OUT_DIR / "day30_input_meanfill.png")
    # (c) day-30 reconstruction mask (non-obs & ocean)
    save_mask(non_obs.astype(np.uint8), land, OUT_DIR / "day30_mask.png")

    # --- run model (convention B) on the ORIGINAL JAXA frame ---
    sst_input = sst_seq.copy()
    sst_input[-1] = np.where(non_obs, norm_mean, sst_input[-1])
    mask_in = miss_seq.copy().astype(np.float32)
    mask_in[-1] = non_obs.astype(np.float32)
    sst_norm = np.nan_to_num((sst_input - norm_mean) / norm_std, nan=0.0)
    st = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mt = torch.from_numpy(mask_in).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(st, mt)
    pred_k = pred.squeeze().cpu().numpy() * norm_std + norm_mean

    save_sst(np.where(ocean, pred_k, np.nan), land, vmin, vmax,
             OUT_DIR / "jaxa_fill_modelonly.png")

    composed = np.where(non_obs, pred_k, sst_seq[-1])
    composed = np.where(land == 1, np.nan, composed)
    composed = gauss_fill_region(composed, land, GAUSSIAN_SIGMA, non_obs.astype(np.uint8))
    save_sst(composed, land, vmin, vmax, OUT_DIR / "jaxa_fill_h12.png")

    # --- references + meta ---
    save_colorbar(vmin, vmax, OUT_DIR / "_colorbar_sst.png")
    save_mask_legend(OUT_DIR / "_legend_mask.png")
    with open(OUT_DIR / "_meta.txt", "w") as f:
        f.write(f"series={args.series} idx={idx} timestamp={ts[idx]}\n")
        f.write(f"window=[{idx - WINDOW_SIZE + 1}..{idx}] nframes_exported={WINDOW_SIZE - t0}\n")
        f.write(f"vmin={vmin:.4f} vmax={vmax:.4f} (K, shared across all SST tiles)\n")
        f.write(f"day30 obs coverage(ocean)={cov:.4f} cloud={1 - cov:.4f}\n")
        f.write(f"norm_mean={norm_mean:.4f} norm_std={norm_std:.4f}\n")
        f.write("mask tiles = original_missing_mask (= mask_seq fed to model); "
                "day30_mask = non-obs&ocean (reconstruction target region)\n")
    print(f"\nDONE -> {OUT_DIR}")


if __name__ == "__main__":
    main()
