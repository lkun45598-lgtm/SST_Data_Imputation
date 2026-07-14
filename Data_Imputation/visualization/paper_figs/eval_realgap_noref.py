#!/usr/bin/env python3
"""No-reference comparison of fill quality on the REAL cloud gaps of the data.

There is NO pointwise ground truth under real clouds, so we rank methods by
PHYSICAL REALISM of their fill of the actual cloud region:
  (C1) radial power spectrum vs clear-sky climatology  (structure realism)
  (C2) temporal consistency: mean |field[t]-field[t-1]| at FILLED pixels
       normalized by the same at OBSERVED pixels (1.0 = fills move like real
       obs; >>1 = single-frame method injects frame-to-frame jitter)
  (C3) seam gradient at the obs<->fill boundary        (discontinuity)

Baselines (linear/cubic/knn) run on CPU. Model fills (old square-trained FNO,
new real-cloud pilot) are added with --with-models (needs a free GPU).
Outputs: cache/realgap_noref.npz + eval_diag figures.
"""
import sys, argparse
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation

DI = Path("/data1/user/lz/SST_Data_Imputation/Data_Imputation")
for p in ["", "inference", "baselines", "visualization/paper_figs"]:
    sys.path.insert(0, str(DI / p) if p else str(DI))
from simple_interp import linear_interp_2d, cubic_interp_2d

HR = DI / "experiments/hourly_data/h12"
OUT = DI / "eval_diag"; OUT.mkdir(exist_ok=True)
CACHE = DI / "visualization/paper_figs/cache"; CACHE.mkdir(exist_ok=True)
TILE = (112, 96, 192)   # y0, x0, size  (all-ocean square)
OLD_MODEL = DI / "experiments/jaxa_finetune_h12/best_model.pth"          # square-trained
NEW_MODEL = DI / "experiments/jaxa_finetune_h12_realmask/best_model.pth"  # real-cloud pilot


def radial_psd(tile, hann):
    S = tile.shape[0]
    t = np.nan_to_num(tile - np.nanmean(tile)) * hann
    P = np.abs(np.fft.fftshift(np.fft.fft2(t))) ** 2
    yy, xx = np.indices((S, S)); r = np.sqrt((yy - S // 2) ** 2 + (xx - S // 2) ** 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def knn_fill(field, cloud, ocean, k=20, power=2.0):
    from scipy.spatial import cKDTree
    src = (cloud == 0) & (ocean == 1) & ~np.isnan(field)
    sy, sx = np.where(src)
    if len(sy) == 0: return field.copy()
    tree = cKDTree(np.column_stack([sy, sx])); vals = field[sy, sx]
    ty, tx = np.where(cloud == 1)
    if len(ty) == 0: return field.copy()
    d, i = tree.query(np.column_stack([ty, tx]), k=min(k, len(sy)))
    if d.ndim == 1: d, i = d[:, None], i[:, None]
    w = 1.0 / (d ** power + 1e-10)
    out = field.copy(); out[ty, tx] = (vals[i] * w).sum(1) / w.sum(1)
    return out


def gauss_keep_obs(field, land, cloud, sigma=1.0):
    valid = ~np.isnan(field) & (land == 0)
    f = field.copy(); f[~valid] = np.nanmean(field)
    o = gaussian_filter(f, sigma=sigma)
    return np.where(valid & (cloud == 1), o, np.where(valid, field, np.nan))


def fill_baselines(sst_seq, obs_t, land):
    """Fill the REAL cloud (obs_t==0 & ocean) of day-30 from REAL observations only."""
    ocean = (land == 0).astype(np.uint8)
    cloud = ((obs_t == 0) & (ocean == 1)).astype(np.uint8)
    day30 = sst_seq[-1].copy()
    obs_field = np.where(obs_t > 0, day30, np.nan)     # keep ONLY real obs as source
    src = (obs_t > 0) & (ocean == 1) & ~np.isnan(day30)
    out = {}
    for nm, fn in (("linear", linear_interp_2d), ("cubic", cubic_interp_2d)):
        f = fn(np.where(src, day30, np.nan), src.astype(np.float32), ocean)
        c = np.where(obs_t > 0, day30, f)
        out[nm] = gauss_keep_obs(np.where(ocean == 1, c, np.nan), land, cloud)
    out["knn"] = gauss_keep_obs(knn_fill(obs_field, cloud, ocean), land, cloud)
    return out, cloud


def load_fno(ckpt_path, device):
    import torch
    from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64,
                              width=64, depth=6, cbam_reduction_ratio=16).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, float(ck.get("norm_mean", 299.9221)), float(ck.get("norm_std", 2.6919))


def fill_model(model, sst_seq, miss_seq, cloud, land, nm, ns, device):
    """Fill ONLY the real cloud region; real obs pixels are never overwritten."""
    import torch
    art = cloud.astype(np.float32)
    si = sst_seq.copy(); si[-1] = np.where(art > 0, nm, si[-1])
    ms = miss_seq.copy().astype(np.float32); ms[-1] = art
    sn = np.nan_to_num((si - nm) / ns, nan=0.0)
    st = torch.from_numpy(sn).unsqueeze(0).float().to(device)
    mt = torch.from_numpy(ms).unsqueeze(0).to(device)
    with torch.no_grad():
        pr = model(st, mt).squeeze().cpu().numpy() * ns + nm
    out = np.where(art > 0, pr, sst_seq[-1])                 # obs preserved
    return gauss_keep_obs(np.where(land == 0, out, np.nan), land, cloud)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=8)
    ap.add_argument("--n-cloudy", type=int, default=25)
    ap.add_argument("--with-models", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    with h5py.File(HR / f"jaxa_knn_filled_{args.series:02d}.h5", "r") as f:
        sst = f["sst_data"][:]; obs = f["original_obs_mask"][:]
        miss = f["original_missing_mask"][:]; land = f["land_mask"][:]
    ocean = land == 0; T = sst.shape[0]; WIN = 30
    y0, x0, S = TILE; hann = np.outer(np.hanning(S), np.hanning(S))
    tile_cov = obs[:, y0:y0 + S, x0:x0 + S].reshape(T, -1).mean(1)

    # clear-sky climatology reference spectrum
    clear = [i for i in range(T) if tile_cov[i] > 0.95]
    ref_psd = np.mean([radial_psd(sst[i, y0:y0 + S, x0:x0 + S], hann) for i in clear[:30]], 0)

    # real cloudy frames: genuine gap present in the tile (need idx-1 for temporal)
    cloudy = [i for i in range(WIN, T) if 0.3 < tile_cov[i] < 0.85][:args.n_cloudy]
    print(f"clear ref frames={len(clear)}  cloudy test frames={len(cloudy)}")

    methods = ["linear", "cubic", "knn"]
    models = {}
    if args.with_models:
        import torch
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"loading models on {device}")
        if OLD_MODEL.exists():
            models["FNO_square"] = load_fno(OLD_MODEL, device)
        if NEW_MODEL.exists():
            models["FNO_realmask"] = load_fno(NEW_MODEL, device)
        methods = methods + list(models.keys())
        print(f"models: {list(models.keys())}")

    psd_acc = {m: [] for m in methods}
    seam_acc = {m: [] for m in methods}
    tcons_num = {m: [] for m in methods}   # mean|Δt| at filled pixels
    tcons_den = []                          # mean|Δt| at observed pixels (shared ref)
    hik = slice(S // 8, S // 2)

    def build_win(idx):
        ss = np.zeros((WIN, *sst.shape[1:]), np.float32)
        ms = np.zeros_like(ss)
        for t in range(WIN):
            s = max(0, idx - (WIN - 1) + t); ss[t] = sst[s]; ms[t] = miss[s]
        return ss, ms

    for idx in cloudy:
        ss, ms = build_win(idx)
        ss_p, ms_p = build_win(idx - 1)
        fills, cloud = fill_baselines(ss, obs[idx], land)
        fills_p, cloud_p = fill_baselines(ss_p, obs[idx - 1], land)
        if args.with_models:
            device = next(models[list(models)[0]][0].parameters()).device
            for name, (mdl, nm, ns) in models.items():
                fills[name] = fill_model(mdl, ss, ms, cloud, land, nm, ns, device)
                fills_p[name] = fill_model(mdl, ss_p, ms_p, cloud_p, land, nm, ns, device)
        ring = binary_dilation(cloud.astype(bool), iterations=1) & (~cloud.astype(bool)) & ocean
        # temporal reference: pixels observed in BOTH frames
        both_obs = (obs[idx] > 0) & (obs[idx - 1] > 0) & ocean
        # temporal test: pixels cloud-filled in BOTH frames
        both_fill = (cloud.astype(bool)) & (cloud_p.astype(bool)) & ocean
        for m in methods:
            fl, fp = fills[m], fills_p[m]
            psd_acc[m].append(radial_psd(fl[y0:y0 + S, x0:x0 + S], hann))
            gy, gx = np.gradient(np.nan_to_num(fl))
            seam_acc[m].append(float(np.nanmean(np.sqrt(gy ** 2 + gx ** 2)[ring])) * 1000)
            if both_fill.sum() > 20:
                dt = np.abs(fl - fp)
                tcons_num[m].append(float(np.nanmean(dt[both_fill])))
        if both_obs.sum() > 20:
            day30, day29 = ss[-1], ss_p[-1]
            tcons_den.append(float(np.nanmean(np.abs(day30 - day29)[both_obs])))

    obs_jit = np.mean(tcons_den) if tcons_den else np.nan
    print(f"\nobserved-pixel frame-to-frame |Δt| (temporal ref) = {obs_jit*1000:.1f} mK")
    print(f"\n{'method':<13}{'high-k PSD ratio':>16}{'seam(mK/px)':>13}{'temporal ratio':>16}")
    curves = {"clear_ref": ref_psd}; table = {}
    for m in methods:
        pm = np.mean(psd_acc[m], 0); curves[m] = pm
        psd_r = pm[hik].sum() / ref_psd[hik].sum()
        seam = np.mean(seam_acc[m])
        tr = (np.mean(tcons_num[m]) / obs_jit) if tcons_num[m] and obs_jit else np.nan
        table[m] = dict(psd=float(psd_r), seam=float(seam), tratio=float(tr))
        print(f"{m:<13}{psd_r:>16.3f}{seam:>13.1f}{tr:>16.2f}")

    np.savez_compressed(CACHE / "realgap_noref.npz",
                        curves={k: v for k, v in curves.items()},
                        methods=methods, ref_psd=ref_psd, tile=TILE,
                        table=table, obs_jit=obs_jit)
    # PSD figure
    k = np.arange(len(ref_psd))
    plt.figure(figsize=(9, 6))
    plt.loglog(k[1:S // 2], ref_psd[1:S // 2], "k-", lw=2.5, label="clear-sky climatology (ref)")
    sty = {"linear": ("C1", "--"), "cubic": ("C3", ":"), "knn": ("C2", "-."),
           "FNO_square": ("C0", "-"), "FNO_realmask": ("C4", "-")}
    for m in methods:
        c, l = sty.get(m, ("C7", "-")); lw = 2.2 if m.startswith("FNO") else 1.6
        plt.loglog(k[1:S // 2], curves[m][1:S // 2], color=c, ls=l, lw=lw, label=m)
    plt.xlabel("wavenumber k"); plt.ylabel("radial PSD")
    plt.title("Real-cloud-gap fill: structure realism (closest to black = best)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    tag = "models" if args.with_models else "baselines"
    plt.savefig(OUT / f"06_realgap_PSD_{tag}.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT}/06_realgap_PSD_{tag}.png")


if __name__ == "__main__":
    main()
