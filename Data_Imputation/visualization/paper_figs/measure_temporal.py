"""Does the temporal-consistency loss (L_temp) actually help?

Compares cbam_boundary (no L_temp) vs cbam_full (with L_temp) on the quantity
L_temp is designed to improve: the day-to-day CHANGE at the same hour,
  temporal_err = mean | (Yhat_T - Yhat_{T-1}) - (Y_T - Y_{T-1}) |
evaluated on artificially-masked pixels that are observed on BOTH days.
Also reports per-frame MAE as a reference (which L_temp is NOT expected to move).
Reuses the exact data / masking / prediction pipeline from gen_eval_data.py.
"""
import sys
from pathlib import Path
import importlib.util
import numpy as np
import torch
import h5py
from tqdm import tqdm

THIS = Path(__file__).parent
sys.path.insert(0, str(THIS))
import gen_eval_data as G  # SquareMaskGenerator, predict_fno, gauss_filter, consts

DATA_IMPUTATION_DIR = G.DATA_IMPUTATION_DIR
_spec = importlib.util.spec_from_file_location(
    "fno_cbam_ablation",
    DATA_IMPUTATION_DIR / "ablation" / "models" / "fno_cbam_ablation.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
FNO_CBAM_Ablation = _m.FNO_CBAM_Ablation

VARIANTS = {
    "cbam_boundary": "ablation/experiments/cbam_boundary/best_model.pth",  # no L_temp
    "cbam_full":     "ablation/experiments/cbam_full/best_model.pth",      # with L_temp
}
MASK_RATIO = 0.50
N_PAIRS = 40
SEED = 7


def load_variant(rel, device):
    model = FNO_CBAM_Ablation(out_size=(451, 351), modes1=80, modes2=64,
                              width=64, depth=6, cbam_reduction_ratio=16,
                              use_cbam=True).to(device)
    ck = torch.load(DATA_IMPUTATION_DIR / rel, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck.get("norm_mean", 299.9221), ck.get("norm_std", 2.6919)


def window(sst_all, miss_all, idx):
    W = G.WINDOW_SIZE
    sst_seq = np.zeros((W, *sst_all.shape[1:]), dtype=np.float32)
    miss_seq = np.zeros_like(sst_seq)
    for t in range(W):
        s = max(0, idx - (W - 1) + t)
        sst_seq[t] = sst_all[s]
        miss_seq[t] = miss_all[s]
    return sst_seq, miss_seq


def predict_frame(model, sst_all, miss_all, idx, mask, mean, std, land, device):
    sst_seq, miss_seq = window(sst_all, miss_all, idx)
    pred = G.predict_fno(model, sst_seq, miss_seq, mask, mean, std, device)
    return G.gauss_filter(pred, land, G.GAUSSIAN_SIGMA, fill_region=mask)


def main():
    device = torch.device(f"cuda:{G.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    path = G.KNN_FILLED_DIR / f"jaxa_knn_filled_{G.SERIES_ID:02d}.h5"
    with h5py.File(path, "r") as f:
        sst_all = f["sst_data"][:]
        obs_all = f["original_obs_mask"][:]
        miss_all = f["original_missing_mask"][:]
        land = f["land_mask"][:]
    ocean = 1 - land
    T = sst_all.shape[0]

    models = {k: load_variant(v, device) for k, v in VARIANTS.items()}

    rng = np.random.default_rng(SEED)
    start = G.WINDOW_SIZE
    cand = [i for i in range(start, T)
            if obs_all[i].sum() > 5000 and obs_all[i - 1].sum() > 5000]
    pairs = rng.choice(cand, size=min(N_PAIRS, len(cand)), replace=False)
    print(f"consecutive-day pairs evaluated: {len(pairs)}")

    gen = G.SquareMaskGenerator(mask_ratio=MASK_RATIO, seed=123)
    res = {k: {"temporal_err": [], "frame_mae": []} for k in VARIANTS}

    for idx in tqdm(pairs, desc="pairs"):
        eligible = (obs_all[idx] * obs_all[idx - 1] * ocean).astype(np.float32)
        if eligible.sum() < 2000:
            continue
        mask = gen.generate(eligible)
        ev = (mask * eligible) > 0
        if ev.sum() == 0:
            continue
        dY_true = (sst_all[idx] - sst_all[idx - 1])[ev]
        for k, (model, mean, std) in models.items():
            pT = predict_frame(model, sst_all, miss_all, idx, mask, mean, std, land, device)
            pTm1 = predict_frame(model, sst_all, miss_all, idx - 1, mask, mean, std, land, device)
            dY_pred = (pT - pTm1)[ev]
            res[k]["temporal_err"].append(float(np.abs(dY_pred - dY_true).mean()))
            res[k]["frame_mae"].append(float(np.abs(pT[ev] - sst_all[idx][ev]).mean()))

    print("\n================  RESULT  ================")
    print(f"{'variant':<16}{'day-to-day change err':>24}{'frame MAE':>14}")
    for k in VARIANTS:
        te = np.array(res[k]["temporal_err"]); fm = np.array(res[k]["frame_mae"])
        print(f"{k:<16}{te.mean():>18.4f}±{te.std()/np.sqrt(len(te)):.3f}{fm.mean():>14.4f}")
    tb = np.array(res["cbam_boundary"]["temporal_err"])
    tf = np.array(res["cbam_full"]["temporal_err"])
    n = min(len(tb), len(tf))
    d = tb[:n] - tf[:n]  # >0 means full is better (lower change error)
    print(f"\nΔ(change err) = boundary - full = {d.mean():+.4f} K "
          f"(SEM {d.std()/np.sqrt(n):.4f})")
    print(f"full better on {int((d>0).sum())}/{n} pairs "
          f"({100*(d>0).mean():.0f}%)")
    rel = d.mean() / tb[:n].mean() * 100
    print(f"relative improvement from L_temp: {rel:+.1f}%")


if __name__ == "__main__":
    main()
