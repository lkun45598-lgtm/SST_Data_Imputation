"""Stage-1 OSTIA pretraining for the DINCAE baseline — mirrors the FNO's
two-stage transfer (training/train_ostia.py) so the JAXA comparison is fair
(same pretraining, only the architecture differs).

  data   : /data/sst_data/.../processed_data/processed_sst_{train,valid}.h5
  input  : sst_seq [B,30,H,W] (nearest-filled) + mask_seq [B,30,H,W]
  target : full ground_truth_sst  (OSTIA is a gap-free reference)
  loss   : DINCAE Gaussian-NLL on the missing∩ocean region  [same region as FNO's L_missing]

Usage: python3 baselines/pretrain_dincae_ostia.py --gpu 0 --epochs 25
Saves: baselines/experiments/dincae_ostia_pretrain/best_model.pth
"""
import sys, argparse, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

DI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DI))
from datasets.ostia_dataset import SSTDatasetTemporal
from baselines.dincae import DINCAE, dincae_nll

OSTIA_DIR = '/data/sst_data/sst_missing_value_imputation/processed_data'


def run_epoch(model, loader, dev, opt=None):
    train = opt is not None
    model.train(train)
    tot_nll, tot_mae, nb = 0.0, 0.0, 0
    std = loader.dataset.std
    for b in loader:
        sst = b['input_sst_seq'].to(dev).float()
        msk = b['mask_seq'].to(dev).float()
        gt = b['ground_truth_sst'].to(dev).float()
        miss = b['missing_mask'].to(dev).float()
        land = b['land_mask'].to(dev).float()
        region = miss * (1.0 - land)                 # missing ∩ ocean (same as FNO L_missing)
        with torch.set_grad_enabled(train):
            mean, ivar = model(sst, msk)
            loss = dincae_nll(mean, ivar, gt, region)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            m = region.unsqueeze(1)
            mae = (torch.abs(mean - gt.unsqueeze(1)) * m).sum() / (m.sum() + 1e-8)
        tot_nll += float(loss); tot_mae += float(mae) * std; nb += 1
    return tot_nll / max(nb, 1), tot_mae / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--patience', type=int, default=6)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = DI / 'baselines/experiments/dincae_ostia_pretrain'; out_dir.mkdir(parents=True, exist_ok=True)

    tr = SSTDatasetTemporal(hdf5_path=f'{OSTIA_DIR}/processed_sst_train.h5', normalize=True)
    va = SSTDatasetTemporal(hdf5_path=f'{OSTIA_DIR}/processed_sst_valid.h5', normalize=True,
                            mean=tr.mean, std=tr.std)
    print(f'OSTIA pretrain | dev={dev} | train={len(tr)} val={len(va)} | mean={tr.mean:.3f} std={tr.std:.3f}')
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                     pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2,
                     pin_memory=True, persistent_workers=True)

    model = DINCAE(in_frames=30, base=32, depth=4, out_size=(451, 351)).to(dev)
    n = sum(p.numel() for p in model.parameters()); print(f'DINCAE params: {n/1e6:.2f} M')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best, bad = 1e9, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_nll, tr_mae = run_epoch(model, trl, dev, opt)
        va_nll, va_mae = run_epoch(model, val, dev, None)
        sched.step(va_mae)
        print(f'ep{ep:02d} | train NLL={tr_nll:.4f} MAE={tr_mae:.4f}K | '
              f'val NLL={va_nll:.4f} MAE={va_mae:.4f}K | {time.time()-t0:.0f}s '
              f'| lr={opt.param_groups[0]["lr"]:.1e}', flush=True)
        if va_mae < best:
            best, bad = va_mae, 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep,
                        'val_mae_K': va_mae, 'norm_mean': tr.mean, 'norm_std': tr.std,
                        'arch': dict(in_frames=30, base=32, depth=4, out_size=(451, 351))},
                       out_dir / 'best_model.pth')
            print(f'   * saved best (val_MAE={va_mae:.4f} K)')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'early stop ep{ep} (best val_MAE={best:.4f} K)'); break
    print(f'DONE OSTIA pretrain | best val_MAE={best:.4f} K | -> {out_dir}/best_model.pth')


if __name__ == '__main__':
    main()
