"""Train the DINCAE deep baseline on the SAME protocol as the FNO fine-tune.

Fair comparison guarantees:
  * data      = experiments/hourly_data/h{HH}/jaxa_knn_filled_{0..8}.h5
  * split     = train series 0-7, val series 8 (held-out)  [identical to FNO]
  * masking   = real-cloud masks, ratio 0.2-0.3            [identical to FNO realmask]
  * loss region = artificial_mask ∩ original_obs_mask ∩ ocean  [identical to FNO]
  * normalization = norm_mean/std pinned to FNO's (299.9221 / 2.6919)
Only the ARCHITECTURE differs (conv autoencoder vs FNO spectral).

Usage: python3 baselines/train_dincae.py --hour 12 --gpu 0 --epochs 40
Saves: baselines/experiments/dincae_h{HH}/best_model.pth
"""
import sys, argparse, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

DI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DI))
from inference.jaxa_inference_dataset import JAXAFinetuneDataset
from baselines.dincae import DINCAE, dincae_nll, masked_mae

NORM_MEAN, NORM_STD = 299.9221, 2.6919


def build_loader(data_dir, series_ids, ratio_range, batch, shuffle, workers, seed):
    ds = JAXAFinetuneDataset(
        data_dir=str(data_dir), series_ids=series_ids, window_size=30,
        normalize=True, mean=NORM_MEAN, std=NORM_STD, cache_size=60, seed=seed,
        mask_mode='realcloud', ratio_range=ratio_range)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hour', type=int, default=12)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--ratio-lo', type=float, default=0.2)
    ap.add_argument('--ratio-hi', type=float, default=0.3)
    ap.add_argument('--pretrained', type=str,
                    default='baselines/experiments/dincae_ostia_pretrain/best_model.pth',
                    help='OSTIA-pretrained weights to transfer (two-stage, mirrors FNO); "" to train from scratch')
    args = ap.parse_args()
    hh = f'{args.hour:02d}'
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_dir = DI / f'experiments/hourly_data/h{hh}'
    out_dir = DI / f'baselines/experiments/dincae_h{hh}'; out_dir.mkdir(parents=True, exist_ok=True)
    print(f'DINCAE h{hh} | dev={dev} | data={data_dir}')

    rr = (args.ratio_lo, args.ratio_hi)
    train_loader = build_loader(data_dir, [0, 1, 2, 3, 4, 5, 6, 7], rr, args.batch, True, args.workers, 42)
    val_loader = build_loader(data_dir, [8], rr, args.batch, False, 2, 7)

    model = DINCAE(in_frames=30, base=32, depth=4, out_size=(451, 351)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'DINCAE params: {nparam:,} ({nparam/1e6:.2f} M)')
    if args.pretrained:
        pp = DI / args.pretrained if not Path(args.pretrained).is_absolute() else Path(args.pretrained)
        if pp.exists():
            ck = torch.load(pp, map_location=dev, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            print(f'  loaded OSTIA-pretrained weights: {pp} (val_MAE={ck.get("val_mae_K","?")} K)')
        else:
            print(f'  WARNING pretrained not found ({pp}) -> training from scratch')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best_mae, bad = 1e9, 0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tr_nll = 0.0; nb = 0
        for b in train_loader:
            sst = b['input_sst_seq'].to(dev); msk = b['mask_seq'].to(dev)
            gt = b['ground_truth_sst'].to(dev); lm = b['loss_mask'].to(dev)
            opt.zero_grad()
            mean, ivar = model(sst, msk)
            loss = dincae_nll(mean, ivar, gt, lm)
            loss.backward(); opt.step()
            tr_nll += float(loss); nb += 1
        # validate
        model.eval(); vmae = 0.0; vnb = 0
        with torch.no_grad():
            for b in val_loader:
                sst = b['input_sst_seq'].to(dev); msk = b['mask_seq'].to(dev)
                gt = b['ground_truth_sst'].to(dev); lm = b['loss_mask'].to(dev)
                mean, ivar = model(sst, msk)
                vmae += float(masked_mae(mean, gt, lm)); vnb += 1
        vmae_k = (vmae / max(vnb, 1)) * NORM_STD    # normalized -> Kelvin
        sched.step(vmae_k)
        print(f'ep{ep:02d} | train_nll={tr_nll/max(nb,1):.4f} | val_MAE={vmae_k:.4f} K '
              f'| {time.time()-t0:.0f}s | lr={opt.param_groups[0]["lr"]:.1e}', flush=True)
        if vmae_k < best_mae:
            best_mae = vmae_k; bad = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep,
                        'val_mae_K': vmae_k, 'norm_mean': NORM_MEAN, 'norm_std': NORM_STD,
                        'arch': dict(in_frames=30, base=32, depth=4, out_size=(451, 351))},
                       out_dir / 'best_model.pth')
            print(f'   * saved best (val_MAE={vmae_k:.4f} K)')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'early stop at ep{ep} (best val_MAE={best_mae:.4f} K)'); break
    print(f'DONE h{hh} | best val_MAE={best_mae:.4f} K | -> {out_dir}/best_model.pth')


if __name__ == '__main__':
    main()
