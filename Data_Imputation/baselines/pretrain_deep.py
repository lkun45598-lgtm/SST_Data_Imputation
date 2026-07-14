"""Generic Stage-1 OSTIA pretraining for a deep baseline (dincae/unet/convlstm).
Matches the FNO's OSTIA pretrain protocol (training/train_ostia.py):
  AdamW lr=1e-3 wd=1e-4, StepLR(step=15, gamma=0.5), 60 epochs, save best-val.
Loss on missing∩ocean vs full GT.  In-RAM data (baselines/fast_data.OSTIAMem).
Usage: python3 baselines/pretrain_deep.py --model unet --gpu 5
Saves: baselines/experiments/{model}_ostia_pretrain/best_model.pth
"""
import sys, argparse, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
DI = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(DI))
from baselines.fast_data import OSTIAMem
from baselines.deep_models import build_model, deep_loss, masked_mae_out
OSTIA_DIR = '/data/sst_data/sst_missing_value_imputation/processed_data'


def run_epoch(model, loader, dev, std, opt=None):
    train = opt is not None; model.train(train)
    tot, mae_k, nb = 0.0, 0.0, 0
    for b in loader:
        sst = b['input_sst_seq'].to(dev).float(); msk = b['mask_seq'].to(dev).float()
        gt = b['ground_truth_sst'].to(dev).float()
        region = (b['missing_mask'].to(dev).float() * (1.0 - b['land_mask'].to(dev).float())
                  * b['gt_valid'].to(dev).float())          # exclude NaN-gt pixels
        with torch.set_grad_enabled(train):
            out = model(sst, msk); loss = deep_loss(out, gt, region)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            mae_k += float(masked_mae_out(out, gt, region)) * std
        tot += float(loss); nb += 1
    return tot / max(nb, 1), mae_k / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['dincae', 'unet', 'convlstm'])
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = DI / f'baselines/experiments/{args.model}_ostia_pretrain'; out_dir.mkdir(parents=True, exist_ok=True)
    tr = OSTIAMem(f'{OSTIA_DIR}/processed_sst_train.h5')
    va = OSTIAMem(f'{OSTIA_DIR}/processed_sst_valid.h5', mean=tr.mean, std=tr.std)
    print(f'[{args.model}] OSTIA pretrain | dev={dev} | train={len(tr)} val={len(va)} | mean={tr.mean:.3f} std={tr.std:.3f}')
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                     drop_last=True, persistent_workers=args.workers > 0)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    model = build_model(args.model).to(dev)
    n = sum(p.numel() for p in model.parameters()); print(f'[{args.model}] params: {n/1e6:.2f} M')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)      # same as FNO OSTIA
    best = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_l, tr_mae = run_epoch(model, trl, dev, tr.std, opt)
        va_l, va_mae = run_epoch(model, val, dev, tr.std, None)
        sched.step()
        print(f'[{args.model}] ep{ep:02d}/{args.epochs} | train L={tr_l:.4f} MAE={tr_mae:.4f}K | '
              f'val L={va_l:.4f} MAE={va_mae:.4f}K | {time.time()-t0:.0f}s | lr={opt.param_groups[0]["lr"]:.1e}', flush=True)
        if va_mae < best:
            best = va_mae
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep, 'val_mae_K': va_mae,
                        'norm_mean': tr.mean, 'norm_std': tr.std, 'model_name': args.model}, out_dir / 'best_model.pth')
            print(f'   * saved best (val_MAE={va_mae:.4f} K)')
    print(f'[{args.model}] DONE OSTIA pretrain | best val_MAE={best:.4f} K')


if __name__ == '__main__':
    main()
