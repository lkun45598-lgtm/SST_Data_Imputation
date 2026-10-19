"""Generic Stage-2 JAXA finetuning for a deep baseline (dincae/unet/convlstm).
Matches the FNO's JAXA finetune protocol (training/train_jaxa_hourly.py):
  AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR(T_max=epochs, eta_min=1e-6),
  50 epochs, save best-val; train series 0-7, val 8, real-cloud masks,
  loss on artificial_mask∩obs∩ocean.  In-RAM data (baselines/fast_data.JAXAMem).
Usage: python3 baselines/finetune_deep.py --model unet --hour 12 --gpu 5
Saves: baselines/experiments/{model}_h{HH}/best_model.pth
"""
import sys, argparse, time, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
DI = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(DI))
from baselines.fast_data import JAXAMem
from baselines.deep_models import build_model, deep_loss, masked_mae_out
NORM_MEAN, NORM_STD = 299.9221, 2.6919


class _Tee:
    """Duplicate stdout to a persistent train.log in the save dir (like FNO)."""
    def __init__(self, path):
        self.f = open(path, 'a'); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


def run_epoch(model, dl, dev, opt=None):
    train = opt is not None; model.train(train); tot, mae, rmse, nb = 0.0, 0.0, 0.0, 0
    for b in dl:
        sst = b['input_sst_seq'].to(dev).float(); msk = b['mask_seq'].to(dev).float()
        gt = b['ground_truth_sst'].to(dev).float(); lm = b['loss_mask'].to(dev).float()
        with torch.set_grad_enabled(train):
            out = model(sst, msk); loss = deep_loss(out, gt, lm)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            mae += float(masked_mae_out(out, gt, lm)) * NORM_STD
            m = lm.unsqueeze(1)
            mse = ((out[0] - gt.unsqueeze(1)) ** 2 * m).sum() / (m.sum() + 1e-8)
            rmse += float(mse.sqrt()) * NORM_STD
        tot += float(loss); nb += 1
    return tot / max(nb, 1), mae / max(nb, 1), rmse / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['dincae', 'unet', 'convlstm', 'lama', 'partialconv'])
    ap.add_argument('--hour', type=int, default=12); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ratio-lo', type=float, default=0.2); ap.add_argument('--ratio-hi', type=float, default=0.3)
    ap.add_argument('--pretrained', type=str, default='auto',
                    help='"auto"=baselines/experiments/{model}_ostia_pretrain/best_model.pth; ""=from scratch')
    ap.add_argument('--tag', type=str, default='', help='optional experiment suffix, e.g. reduced')
    args = ap.parse_args()
    hh = f'{args.hour:02d}'; dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_dir = str(DI / f'experiments/hourly_data/h{hh}')
    tag = f'_{args.tag}' if args.tag else ''
    out_dir = DI / f'baselines/experiments/{args.model}{tag}_h{hh}'; out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'train.log')      # persistent log in the save dir (like FNO)
    rr = (args.ratio_lo, args.ratio_hi)
    tr = JAXAMem(data_dir, [0, 1, 2, 3, 4, 5, 6, 7], NORM_MEAN, NORM_STD, rr, seed=42)
    va = JAXAMem(data_dir, [8], NORM_MEAN, NORM_STD, rr, seed=7)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                     drop_last=True, persistent_workers=args.workers > 0)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True,
                     persistent_workers=args.workers > 0)
    model = build_model(args.model).to(dev)
    n = sum(p.numel() for p in model.parameters()); print(f'[{args.model}] h{hh} params: {n/1e6:.2f} M | dev={dev}')
    pre = args.pretrained
    if pre == 'auto':
        pre = f'baselines/experiments/{args.model}{tag}_ostia_pretrain/best_model.pth'
    if pre:
        pp = DI / pre if not Path(pre).is_absolute() else Path(pre)
        if pp.exists():
            ck = torch.load(pp, map_location=dev, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            print(f'   loaded OSTIA-pretrained: {pp} (val_MAE={ck.get("val_mae_K","?")} K)')
        else:
            print(f'   WARNING no pretrained at {pp} -> from scratch')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)  # same as FNO
    # config.json (like FNO's save format)
    config = {'model': args.model, 'hour': args.hour, 'tag': args.tag, 'epochs': args.epochs, 'batch': args.batch,
              'lr': args.lr, 'weight_decay': 1e-4, 'scheduler': 'CosineAnnealingLR', 'eta_min': 1e-6,
              'ratio_range': [args.ratio_lo, args.ratio_hi], 'train_series': [0, 1, 2, 3, 4, 5, 6, 7],
              'val_series': [8], 'norm_mean': NORM_MEAN, 'norm_std': NORM_STD,
              'params_M': round(n / 1e6, 4), 'pretrained': str(pre)}
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    history = {'train': [], 'valid': []}
    best = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_l, tr_mae, tr_rmse = run_epoch(model, trl, dev, opt)
        va_l, va_mae, va_rmse = run_epoch(model, val, dev, None)
        sched.step()
        dt = time.time() - t0
        print(f'[{args.model}] ep{ep:02d}/{args.epochs} | train L={tr_l:.4f} MAE={tr_mae:.4f}K RMSE={tr_rmse:.4f}K | '
              f'val MAE={va_mae:.4f}K RMSE={va_rmse:.4f}K | {dt:.0f}s | lr={opt.param_groups[0]["lr"]:.1e}', flush=True)
        history['train'].append({'epoch': ep, 'loss': tr_l, 'mae_K': tr_mae, 'rmse_K': tr_rmse})
        history['valid'].append({'epoch': ep, 'loss': va_l, 'mae_K': va_mae, 'rmse_K': va_rmse,
                                 'time_s': round(dt, 1), 'lr': opt.param_groups[0]['lr']})
        with open(out_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        if va_mae < best:
            best = va_mae
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep, 'val_mae_K': va_mae,
                        'val_rmse_K': va_rmse, 'norm_mean': NORM_MEAN, 'norm_std': NORM_STD,
                        'model_name': args.model}, out_dir / 'best_model.pth')
            print(f'   * saved best (val_MAE={va_mae:.4f} K)')
        if ep % 10 == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep, 'val_mae_K': va_mae,
                        'norm_mean': NORM_MEAN, 'norm_std': NORM_STD, 'model_name': args.model},
                       out_dir / f'checkpoint_epoch_{ep}.pth')
    torch.save({'model_state_dict': model.state_dict(), 'epoch': args.epochs, 'val_mae_K': va_mae,
                'val_rmse_K': va_rmse, 'norm_mean': NORM_MEAN, 'norm_std': NORM_STD,
                'model_name': args.model}, out_dir / 'final_model.pth')
    print(f'[{args.model}] DONE h{hh} | best val_MAE={best:.4f} K')


if __name__ == '__main__':
    main()
