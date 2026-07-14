"""DINCAE-style deep baseline for SST gap-filling.

Data-Interpolating Convolutional Auto-Encoder (Barth et al., 2020/2022, GMD).
A U-Net convolutional encoder-decoder with skip connections and DINCAE's
signature two-headed output: a mean field and an inverse-variance field,
trained with a Gaussian negative-log-likelihood on cross-validation-masked
(artificially hidden) observed pixels.

Adapted to THIS project's protocol for a fair comparison against FNO-CBAM:
  * SAME inputs as the FNO: sst_seq [B,30,H,W] + mask_seq [B,30,H,W]
    (the 30-day KNN-filled window carries the temporal context DINCAE 2.0
     is designed to exploit; mask marks the pixels to reconstruct).
  * SAME training data / split / masking (see train_dincae.py).
  * Prediction used for comparison = the mean head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class DINCAE(nn.Module):
    def __init__(self, in_frames=30, base=32, depth=4, out_size=(451, 351)):
        super().__init__()
        self.out_size = out_size
        in_ch = 2 * in_frames                      # sst_seq + mask_seq
        chs = [base * (2 ** i) for i in range(depth)]   # 32,64,128,256
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(_double_conv(prev, c)); prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(prev, prev * 2)
        self.reduce = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev2 = prev * 2
        for c in reversed(chs):
            self.reduce.append(nn.Conv2d(prev2, c, 1))
            self.dec.append(_double_conv(c * 2, c))
            prev2 = c
        self.head_mean = nn.Conv2d(prev2, 1, 1)
        self.head_ivar = nn.Conv2d(prev2, 1, 1)     # inverse variance (precision)

    def forward(self, sst_seq, mask_seq):
        x = torch.cat([sst_seq, mask_seq], dim=1)   # [B, 2*30, H, W]
        skips = []
        for enc in self.enc:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for reduce, dec, skip in zip(self.reduce, self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        mean = self.head_mean(x)                    # [B,1,H,W]
        # precision = 1/sigma^2, clamped to [1e-3, 100] (sigma in [0.1, ~32]) so the
        # NLL cannot be gamed by driving the variance head to extremes
        ivar = torch.clamp(F.softplus(self.head_ivar(x)) + 1e-3, max=100.0)
        return mean, ivar


def dincae_nll(mean, ivar, target, loss_mask):
    """Gaussian negative log-likelihood on loss_mask==1 pixels (DINCAE loss).
    L = 0.5 * mean_over_mask[ ivar*(mean-y)^2 - log(ivar) ]."""
    m = loss_mask.unsqueeze(1)
    diff2 = (mean - target.unsqueeze(1)) ** 2
    nll = 0.5 * (ivar * diff2 - torch.log(ivar))
    return (nll * m).sum() / (m.sum() + 1e-8)


def masked_mae(mean, target, loss_mask):
    m = loss_mask.unsqueeze(1)
    ae = torch.abs(mean - target.unsqueeze(1))
    return (ae * m).sum() / (m.sum() + 1e-8)
