"""Deep baselines for SST gap-filling — unified interface for fair comparison.

All models share the SAME I/O as the FNO:
    forward(sst_seq [B,30,H,W], mask_seq [B,30,H,W]) -> (mean [B,1,H,W], ivar or None)
and are trained with the SAME two-stage protocol (OSTIA pretrain -> JAXA finetune),
data, split and masking. Only the ARCHITECTURE differs.

  - DINCAE   : U-Net conv autoencoder + inverse-variance head, Gaussian NLL loss.
  - UNet     : same conv autoencoder, single (mean) head, masked-MSE loss
               (= generic deep inpainting; also an ablation of DINCAE's NLL head).
  - ConvLSTM : spatiotemporal recurrent encoder over the 30 frames + conv decoder,
               masked-MSE loss (a recurrent inductive bias, contrast to FNO spectral).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.dincae import DINCAE, _double_conv, dincae_nll


# ----------------------------------------------------------------- U-Net (MSE)
class UNet(nn.Module):
    def __init__(self, in_frames=30, base=32, depth=4, out_size=(451, 351)):
        super().__init__()
        self.out_size = out_size
        in_ch = 2 * in_frames
        chs = [base * (2 ** i) for i in range(depth)]
        self.enc = nn.ModuleList(); prev = in_ch
        for c in chs:
            self.enc.append(_double_conv(prev, c)); prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(prev, prev * 2)
        self.reduce = nn.ModuleList(); self.dec = nn.ModuleList(); prev2 = prev * 2
        for c in reversed(chs):
            self.reduce.append(nn.Conv2d(prev2, c, 1)); self.dec.append(_double_conv(c * 2, c)); prev2 = c
        self.head = nn.Conv2d(prev2, 1, 1)

    def forward(self, sst_seq, mask_seq):
        x = torch.cat([sst_seq, mask_seq], dim=1)
        skips = []
        for enc in self.enc:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for reduce, dec, skip in zip(self.reduce, self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([reduce(x), skip], dim=1); x = dec(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return self.head(x), None


# ----------------------------------------------------------------- ConvLSTM
class ConvLSTMCell(nn.Module):
    def __init__(self, ch, hid, k=3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(ch + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, h, c):
        g = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g_ = torch.chunk(g, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g_)
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """Downsample each frame, run ConvLSTM over 30 steps, decode last hidden state."""
    def __init__(self, in_frames=30, hid=48, out_size=(451, 351)):
        super().__init__()
        self.out_size = out_size
        self.frame_enc = nn.Sequential(                    # per-frame 2ch -> features, /4 spatial
            nn.Conv2d(2, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, hid, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.cell = ConvLSTMCell(hid, hid)
        self.dec = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.BatchNorm2d(hid), nn.ReLU(inplace=True),
            nn.Conv2d(hid, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, sst_seq, mask_seq):
        B, T, H, W = sst_seq.shape
        h = c = None
        for t in range(T):
            x = torch.stack([sst_seq[:, t], mask_seq[:, t]], dim=1)  # [B,2,H,W]
            f = self.frame_enc(x)                                    # [B,hid,H/4,W/4]
            if h is None:
                h = torch.zeros_like(f); c = torch.zeros_like(f)
            h, c = self.cell(f, h, c)
        y = self.dec(h)
        y = F.interpolate(y, size=self.out_size, mode="bilinear", align_corners=False)
        return self.head(y), None


# ----------------------------------------------------------------- factory + loss
def build_model(name, out_size=(451, 351)):
    name = name.lower()
    if name == "dincae":
        return DINCAE(in_frames=30, base=32, depth=4, out_size=out_size)
    if name == "unet":
        return UNet(in_frames=30, base=32, depth=4, out_size=out_size)
    if name == "convlstm":
        return ConvLSTM(in_frames=30, hid=48, out_size=out_size)
    raise ValueError(f"unknown model {name}")


def deep_loss(out, target, region):
    """NLL if the model has an inverse-variance head, else masked MSE."""
    mean, ivar = out
    if ivar is None:
        m = region.unsqueeze(1)
        return ((mean - target.unsqueeze(1)) ** 2 * m).sum() / (m.sum() + 1e-8)
    return dincae_nll(mean, ivar, target, region)


def masked_mae_out(out, target, region):
    mean = out[0]; m = region.unsqueeze(1)
    return (torch.abs(mean - target.unsqueeze(1)) * m).sum() / (m.sum() + 1e-8)
