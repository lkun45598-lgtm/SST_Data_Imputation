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
    def __init__(self, in_frames=30, base=16, depth=3, out_size=(451, 351)):
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
    def __init__(self, in_frames=30, hid=8, out_size=(451, 351)):
        super().__init__()
        self.out_size = out_size
        self.frame_enc = nn.Sequential(                    # per-frame 2ch -> features, /8 spatial
            nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, hid, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.cell = ConvLSTMCell(hid, hid)
        self.dec = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.BatchNorm2d(hid), nn.ReLU(inplace=True),
            nn.Conv2d(hid, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(8, 1, 1)

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


# ----------------------------------------------------------------- Partial-Conv U-Net
class PartialConv2d(nn.Module):
    """Mask-aware convolution (Liu et al. 2018, NVIDIA): the conv sees only valid
    pixels and is renormalized by the valid-count in each window; the validity
    mask propagates forward (a window with any valid pixel becomes valid)."""
    def __init__(self, cin, cout, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding, bias=True)
        self.stride, self.padding = stride, padding
        self.register_buffer("wmask", torch.ones(1, 1, k, k))
        self.winsize = float(k * k)

    def forward(self, x, m):
        # x:[B,Cin,H,W]  m:[B,1,H,W] validity (1=valid)
        with torch.no_grad():
            msum = F.conv2d(m, self.wmask, stride=self.stride, padding=self.padding)
            mnew = (msum > 0).float()
            ratio = self.winsize / (msum + 1e-8) * mnew          # scale valid-only sum back up
        out = self.conv(x * m)
        b = self.conv.bias.view(1, -1, 1, 1)
        out = (out - b) * ratio + b
        return out * mnew, mnew


class _PCDown(nn.Module):
    def __init__(self, cin, cout, k, stride, padding, bn=True):
        super().__init__()
        self.pc = PartialConv2d(cin, cout, k, stride, padding)
        self.bn = nn.BatchNorm2d(cout) if bn else None

    def forward(self, x, m):
        x, m = self.pc(x, m)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True), m


class PartialConvUNet(nn.Module):
    """Classic inpainting baseline: partial-convolution U-Net. Same unified I/O as
    the FNO (30×SST+30×mask in, 1 mean map out). Validity = last-frame observed."""
    def __init__(self, in_frames=30, base=20, out_size=(451, 351)):
        super().__init__()
        self.out_size = out_size
        cin = 2 * in_frames
        self.d1 = _PCDown(cin,    base,    7, 2, 3, bn=False)
        self.d2 = _PCDown(base,   base*2,  5, 2, 2)
        self.d3 = _PCDown(base*2, base*4,  5, 2, 2)
        self.d4 = _PCDown(base*4, base*8,  3, 2, 1)
        self.u4 = PartialConv2d(base*8 + base*4, base*4, 3, 1, 1)
        self.u3 = PartialConv2d(base*4 + base*2, base*2, 3, 1, 1)
        self.u2 = PartialConv2d(base*2 + base,   base,   3, 1, 1)
        self.u1 = PartialConv2d(base   + cin,    base,   3, 1, 1)
        self.bn4 = nn.BatchNorm2d(base*4); self.bn3 = nn.BatchNorm2d(base*2)
        self.bn2 = nn.BatchNorm2d(base);   self.bn1 = nn.BatchNorm2d(base)
        self.head = nn.Conv2d(base, 1, 1)

    def _up(self, x, m, xs, ms):
        size = xs.shape[-2:]
        x = F.interpolate(x, size=size, mode="nearest")
        m = F.interpolate(m, size=size, mode="nearest")
        return torch.cat([x, xs], 1), torch.max(m, ms)   # union validity (single channel)

    def forward(self, sst_seq, mask_seq):
        x0 = torch.cat([sst_seq, mask_seq], 1)          # [B,60,H,W]
        m0 = (mask_seq[:, -1:] < 0.5).float()            # validity of the day being filled
        x1, m1 = self.d1(x0, m0)
        x2, m2 = self.d2(x1, m1)
        x3, m3 = self.d3(x2, m2)
        x4, m4 = self.d4(x3, m3)
        xc, mc = self._up(x4, m4, x3, m3); x, m = self.u4(xc, mc); x = F.relu(self.bn4(x), inplace=True)
        xc, mc = self._up(x,  m,  x2, m2); x, m = self.u3(xc, mc); x = F.relu(self.bn3(x), inplace=True)
        xc, mc = self._up(x,  m,  x1, m1); x, m = self.u2(xc, mc); x = F.relu(self.bn2(x), inplace=True)
        xc, mc = self._up(x,  m,  x0, m0); x, m = self.u1(xc, mc); x = F.relu(self.bn1(x), inplace=True)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return self.head(x), None


# ----------------------------------------------------------------- LaMa / FFC (spectral)
class FourierUnit(nn.Module):
    """Global-context branch: 1×1 conv applied in the 2-D Fourier domain."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin * 2, cout * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout * 2)

    def forward(self, x):
        B, C, H, W = x.shape
        ff = torch.fft.rfft2(x.float(), norm="ortho")            # complex [B,C,H,W//2+1]
        ff = torch.cat([ff.real, ff.imag], dim=1)
        ff = F.relu(self.bn(self.conv(ff)), inplace=True)
        r, i = torch.chunk(ff, 2, dim=1)
        return torch.fft.irfft2(torch.complex(r, i), s=(H, W), norm="ortho")


class SpectralTransform(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(cin, cout // 2, 1, bias=False),
                                   nn.BatchNorm2d(cout // 2), nn.ReLU(inplace=True))
        self.fu = FourierUnit(cout // 2, cout // 2)
        self.conv2 = nn.Conv2d(cout // 2, cout, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


class FFC(nn.Module):
    """Fast Fourier Convolution: channels split into a local (spatial conv) and a
    global (spectral) branch with cross-branch mixing."""
    def __init__(self, cin, cout, ratio=0.5, k=3, padding=1):
        super().__init__()
        cin_g = int(cin * ratio); cin_l = cin - cin_g
        cout_g = int(cout * ratio); cout_l = cout - cout_g
        self.l2l = nn.Conv2d(cin_l, cout_l, k, 1, padding, bias=False) if cin_l and cout_l else None
        self.l2g = nn.Conv2d(cin_l, cout_g, k, 1, padding, bias=False) if cin_l and cout_g else None
        self.g2l = nn.Conv2d(cin_g, cout_l, k, 1, padding, bias=False) if cin_g and cout_l else None
        self.g2g = SpectralTransform(cin_g, cout_g) if cin_g and cout_g else None

    def forward(self, x):
        x_l, x_g = x
        out_l = out_g = 0
        if self.l2l is not None: out_l = out_l + self.l2l(x_l)
        if self.g2l is not None: out_l = out_l + self.g2l(x_g)
        if self.l2g is not None: out_g = out_g + self.l2g(x_l)
        if self.g2g is not None: out_g = out_g + self.g2g(x_g)
        return out_l, out_g


class FFCBNAct(nn.Module):
    def __init__(self, cin, cout, ratio=0.5):
        super().__init__()
        self.ffc = FFC(cin, cout, ratio)
        cout_g = int(cout * ratio); cout_l = cout - cout_g
        self.bn_l = nn.BatchNorm2d(cout_l); self.bn_g = nn.BatchNorm2d(cout_g)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        return self.act(self.bn_l(x_l)), self.act(self.bn_g(x_g))


class FFCResBlock(nn.Module):
    def __init__(self, dim, ratio=0.5):
        super().__init__()
        self.c1 = FFCBNAct(dim, dim, ratio)
        self.c2 = FFCBNAct(dim, dim, ratio)

    def forward(self, x):
        x_l, x_g = x
        o_l, o_g = self.c2(self.c1((x_l, x_g)))
        return x_l + o_l, x_g + o_g


class LaMaFFC(nn.Module):
    """LaMa-style generator with Fast Fourier Convolution residual blocks — a
    SPECTRAL inpainting baseline (global receptive field via FFT), the natural
    counterpart to the FNO's spectral convolutions. Same unified I/O."""
    def __init__(self, in_frames=30, base=40, n_blocks=4, ratio=0.5, out_size=(451, 351)):
        super().__init__()
        self.out_size = out_size
        cin = 2 * in_frames
        self.stem = nn.Sequential(
            nn.Conv2d(cin, base, 7, 1, 3, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, 2, 1, bias=False), nn.BatchNorm2d(base*2), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, 2, 1, bias=False), nn.BatchNorm2d(base*4), nn.ReLU(inplace=True))
        dim = base * 4
        self.cg = int(dim * ratio); self.cl = dim - self.cg
        self.blocks = nn.ModuleList([FFCResBlock(dim, ratio) for _ in range(n_blocks)])
        self.up2 = nn.Sequential(
            nn.Conv2d(dim, base*2, 3, 1, 1, bias=False), nn.BatchNorm2d(base*2), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 1, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.tail = nn.Conv2d(base, 1, 7, 1, 3)

    def forward(self, sst_seq, mask_seq):
        x = torch.cat([sst_seq, mask_seq], dim=1)
        s1 = x.shape[-2:]
        x = self.stem(x)
        x = self.down1(x); s_d1 = x.shape[-2:]
        x = self.down2(x)
        x_l, x_g = x[:, :self.cl], x[:, self.cl:]
        for blk in self.blocks:
            x_l, x_g = blk((x_l, x_g))
        x = torch.cat([x_l, x_g], dim=1)
        x = self.up2(F.interpolate(x, size=s_d1, mode="bilinear", align_corners=False))
        x = self.up1(F.interpolate(x, size=s1,   mode="bilinear", align_corners=False))
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return self.tail(x), None


# ----------------------------------------------------------------- factory + loss
def build_model(name, out_size=(451, 351)):
    name = name.lower()
    if name == "dincae":
        return DINCAE(in_frames=30, base=16, depth=3, out_size=out_size)
    if name == "unet":
        return UNet(in_frames=30, base=16, depth=3, out_size=out_size)
    if name == "convlstm":
        return ConvLSTM(in_frames=30, hid=8, out_size=out_size)
    if name == "lama":
        return LaMaFFC(in_frames=30, base=40, n_blocks=4, out_size=out_size)
    if name == "partialconv":
        return PartialConvUNet(in_frames=30, base=20, out_size=out_size)
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
