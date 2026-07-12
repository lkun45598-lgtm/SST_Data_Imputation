"""
FNO-CBAM model variant for ablation experiments.

Adapted from Data_Imputation/models/fno_cbam_temporal.py with a `use_cbam`
toggle in the FNO block. Kept as a SEPARATE file so the main model is never
touched by ablation work — protects reproducibility of the production model.

Toggling:
  use_cbam=True   -> identical forward to the main model
  use_cbam=False  -> CBAM block is bypassed (parameters still allocated to
                     allow loading from a CBAM-pretrained checkpoint; pass
                     find_unused_parameters=True to DDP)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def compl_mul2d(self, x, w):
        w_c = torch.view_as_complex(w)
        return torch.einsum("bixy,ioxy->boxy", x, w_c)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class CBAM_Block(nn.Module):
    def __init__(self, c_in, m_in, reduction_ratio=16):
        super().__init__()
        self.fc1 = nn.Linear(c_in, c_in // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(c_in // reduction_ratio, c_in, bias=False)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention
        avg_p = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_p = F.adaptive_max_pool2d(x, 1).view(B, C)
        avg_out = self.fc2(F.relu(self.fc1(avg_p)))
        max_out = self.fc2(F.relu(self.fc1(max_p)))
        ch_attn = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        x = x * ch_attn
        # Spatial attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        sp_in = torch.cat([avg_s, max_s], dim=1)
        sp_attn = torch.sigmoid(self.conv_spatial(sp_in))
        return x * sp_attn


class FNO_CBAM_Ablation(nn.Module):
    """Same architecture as FNO_CBAM_SST_Temporal but with a use_cbam toggle.

    Renamed to avoid pickle/import collisions with the main model class."""

    def __init__(self, out_size, modes1=32, modes2=32, width=64, depth=6,
                 cbam_reduction_ratio=16, use_cbam=True):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.out_size = out_size
        self.use_cbam = use_cbam

        self.sst_encoder = nn.Linear(30, self.width // 2)
        self.mask_encoder = nn.Linear(30, self.width // 2)

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        self.cbams = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.depth):
            self.convs.append(SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))
            m_in = out_size[0] * out_size[1]
            self.cbams.append(CBAM_Block(c_in=self.width, m_in=m_in,
                                         reduction_ratio=cbam_reduction_ratio))
            self.norms.append(nn.LayerNorm([self.width, out_size[0], out_size[1]]))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, sst_seq, mask_seq):
        sst_seq = sst_seq.permute(0, 2, 3, 1)
        mask_seq = (1 - mask_seq).permute(0, 2, 3, 1)
        sst_feat = self.sst_encoder(sst_seq)
        mask_feat = self.mask_encoder(mask_seq)
        x = torch.cat([sst_feat, mask_feat], dim=-1)
        x = x.permute(0, 3, 1, 2)
        for i in range(self.depth):
            residual = x
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if self.use_cbam:
                x = self.cbams[i](x)
            x = self.norms[i](x)
            x = residual + x
            x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for flag in (True, False):
        m = FNO_CBAM_Ablation(out_size=(451, 351), modes1=80, modes2=64,
                              width=64, depth=6, use_cbam=flag).to(device)
        n = sum(p.numel() for p in m.parameters())
        sst = torch.randn(1, 30, 451, 351, device=device)
        msk = torch.randint(0, 2, (1, 30, 451, 351), device=device).float()
        with torch.no_grad():
            y = m(sst, msk)
        print(f"use_cbam={flag}: params={n:,}, out_shape={tuple(y.shape)}")
