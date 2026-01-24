"""
FNO_CBAM with Temporal (30-day) Input
支持60通道输入：30天SST + 30天mask
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def compl_mul2d(self, input, weights):
        weights_complex = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, weights_complex)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class CBAM_Block(nn.Module):
    def __init__(self, c_in, m_in, reduction_ratio=16):
        super(CBAM_Block, self).__init__()
        self.fc1 = nn.Linear(c_in, c_in // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(c_in // reduction_ratio, c_in, bias=False)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        batch_size, c, h, w = x.shape

        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, c)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        channel_attention = torch.sigmoid(avg_out + max_out).view(batch_size, c, 1, 1)
        x = x * channel_attention

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = torch.sigmoid(self.conv_spatial(spatial_input))
        x = x * spatial_attention

        return x


class FNO_CBAM_SST_Temporal(nn.Module):
    """
    FNO with CBAM for SST reconstruction - Temporal version

    输入：60通道 (30天SST + 30天mask)
    输出：1通道 (第30天重建SST)
    """
    def __init__(self, out_size, modes1=32, modes2=32, width=64, depth=6, cbam_reduction_ratio=16):
        super(FNO_CBAM_SST_Temporal, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.out_size = out_size

        # 输入编码：60通道 → width
        # 方法1：简单Linear
        # self.fc0 = nn.Linear(60, self.width)

        # 方法2：分别编码SST和mask后融合（更好）
        self.sst_encoder = nn.Linear(30, self.width // 2)
        self.mask_encoder = nn.Linear(30, self.width // 2)

        # FNO层
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        self.cbams = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.depth):
            self.convs.append(SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))

            m_in = out_size[0] * out_size[1]
            self.cbams.append(CBAM_Block(c_in=self.width, m_in=m_in, reduction_ratio=cbam_reduction_ratio))
            self.norms.append(nn.LayerNorm([self.width, out_size[0], out_size[1]]))

        # 输出解码
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, sst_seq, mask_seq):
        """
        Args:
            sst_seq: [B, 30, H, W] - 30天的SST
            mask_seq: [B, 30, H, W] - 30天的missing_mask

        Returns:
            pred: [B, 1, H, W] - 第30天的重建SST
        """
        B, T, H, W = sst_seq.shape

        # [B, 30, H, W] → [B, H, W, 30]
        sst_seq = sst_seq.permute(0, 2, 3, 1)
        mask_seq = (1 - mask_seq).permute(0, 2, 3, 1)  # 转成观测=1

        # 分别编码SST和mask
        sst_feat = self.sst_encoder(sst_seq)  # [B, H, W, width//2]
        mask_feat = self.mask_encoder(mask_seq)  # [B, H, W, width//2]

        # 融合
        x = torch.cat([sst_feat, mask_feat], dim=-1)  # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]

        # FNO处理
        for i in range(self.depth):
            residual = x

            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2

            x = self.cbams[i](x)
            x = self.norms[i](x)

            x = residual + x
            x = F.gelu(x)

        # 解码
        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # [B, H, W, 1]
        x = x.permute(0, 3, 1, 2)  # [B, 1, H, W]

        return x


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6
    ).to(device)

    print("="*60)
    print("FNO_CBAM_SST_Temporal 模型测试")
    print("="*60)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 测试前向传播
    B = 4
    sst_seq = torch.randn(B, 30, 451, 351).to(device)
    mask_seq = torch.randint(0, 2, (B, 30, 451, 351)).float().to(device)

    print(f"\n输入:")
    print(f"  sst_seq: {sst_seq.shape}")
    print(f"  mask_seq: {mask_seq.shape}")

    with torch.no_grad():
        output = model(sst_seq, mask_seq)

    print(f"\n输出:")
    print(f"  pred: {output.shape}")
    print(f"  pred范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

    print("\n" + "="*60)
    print("✓ 模型测试通过")
