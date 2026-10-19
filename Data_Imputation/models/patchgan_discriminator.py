#!/usr/bin/env python3
"""
浅层 PatchGAN 判别器 + 块有效性计算 + LSGAN 损失

只在训练时使用，推理不需要。输入整帧 SST (B,1,H,W)，输出分数图 (B,1,h,w)，
分数图上每个位置只由其感受野内的像素决定。

块有效性: 判别器只应评价"感受野内全是真实观测"的位置，
否则 KNN/时间填充值会被当成真实纹理。patch_fraction 用与判别器相同的
(k, s, p) 几何对掩码做平均池化，结果==1 表示感受野完全落在掩码内。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """
    n_down 个 stride-2 卷积 + 1 个 3x3 输出卷积，谱归一化。
    n_down=2 时感受野 18 像素，n_down=3 时 42 像素 (1 像素约 0.02°)。
    """

    def __init__(self, in_ch=1, base=32, n_down=2):
        super().__init__()
        self.geometry = [(4, 2, 1)] * n_down + [(3, 1, 1)]

        layers = []
        c_in = in_ch
        c_out = base
        for k, s, p in self.geometry[:-1]:
            layers += [spectral_norm(nn.Conv2d(c_in, c_out, k, s, p)),
                       nn.LeakyReLU(0.2, inplace=True)]
            c_in, c_out = c_out, min(c_out * 2, base * 8)
        k, s, p = self.geometry[-1]
        layers.append(spectral_norm(nn.Conv2d(c_in, 1, k, s, p)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def patch_fraction(self, mask):
        """mask: (B,1,H,W) 0/1 -> (B,1,h,w)，每个输出位置的感受野内 mask 的覆盖比例"""
        return patch_fraction(mask, self.geometry)


def patch_fraction(mask, geometry):
    x = mask
    for k, s, p in geometry:
        # 零填充计入分母: 越界部分视为无效，边缘位置只有在全有效时才满分
        x = F.avg_pool2d(x, k, s, p, count_include_pad=True)
    return x


def full_valid(frac, tol=1e-4):
    """感受野完全落在掩码内"""
    return (frac > 1.0 - tol).float()


def any_overlap(frac, eps=1e-6):
    """感受野内至少含一个掩码像素"""
    return (frac > eps).float()


def masked_mean(x, w):
    return (x * w).sum() / (w.sum() + 1e-8)


def d_loss_lsgan(real_score, real_w, fake_score, fake_w):
    return 0.5 * (masked_mean((real_score - 1.0) ** 2, real_w)
                  + masked_mean(fake_score ** 2, fake_w))


def g_loss_lsgan(fake_score, fake_w):
    return masked_mean((fake_score - 1.0) ** 2, fake_w)


if __name__ == '__main__':
    D = PatchDiscriminator()
    x = torch.randn(2, 1, 451, 351)
    s = D(x)
    m = torch.ones(2, 1, 451, 351)
    f = D.patch_fraction(m)
    print('score', tuple(s.shape), 'frac', tuple(f.shape),
          'valid cells', int(full_valid(f).sum().item()), '/', f.numel())
