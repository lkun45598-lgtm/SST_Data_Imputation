"""
单模型 (全 24 小时共用一个网络) 的 FNO-CBAM: 在 FNO_CBAM_SST_Temporal 上增加
  - C 通道 (过去 48h 观测次数 / 48): 单独线性编码后加到编码特征上
  - 小时 sin/cos 的 FiLM: 每个 FNO 块 LayerNorm 之后做 x*(1+gamma)+beta
新增模块全部零初始化: 刚加载 OSTIA 预训练权重时，输出与原模型完全一致。
输出仍是绝对 SST (不做 B+网络输出 的残差化)。不修改原网络文件。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal


class FNO_CBAM_SST_Single(FNO_CBAM_SST_Temporal):
    def __init__(self, *args, use_c=True, use_film=True, **kw):
        super().__init__(*args, **kw)
        self.use_c = use_c
        self.use_film = use_film
        if use_c:
            self.c_encoder = nn.Linear(1, self.width)
            nn.init.zeros_(self.c_encoder.weight); nn.init.zeros_(self.c_encoder.bias)
        if use_film:
            self.film = nn.ModuleList()
            for _ in range(self.depth):
                lin = nn.Linear(2, 2 * self.width)
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
                self.film.append(lin)

    def forward(self, sst_seq, mask_seq, c_seq=None, hour_feat=None):
        """
        sst_seq, mask_seq: [B,30,H,W] (mask 1=缺失)；c_seq: [B,1,H,W] (C/48)；hour_feat: [B,2] (sin, cos)
        返回 [B,1,H,W] 绝对 SST (归一化空间)
        """
        sst = sst_seq.permute(0, 2, 3, 1)
        obs = (1 - mask_seq).permute(0, 2, 3, 1)
        x = torch.cat([self.sst_encoder(sst), self.mask_encoder(obs)], dim=-1)      # [B,H,W,width]
        if self.use_c:
            if c_seq is None:
                raise ValueError('use_c=True 需要 c_seq')
            x = x + self.c_encoder(c_seq.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)

        for i in range(self.depth):
            residual = x
            x = self.convs[i](x) + self.ws[i](x)
            if self.use_cbam:
                x = self.cbams[i](x)
            x = self.norms[i](x)
            if self.use_film:
                if hour_feat is None:
                    raise ValueError('use_film=True 需要 hour_feat')
                g, b = self.film[i](hour_feat).chunk(2, dim=-1)                       # [B,width] 各一
                x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
            x = F.gelu(residual + x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x.permute(0, 3, 1, 2)


def load_pretrained(model, path, map_location='cpu'):
    """加载 OSTIA 预训练权重 (strict=False)；返回 (missing, unexpected)。新增模块应只出现在 missing 里。"""
    ck = torch.load(path, map_location=map_location, weights_only=False)   # 自己的 OSTIA 检查点; torch>=2.6 默认 True 会拒绝非张量对象
    sd = ck.get('model_state_dict', ck)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    r = model.load_state_dict(sd, strict=False)
    return list(r.missing_keys), list(r.unexpected_keys)
