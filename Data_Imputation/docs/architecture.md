# 模型架构说明

## 概述

FNO-CBAM-Temporal（类名 `FNO_CBAM_SST_Temporal`）是一个结合傅里叶神经算子 (FNO) 和卷积注意力模块 (CBAM) 的时序 SST 重建模型。模型接收 30 天的 SST 序列与对应的缺失掩膜序列，输出第 30 天的重建 SST。

---

## 整体架构

```
输入: sst_seq  [B, 30, H, W]  (Day 1-30 的 SST，归一化)
      mask_seq [B, 30, H, W]  (Day 1-30 的缺失掩膜, 1=缺失/0=观测)
          │
          │  注意: forward() 内部将 mask 取反 (1 - mask)，
          │        转换成 "观测=1" 后再送入编码器
          ▼
┌──────────────────────────────────────────────┐
│  Dual Lifting (双通道提升)                     │
│  sst_encoder : Linear(30 → width//2 = 32)     │
│  mask_encoder: Linear(30 → width//2 = 32)     │
│  在通道维拼接 → width = 64                      │
└──────────────────────────────────────────────┘
          │  [B, width=64, H, W]
          ▼
┌──────────────────────────────────────────────┐
│  FNO-CBAM Block × 6 (depth=6)                 │
│  residual = x                                  │
│  x1 = SpectralConv2d_fast(x)                   │
│  x2 = Conv2d 1×1 (x)                           │
│  x  = x1 + x2                                  │
│  x  = CBAM(x)                                  │
│  x  = LayerNorm(x)                             │
│  x  = residual + x                             │
│  x  = GELU(x)                                  │
└──────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│  Decoder (逐像素 MLP，在通道维操作)             │
│  fc1: Linear(64 → 128)                         │
│  GELU                                          │
│  fc2: Linear(128 → 1)                          │
└──────────────────────────────────────────────┘
          │
          ▼
输出: [B, 1, H, W]  (Day 30 重建 SST，归一化)
      反归一化: sst_kelvin = output * std + mean
```

---

## 核心组件

### 1. Spectral Convolution (傅里叶卷积)

实现类 `SpectralConv2d_fast`，是傅里叶神经算子的核心，在频域进行全局卷积：

```python
class SpectralConv2d_fast(nn.Module):
    def forward(self, x):
        # 1. FFT 变换到频域 (实数 FFT)
        x_ft = torch.fft.rfft2(x)

        # 2. 只保留低频模式，并分别处理"正纬度"和"负纬度"两个模式块：
        #    [:modes1]  的前 modes2 列 → 与 weights1 复数相乘
        #    [-modes1:] 的前 modes2 列 → 与 weights2 复数相乘
        out_ft[:, :, :modes1,  :modes2] = compl_mul2d(x_ft[:, :, :modes1,  :modes2], weights1)
        out_ft[:, :, -modes1:, :modes2] = compl_mul2d(x_ft[:, :, -modes1:, :modes2], weights2)

        # 3. 逆 FFT 回到空间域
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x
```

要点：
- 权重 `weights1` / `weights2` 形状为 `[in_channels, out_channels, modes1, modes2, 2]`，最后一维通过 `torch.view_as_complex` 解释为复数。
- 复数乘法用 `torch.einsum("bixy,ioxy->boxy", ...)` 实现。
- 保留两个纬度方向的模式块（低频头部 `[:modes1]` 与高频尾部 `[-modes1:]`），是 2D FFT 频谱对称结构下保留低频的标准做法。

**优势**:
- 全局感受野，捕获大尺度空间模式
- 频域参数化，适合处理海温这类具有大尺度连续结构的场
- 天然适合处理周期性/波动性数据

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| modes1 | 80 | 纬度方向保留的傅里叶模式数 |
| modes2 | 64 | 经度方向保留的傅里叶模式数 |

### 2. CBAM (Convolutional Block Attention Module)

实现类 `CBAM_Block`，串行的双重注意力：先通道注意力，后空间注意力。

```python
class CBAM_Block(nn.Module):
    def forward(self, x):
        # 1. 通道注意力
        avg = adaptive_avg_pool2d(x, 1)   # [B, C]
        max = adaptive_max_pool2d(x, 1)   # [B, C]
        # 共享的两层 MLP: Linear(C → C/16) → ReLU → Linear(C/16 → C)
        ch = sigmoid( mlp(avg) + mlp(max) ).view(B, C, 1, 1)
        x = x * ch

        # 2. 空间注意力
        avg = mean(x, dim=1, keepdim=True)     # 通道维均值 [B, 1, H, W]
        max = max(x,  dim=1, keepdim=True)      # 通道维最大 [B, 1, H, W]
        sp  = sigmoid( Conv2d(2→1, kernel=7×7)( cat([avg, max]) ) )
        x = x * sp
        return x
```

**通道注意力**（reduction_ratio=16）:
```
AvgPool(x) ──┐
             ├──> 共享2层MLP(C→C/16→C) ──> 相加 ──> Sigmoid ──> Channel Weights
MaxPool(x) ──┘
```

**空间注意力**:
```
Mean(x, dim=1) ──┐
                 ├──> Conv 7×7 (2→1) ──> Sigmoid ──> Spatial Weights
Max(x, dim=1)  ──┘
```

### 3. FNO-CBAM Block（每层精确顺序）

`forward()` 中每个 block 的确切执行顺序如下（共 `depth=6` 层）：

```python
for i in range(depth):
    residual = x
    x1 = self.convs[i](x)      # SpectralConv2d_fast
    x2 = self.ws[i](x)         # Conv2d 1×1
    x  = x1 + x2
    x  = self.cbams[i](x)      # CBAM_Block
    x  = self.norms[i](x)      # LayerNorm([width, H, W])
    x  = residual + x          # 残差相加
    x  = F.gelu(x)             # GELU 激活
```

关键点（与旧版说明的区别）：
- LayerNorm 位于 CBAM 之后、残差相加之前，作用于形状 `[width, H, W]`。
- 残差相加发生在 GELU **之前**：`x = residual + x` 然后 `x = GELU(x)`，而不是对激活后的结果再加残差。

---

## 模型配置

### 当前使用配置

```python
FNO_CBAM_SST_Temporal(
    out_size=(451, 351),       # 输出尺寸 (H, W)
    modes1=80,                 # 傅里叶模式 (纬度)
    modes2=64,                 # 傅里叶模式 (经度)
    width=64,                  # 隐藏层宽度
    depth=6,                   # FNO-CBAM 块数量
    cbam_reduction_ratio=16    # CBAM 通道注意力压缩比
)
```

> 注：类定义中的默认值为 `modes1=32, modes2=32`，但实际训练/推理均以 `modes1=80, modes2=64` 显式传入（见 `scripts/inference/batch/infer_jaxa_full.py` 的 `load_model`）。

### 模型参数量（约数）

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Dual Lifting (sst_encoder + mask_encoder) | ~2K | 两个 Linear(30→32) |
| SpectralConv × 6 | ~503M | 每层两个 `[64,64,80,64,2]` 权重张量，主导参数量 |
| LocalConv (Conv2d 1×1) × 6 | ~25K | Conv2d(64→64, 1×1) |
| CBAM × 6 | ~4K | 通道 MLP(64↔4) + 空间 Conv 7×7 |
| LayerNorm × 6 | ~122M | 每层 weight+bias 形状 `[64,451,351]` |
| Decoder (fc1 + fc2) | ~8.5K | Linear(64→128) + Linear(128→1) |
| **总计** | **~625M** | 由 SpectralConv 与 LayerNorm 主导 |

> 参数量以当前配置 `(451,351)/modes 80×64/width 64/depth 6` 估算。SpectralConv 频域权重和作用于全分辨率的 LayerNorm 是两大主要贡献项。

---

## 输入输出规格

### 输入格式

```python
# sst_seq: [B, 30, H, W]
# - B: batch size
# - 30: 时间步 (Day 1 到 Day 30)
# - H, W: 空间维度 (451, 351)
# - 值: 归一化后的 SST（(K - mean) / std）
# - 缺失/陆地区域在预处理中已填充（如 KNN 空间填充）

# mask_seq: [B, 30, H, W]
# - 1: 缺失/需预测
# - 0: 有效观测
# - 注意: forward() 内部执行 (1 - mask_seq)，
#         使编码器实际看到的是 "观测=1, 缺失=0"

# 两个输入分别经 sst_encoder / mask_encoder 提升到 width//2=32，
# 再在通道维拼接为 width=64（并非把 60 通道用单个 Conv2d 一次性提升）
```

### 输出格式

```python
# 输出: [B, 1, H, W]
# - 第 30 天的 SST 重建（归一化空间）
# - 反归一化: sst_kelvin = output * std + mean
```

---

## Output Composition（推理时的场合成）

见 `scripts/inference/batch/infer_jaxa_full.py`。推理阶段并非简单的
`final = sst*(1-mask) + pred*mask`，而是"保留真实观测 + 只平滑重建区"的两步流程：

```python
# 1) 场合成：观测像素保留真值(KNN/合成场)，云/缺失像素用模型预测
sst_model = np.where(orig_miss == 1, pred_k, knn_k)
sst_model = np.where(land == 1, np.nan, sst_model)   # 陆地置 NaN

# 2) 高斯平滑：只在"非观测"(重建)区域写回滤波值，真实观测永不被平滑
not_obs   = (obs_all[t] == 0)                         # 当前时刻真实观测=0 的像素
sst_model = apply_gaussian(sst_model, land, fill_region=not_obs, sigma=1.0)
```

`apply_gaussian` 的行为：
- 先把 NaN（陆地/无效）临时填成场均值，对整场做 `gaussian_filter`（σ=1.0），
  使云区能借助真实观测邻居得到平滑过渡；
- 但滤波结果**只写回 `fill_region==1`（即非观测/重建）像素**，真实观测像素保持原值不变，陆地保持 NaN。

**意义**:
- 当前时刻的真实观测被**逐像素精确保留**，绝不被滤波或预测覆盖；
- 模型只负责填充云/缺失区域，时间回填与云区填充等"合成像素"再经高斯平滑；
- 观测区与重建区之间因此获得连续、无接缝的过渡，同时保证真值完整。

---

## 与其他方法对比

| 方法 | 全局建模 | 注意力 | 时序建模 | 参数量 |
|------|---------|--------|---------|--------|
| CNN | ✗ | ✗ | ✗ | ~10M |
| U-Net | △ | ✗ | ✗ | ~30M |
| FNO | ✓ | ✗ | ✗ | ~200M |
| Transformer | ✓ | ✓ | ✓ | ~300M |
| **FNO-CBAM** | **✓** | **✓** | **✓** | **~625M** |

---

## 代码位置

```
models/fno_cbam_temporal.py
├── SpectralConv2d_fast      # 傅里叶卷积层 (rfft2 → 频域权重 → irfft2)
├── CBAM_Block               # 通道 + 空间串行注意力
└── FNO_CBAM_SST_Temporal    # 完整模型 (双通道提升 + 6×FNO-CBAM + MLP解码)

scripts/inference/batch/infer_jaxa_full.py
├── apply_gaussian           # 只平滑非观测区的高斯滤波 (σ=1.0)
└── run_hour                 # 场合成 (观测保留 + 云区模型填充) + 保存
```
