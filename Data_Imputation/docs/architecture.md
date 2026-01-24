# 模型架构说明

## 概述

FNO-CBAM-Temporal 是一个结合傅里叶神经算子 (FNO) 和卷积注意力模块 (CBAM) 的时序SST重建模型。

---

## 整体架构

```
输入: [B, 60, H, W]
      ├─ 30 channels: SST序列 (Day 1-30)
      └─ 30 channels: Mask序列 (Day 1-30)
          │
          ▼
┌─────────────────────────────┐
│  Lifting Layer              │
│  Conv2d(60 → 64)           │
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  FNO-CBAM Block × 6         │
│  ├─ Spectral Conv           │
│  ├─ Local Conv              │
│  ├─ CBAM Attention          │
│  └─ GELU + Residual         │
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  Projection Layer           │
│  Conv2d(64 → 64 → 1)       │
└─────────────────────────────┘
          │
          ▼
输出: [B, 1, H, W] (Day 30 SST预测)
```

---

## 核心组件

### 1. Spectral Convolution (傅里叶卷积)

傅里叶神经算子的核心，在频域进行全局卷积：

```python
class SpectralConv2d(nn.Module):
    def forward(self, x):
        # 1. FFT变换到频域
        x_ft = torch.fft.rfft2(x)

        # 2. 频域权重相乘 (modes1 × modes2)
        out_ft = torch.einsum("bixy,ioxy->boxy", x_ft[:,:,:modes1,:modes2], weights)

        # 3. 逆FFT回到空间域
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x
```

**优势**:
- 全局感受野，捕获大尺度空间模式
- 计算复杂度 O(N log N)，比CNN更高效
- 天然适合处理周期性/波动性数据

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| modes1 | 80 | 纬度方向保留的傅里叶模式数 |
| modes2 | 64 | 经度方向保留的傅里叶模式数 |

### 2. CBAM (Convolutional Block Attention Module)

双重注意力机制，同时关注通道和空间：

```python
class CBAM(nn.Module):
    def forward(self, x):
        # 1. 通道注意力
        channel_att = self.channel_attention(x)  # [B, C, 1, 1]
        x = x * channel_att

        # 2. 空间注意力
        spatial_att = self.spatial_attention(x)  # [B, 1, H, W]
        x = x * spatial_att

        return x
```

**通道注意力**:
```
MaxPool(x) ──┐
             ├──> MLP ──> Sigmoid ──> Channel Weights
AvgPool(x) ──┘
```

**空间注意力**:
```
MaxPool(x, dim=1) ──┐
                    ├──> Conv7×7 ──> Sigmoid ──> Spatial Weights
AvgPool(x, dim=1) ──┘
```

### 3. FNO-CBAM Block

单个块的完整结构：

```python
class FNO_CBAM_Block(nn.Module):
    def forward(self, x):
        # 频域路径
        x1 = self.spectral_conv(x)

        # 空间域路径
        x2 = self.local_conv(x)  # Conv1×1

        # 合并
        x = x1 + x2

        # 注意力
        x = self.cbam(x)

        # 激活 + 残差
        x = F.gelu(x) + residual

        return x
```

---

## 模型配置

### 当前使用配置

```python
FNO_CBAM_SST_Temporal(
    out_size=(451, 351),   # 输出尺寸 (H, W)
    modes1=80,             # 傅里叶模式 (纬度)
    modes2=64,             # 傅里叶模式 (经度)
    width=64,              # 隐藏层宽度
    depth=6                # FNO-CBAM块数量
)
```

### 模型参数量

| 组件 | 参数量 |
|------|--------|
| Lifting | 3.9K |
| SpectralConv × 6 | ~600M |
| LocalConv × 6 | 24.6K |
| CBAM × 6 | 16.8K |
| Projection | 4.2K |
| **总计** | **~606M** |

### 显存占用

| 操作 | 显存 (bs=2) |
|------|------------|
| 前向传播 | ~10GB |
| 反向传播 | ~25GB |
| 推理 | ~8GB |

---

## 输入输出规格

### 输入格式

```python
# SST序列: [B, 30, H, W]
# - B: batch size
# - 30: 时间步 (Day 1 到 Day 30)
# - H, W: 空间维度 (451, 351)
# - 值: 归一化后的SST
# - 缺失区域填充方式:
#   - OSTIA: 最近邻插值填充
#   - JAXA: KNN空间填充

# Mask序列: [B, 30, H, W]
# - 1: 缺失/需预测
# - 0: 有效观测

# 模型输入: [B, 60, H, W]
# 前30通道: SST序列
# 后30通道: Mask序列
```

### 输出格式

```python
# 输出: [B, 1, H, W]
# - 第30天的SST预测
# - 值: 归一化后的SST
# - 需反归一化: sst_kelvin = output * std + mean
```

---

## Output Composition

模型推理后应用Output Composition：

```python
def output_composition(pred, sst_input, mask):
    """
    观测区域保留输入值，缺失区域使用预测值

    pred: [B, 1, H, W] 模型预测
    sst_input: [B, 1, H, W] 第30天输入SST
    mask: [B, 1, H, W] 第30天mask (1=缺失)
    """
    final = sst_input * (1 - mask) + pred * mask
    return final
```

**意义**:
- 保证观测区域完全准确
- 模型只需学习填充缺失区域
- 避免模型破坏已有观测

---

## 与其他方法对比

| 方法 | 全局建模 | 注意力 | 时序建模 | 参数量 |
|------|---------|--------|---------|--------|
| CNN | ✗ | ✗ | ✗ | ~10M |
| U-Net | △ | ✗ | ✗ | ~30M |
| FNO | ✓ | ✗ | ✗ | ~200M |
| Transformer | ✓ | ✓ | ✓ | ~300M |
| **FNO-CBAM** | **✓** | **✓** | **✓** | **~606M** |

---

## 代码位置

```
models/fno_cbam_temporal.py
├── SpectralConv2d          # 傅里叶卷积层
├── CBAM                    # 注意力模块
├── FNO_CBAM_Block          # 单个FNO-CBAM块
└── FNO_CBAM_SST_Temporal   # 完整模型
```
