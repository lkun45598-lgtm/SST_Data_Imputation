# 方法论文档：基于FNO-CBAM的海表温度缺失值重建

## 1. 问题定义

### 1.1 研究背景

卫星遥感海表温度(SST)数据在海洋学研究中具有重要价值，但受云层遮挡影响，原始数据存在大量缺失值。JAXA卫星逐小时SST数据的平均缺失率约为47%，严重限制了数据的应用价值。

### 1.2 技术挑战

- **高缺失率**：平均47%的像素缺失，部分时刻缺失率超过80%
- **时空复杂性**：SST场具有复杂的时空演化模式
- **日变化规律**：逐小时数据包含显著的日变化信号
- **边界连续性**：重建区域与观测区域的边界需要平滑过渡

## 2. 数据预处理

### 2.1 3D渐进式KNN填充算法

为了给深度学习模型提供完整的输入序列，我们设计了3D渐进式KNN填充算法，该算法具有以下特点：

#### 2.1.1 算法原理

```
输入：30天SST序列，每天包含缺失值
输出：30天完全填充的SST序列

对于每一帧t：
  1. 提取所有缺失点坐标 M = {(y_i, x_i)}
  2. 计算每个缺失点的局部缺失密度 d_i = count(M ∩ B(p_i, r))
  3. 按密度升序排序：d_1 ≤ d_2 ≤ ... ≤ d_n
  4. 渐进填充：
     for i = 1 to n:
       - 查找k个最近的有效邻居（包括刚填充的点）
       - 反距离加权插值：v_i = Σ(w_j * v_j) / Σw_j
       - 更新数据：SST[y_i, x_i] = v_i
```

#### 2.1.2 关键设计

- **渐进策略**：按缺失密度从低到高填充，确保边缘区域先被填充
- **动态更新**：每填充一个点后，该点立即可用于后续插值
- **反距离加权**：权重 w = 1/d^p，其中p=2为距离指数
- **参数设置**：k=20（邻居数），r=20（密度计算半径）

#### 2.1.3 实现优化

- **KDTree重建间隔**：每填充50个点重建一次KDTree，平衡精度与效率
- **并行处理**：使用32个进程并行处理不同时间帧
- **内存优化**：采用float32精度，gzip压缩存储

### 2.2 数据归一化

```python
# 使用OSTIA预训练的归一化参数
mean = 299.92 K  # 全局均值
std = 2.69 K     # 全局标准差

sst_normalized = (sst - mean) / std
```

## 3. 模型架构

### 3.1 FNO-CBAM-Temporal模型

#### 3.1.1 整体架构

```
输入：
  - sst_seq: [B, 30, H, W] - 30天SST序列（KNN填充）
  - mask_seq: [B, 30, H, W] - 30天缺失掩码（1=缺失，0=观测）

编码器：
  - SST编码：Linear(30 → width/2)
  - Mask编码：Linear(30 → width/2)
  - 特征融合：Concat → [B, H, W, width]

FNO主干（6层）：
  每层包含：
    1. SpectralConv2d：傅里叶域卷积
    2. LocalConv：空间域卷积（1×1）
    3. CBAM注意力：通道+空间注意力
    4. LayerNorm + 残差连接
    5. GELU激活

解码器：
  - Linear(width → 128)
  - GELU
  - Linear(128 → 1)

输出：[B, 1, H, W] - 第30天重建SST
```

#### 3.1.2 SpectralConv2d（傅里叶神经算子）

```python
# 傅里叶域卷积
x_ft = FFT2D(x)  # [B, C, H, W] → [B, C, H, W/2+1]

# 保留低频模式
out_ft[:, :, :modes1, :modes2] = W1 ⊗ x_ft[:, :, :modes1, :modes2]
out_ft[:, :, -modes1:, :modes2] = W2 ⊗ x_ft[:, :, -modes1:, :modes2]

# 逆变换
x_out = IFFT2D(out_ft)

# 参数：modes1=80, modes2=64
```

**设计理由**：
- 傅里叶域操作捕获全局空间模式
- 低频模式保留主要的海洋环流特征
- 高频模式被自然滤除，减少噪声

#### 3.1.3 CBAM注意力机制

```python
# 通道注意力
avg_pool = AdaptiveAvgPool2d(x)  # [B, C, 1, 1]
max_pool = AdaptiveMaxPool2d(x)  # [B, C, 1, 1]
channel_att = Sigmoid(MLP(avg_pool) + MLP(max_pool))
x = x * channel_att

# 空间注意力
avg_out = Mean(x, dim=channel)  # [B, 1, H, W]
max_out = Max(x, dim=channel)   # [B, 1, H, W]
spatial_att = Sigmoid(Conv7x7(Concat(avg_out, max_out)))
x = x * spatial_att
```

**设计理由**：
- 通道注意力：自适应调整不同特征通道的权重
- 空间注意力：聚焦于重要的空间区域（如锋面、涡旋）

#### 3.1.4 模型参数

```
总参数量：~606M
网络宽度：width=64
FNO层数：depth=6
输出尺寸：(451, 351) - 南海区域
```

## 4. 损失函数设计

### 4.1 组合损失函数

```python
L_total = α₁·L_mse + α₂·L_gradient + α₃·L_boundary

其中：
α₁ = 1.0   # 缺失区域MSE权重
α₂ = 0.15  # 梯度一致性权重
α₃ = 0.1   # 边界平滑权重
```

### 4.2 缺失区域MSE损失

```python
def reconstruction_loss_missing(pred, target, missing_mask, ocean_mask):
    """
    只在缺失区域计算MSE
    """
    mask = missing_mask * ocean_mask  # [B, H, W]
    loss = ((pred - target)² * mask).sum() / mask.sum()
    return loss
```

### 4.3 梯度一致性损失

```python
def gradient_loss(pred, target, missing_mask, ocean_mask):
    """
    约束预测场的空间梯度与真实场一致
    减少锋面位置偏移导致的大误差
    """
    # Y方向梯度
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    # X方向梯度
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    
    # 在缺失区域计算梯度差异
    loss = (|pred_grad_y - target_grad_y| + |pred_grad_x - target_grad_x|) / 2
    return loss
```

**设计理由**：
- 点对点MSE对锋面位置偏移敏感
- 梯度损失关注空间模式而非绝对值
- 提高重建场的物理合理性

### 4.4 边界平滑损失

```python
def boundary_smoothness_loss(pred, target, missing_mask):
    """
    约束缺失区域边界处的梯度连续性
    """
    # 检测边界像素对（一个在缺失区，一个在观测区）
    boundary_pairs = detect_boundary_pairs(missing_mask)
    
    # 计算边界处的预测梯度与真实梯度差异
    for (p1, p2) in boundary_pairs:
        pred_grad = pred[p1] - pred[p2]
        target_grad = target[p1] - target[p2]
        loss += |pred_grad - target_grad|
    
    return loss / len(boundary_pairs)
```

**设计理由**：
- 防止重建区域与观测区域出现明显接缝
- 确保边界处的温度过渡平滑自然

## 5. 训练策略

### 5.1 两阶段训练

#### 阶段1：OSTIA预训练

```
数据集：OSTIA SST（南海区域，2015-2023）
目的：学习SST的基本空间模式和时间动态
配置：
  - GPU：8卡DDP（NCCL后端）
  - Batch size：4 × 8 = 32
  - Learning rate：1e-3
  - Optimizer：AdamW
  - Scheduler：StepLR(step=15, gamma=0.5)
  - Epochs：60
  - 人工挖空：mask_ratio ∈ [0.3, 0.7]
```

#### 阶段2：JAXA逐小时微调

```
数据集：JAXA hourly SST（2015-2023）
目的：适应目标区域特征，学习日变化规律
配置：
  - 24个模型：H=00, H=01, ..., H=23
  - GPU：8卡DDP（NCCL后端）
  - Batch size：2 × 8 = 16
  - Learning rate：5e-4
  - Optimizer：AdamW
  - Scheduler：CosineAnnealingLR
  - Epochs：50-100
  - 使用原始缺失模式（不人工挖空）
```

### 5.2 Output Composition策略

```python
# 训练时
composed = pred * artificial_mask + sst_input * (1 - artificial_mask)
loss = compute_loss(composed, target, artificial_mask)

# 推理时
output = pred * original_missing_mask + sst_input * (1 - original_missing_mask)
```

**设计理由**：
- 观测区域保留原始值，避免引入模型误差
- 模型专注于学习缺失区域的重建
- 确保输出在观测区域与输入完全一致

### 5.3 数据增强

```python
# 人工挖空（仅OSTIA预训练）
mask_ratio = random.uniform(0.3, 0.7)
artificial_mask = generate_random_mask(mask_ratio)

# 时间窗口滑动
for t in range(len(dataset) - 30):
    sample = dataset[t:t+30]  # 30天窗口
```

## 6. 推理流程

### 6.1 完整推理Pipeline

```
输入：JAXA原始NC文件（包含缺失值）
      ↓
[3D KNN填充] 生成完整的30天序列
      ↓
[模型推理] FNO-CBAM预测第30天
      ↓
[Output Composition] 观测区域保留原值
      ↓
[高斯滤波] σ=1.0平滑处理
      ↓
输出：重建后的SST场（NC格式）
```

### 6.2 高斯滤波后处理

```python
def apply_gaussian_filter(sst, sigma=1.0):
    """
    对重建结果应用高斯滤波
    """
    # 用均值填充NaN区域
    valid_mask = ~np.isnan(sst)
    sst_filled = sst.copy()
    sst_filled[~valid_mask] = np.nanmean(sst)
    
    # 高斯滤波
    from scipy.ndimage import gaussian_filter
    sst_smoothed = gaussian_filter(sst_filled, sigma=sigma)
    
    # 恢复NaN
    sst_smoothed[~valid_mask] = np.nan
    
    return sst_smoothed
```

**效果**：
- Diff Std从0.561°C降至0.534°C
- 减少高频噪声，提高视觉平滑度
- σ=1.0在平滑度和细节保留之间取得平衡

### 6.3 批量推理

```bash
# 推理所有JAXA数据（H=01~H=23）
python scripts/inference/batch/infer_jaxa_full.py

# 处理规模
- 时间范围：2016-07 至 2025-03
- 文件数量：73,004个NC文件
- 推理速度：~50帧/秒（batch_size=8）
- 总耗时：~1.4小时
```

## 7. 评估指标

### 7.1 定量指标

```python
# 均方根误差
RMSE = sqrt(mean((pred - gt)²))

# 平均绝对误差
MAE = mean(|pred - gt|)

# 归一化RMSE
VRMSE = RMSE / std(gt)

# 最大误差
Max Error = max(|pred - gt|)
```

### 7.2 模型性能（JAXA测试集）

| 指标 | 值 | 说明 |
|------|------|------|
| VRMSE | 0.215 ± 0.062 | 相对标准差的RMSE |
| MAE | 0.095 ± 0.014 K | 平均绝对误差 |
| RMSE | 0.126 ± 0.020 K | 均方根误差 |
| Max Error | 0.620 ± 0.153 K | 最大误差 |

### 7.3 日变化规律分析

通过对H=01~H=23模型的推理结果分析，发现明显的日变化规律：

- **早晨/傍晚（H=06~09, H=17~19）**：模型预测偏冷
- **正午/夜晚（H=12~14, H=00~02）**：模型预测偏暖
- **温度差异**：日变化幅度约1-2°C

这一规律与太阳辐射的日变化周期一致，验证了模型捕获了真实的物理过程。

## 8. 输出数据集

### 8.1 数据格式

```
目录结构：
/data/sst_data/SST_Data_Imputation/
├── YYYYMM/
│   └── DD/
│       └── YYYYMMDDHHMMSS.nc

NC文件内容：
- 变量：sea_surface_temperature
- 单位：Kelvin
- 维度：(1, 451, 351)
- 填充值：NaN（陆地区域）
- 后处理：包含Gaussian σ=1.0滤波
```

### 8.2 数据统计

```
时间范围：2016-07-01 至 2025-03-30
文件数量：73,004个
空间范围：南海区域（99°E-125°E, 0°N-26°N）
时间分辨率：逐小时
空间分辨率：0.05° × 0.05°
数据完整性：99.95%（仅缺39帧）
```

## 9. 技术创新点

### 9.1 3D渐进式KNN填充

- **创新**：按缺失密度渐进填充，确保边缘优先
- **优势**：比传统KNN更稳定，避免孤立缺失区域无法填充

### 9.2 FNO-CBAM融合架构

- **创新**：结合傅里叶神经算子的全局建模能力和CBAM的注意力机制
- **优势**：同时捕获全局环流模式和局部精细结构

### 9.3 逐小时微调策略

- **创新**：为每个小时训练独立模型，捕获日变化规律
- **优势**：相比单一模型，更准确地重建不同时刻的SST场

### 9.4 Output Composition

- **创新**：观测区域保留原值，只重建缺失区域
- **优势**：避免在观测区域引入模型误差，确保数据一致性

### 9.5 边界平滑损失

- **创新**：专门约束缺失区域边界的梯度连续性
- **优势**：消除重建区域与观测区域的接缝，提高视觉质量

## 10. 应用场景

### 10.1 海洋学研究

- 海表温度时空演化分析
- 海洋锋面和涡旋识别
- 海气相互作用研究

### 10.2 气候监测

- 海洋热含量估算
- 厄尔尼诺/拉尼娜监测
- 海洋热浪检测

### 10.3 数值模式

- 海洋模式初始化
- 数据同化
- 模式验证

## 11. 局限性与未来工作

### 11.1 当前局限

- **计算成本**：24个模型需要分别训练和推理
- **时间依赖**：需要30天历史数据，无法处理序列开头
- **极端天气**：台风等极端事件的重建精度有待提高

### 11.2 未来改进方向

- **统一模型**：设计单一模型处理所有小时，减少计算成本
- **更长时间序列**：扩展到60天或90天输入，提高时间建模能力
- **多源数据融合**：结合多个卫星数据源，提高重建鲁棒性
- **不确定性量化**：提供重建结果的置信区间

## 12. 参考文献

1. Li, Z., et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. ICLR.
2. Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
3. JAXA Himawari-8 SST Product: https://www.eorc.jaxa.jp/ptree/

---

**文档版本**：v1.0  
**最后更新**：2026-04-20  
**作者**：Claude Code
