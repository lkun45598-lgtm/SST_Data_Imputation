# 方法论文档：基于FNO-CBAM的海表温度缺失值重建

## 1. 问题定义

### 1.1 研究背景

卫星遥感海表温度(SST)数据在海洋学研究中具有重要价值，但受云层遮挡影响，原始数据存在大量缺失值。本项目针对 JAXA Himawari 静止卫星的逐小时 L3 SST 数据，在**中国南海北部**海域开展缺失值重建。受云覆盖影响，海洋像素的原始缺失率通常在 50% 以上，部分时刻超过 80%，严重限制了逐小时数据的应用价值。

### 1.2 研究区域与网格

- **空间范围**：南海北部，纬度 15.0°N–24.0°N，经度 111.0°E–118.0°E
- **网格尺寸**：451（纬度）× 351（经度），共 158,301 个像素
- **空间分辨率**：0.02°（约 2.2 km）
- **纬度方向**：数组按纬度从北到南递减存储（24.0°N → 15.0°N，步长 −0.02°）
- **陆地占比**：约 16.6% 的网格为陆地，重建只针对海洋像素

> 注：全流程的所有脚本（预处理、模型、训练、推理）均以 `out_size=(451, 351)` 为固定输出尺寸，与该研究网格严格对应。

### 1.3 技术挑战

- **高缺失率**：海洋像素原始缺失率常达 50%–70%，部分时刻更高
- **时空复杂性**：SST 场具有复杂的时空演化模式
- **日变化规律**：逐小时数据包含显著的日变化信号，不同小时的温度特征差异明显
- **边界连续性**：重建区域与真实观测区域的边界需要平滑过渡，避免接缝

## 2. 数据预处理Pipeline

数据预处理是本项目的核心工作之一，采用**三阶段**处理流程，将原始 JAXA 逐小时观测（含大量云缺失）转换为完全填充、可直接用于模型的逐日序列。对每个目标小时 h（0–23）与每个年度序列，执行完全相同的三阶段处理，输出统一格式的 `jaxa_knn_filled_XX.h5`。

三阶段依次为：
1. **时间加权填充**（Temporal Weighted Filling）：利用历史同小时观测填充部分缺失；
2. **高斯低通滤波**（Gaussian Low-Pass）：仅平滑时间填充像素，保留真实观测不变；
3. **3D 因果渐进式 KNN 填充**（3D Causal Progressive KNN）：在时空域填满剩余缺失。

### 2.1 阶段1：时间加权填充（Temporal Weighted Filling）

#### 2.1.1 算法原理

时间加权填充利用**历史同小时**观测数据，通过反时间距离加权平均填充当前缺失像素。逐小时 SST 具有显著日变化，因此只使用与目标帧**相同小时**、发生在此前若干天的观测。

```python
算法：时间加权填充
输入：目标时刻 t（某年度序列中，固定小时 h 的某一天）的 SST 场（含缺失）
输出：部分填充的 SST 场

1. 对目标帧中的每个缺失像素 (y, x)：

2. 向前查找历史同小时观测：
   - 回溯窗口：LOOKBACK_WINDOW = 48 小时
   - max_lookback_days = 48 // 24 + 1 = 3（最多回溯 3 个此前的同小时帧）
   - 例如：填充 2020-01-15 03:00 的缺失值
     → 查找 2020-01-14 03:00, 2020-01-13 03:00, 2020-01-12 03:00

3. 计算时间权重（以“天数距离”d 为单位）：
   weight(d) = 1 / d
   - 1 天前的同小时观测：weight = 1/1 = 1.0
   - 2 天前：weight = 1/2 = 0.5
   - 3 天前：weight = 1/3 ≈ 0.33

4. 加权平均填充：
   filled_value = Σ(w_i × v_i) / Σ w_i
```

#### 2.1.2 关键设计

**为什么使用同小时历史数据？**
- 逐小时 SST 具有显著的日变化规律；
- 同一小时的历史帧具有相似的太阳辐射条件与热力状态；
- 避免混合不同时刻的温度特征造成系统性偏差。

**为什么使用反时间距离加权？**
- 近期观测更能反映当前海洋状态；
- 权重随天数距离衰减，符合海洋的时间记忆特性；
- 计算高效、无需额外参数。

**效果**：
- 该阶段仅能填充“近几天同小时有观测”的像素；
- 以 h=00、series 00 为例，约 **32%** 的网格像素由该阶段填充（记为 `temporal_fill_mask`）；
- 真实观测像素始终保持不变。

#### 2.1.3 实现细节

```python
def temporal_weighted_fill_frame(target_time, start_time, hour_offset):
    """对单帧做时间加权填充：向前看 48h 内同小时的历史帧，用 1/dt 加权平均。"""
    target_sst, lat, lon = load_jaxa_frame(target_time)
    filled_sst = target_sst.copy()
    missing_mask = np.isnan(target_sst)

    # 收集历史帧（最多回溯 48//24+1 = 3 个同小时帧，不越过序列起点）
    max_lookback_days = LOOKBACK_WINDOW // 24 + 1     # = 3
    history = {}
    for d in range(1, max_lookback_days + 1):
        hist_time = target_time - timedelta(days=d)
        if hist_time < start_time:
            break
        hist_sst, _, _ = load_jaxa_frame(hist_time)
        if hist_sst is not None:
            history[d] = hist_sst                     # d = 天数距离

    # 对每个缺失像素做反距离加权
    for y, x in zip(*np.where(missing_mask)):
        weights, values = [], []
        for d, hist_sst in history.items():
            if not np.isnan(hist_sst[y, x]):
                weights.append(1.0 / d)               # 反距离权重
                values.append(hist_sst[y, x])
        if weights:
            filled_sst[y, x] = np.sum(np.array(weights) * np.array(values)) / np.sum(weights)
    return filled_sst
```

处理产物：该阶段同时记录 `fill_mask`（被时间填充的像素）与阶段后仍缺失的掩码 `missing_mask`，供后续阶段使用。

### 2.2 阶段2：高斯低通滤波（Gaussian Low-Pass Filter）

#### 2.2.1 算法原理

高斯低通滤波用于平滑阶段 1 时间填充所引入的空间不连续与高频噪声。**关键约束**：滤波结果只写回**时间填充像素**（`fill_mask == 1`），而**真实观测像素与仍缺失像素保持不变**。高斯卷积的输入仍然使用真实观测值，使填充区能够借助邻近真值被平滑，但滤波结果**绝不覆盖真实观测**。

```python
算法：高斯低通滤波（仅平滑时间填充像素）
输入：阶段 1 输出的 SST 场、缺失掩码 missing_mask、时间填充掩码 fill_mask
输出：滤波后的 SST 场

1. 有效区域（观测 + 时间填充）：valid = ~isnan(data) & (missing_mask == 0)
2. 临时填充：temp = data.copy(); temp[~valid] = nanmean(data)  # 仅为卷积提供背景
3. 高斯卷积：filtered = gaussian_filter(temp, sigma=1.5)
4. 写回：只在 write = valid & (fill_mask == 1) 处使用 filtered
         其余像素（真实观测、缺失）保留原值
```

#### 2.2.2 参数选择

**σ = 1.5 的设计理由**：
- σ 过小（<1.0）：平滑不足，填充区仍显斑驳；
- σ 过大（>2.0）：过度平滑，抹去中尺度特征（如锋面、涡旋）；
- σ = 1.5：在去噪与保留细节之间取得平衡，对应高斯核约 ±3σ ≈ 4–5 像素的有效作用范围。

> 说明：本阶段使用的是各向同性**高斯低通滤波**，而非 Butterworth 或固定截止频率的频域滤波器；脚本同时提供中值/均值/双边滤波作为对照选项，但正式流程采用高斯滤波。

#### 2.2.3 实现细节

```python
def gaussian_filter_frame(data, missing_mask, fill_mask, sigma=1.5):
    valid_mask = ~np.isnan(data) & (missing_mask == 0)
    if valid_mask.sum() == 0:
        return data
    data_for_filter = data.copy()
    data_for_filter[~valid_mask] = np.nanmean(data)   # 卷积背景
    filtered = ndimage.gaussian_filter(data_for_filter, sigma=sigma)
    # 只在“时间填充像素”处写回滤波值；真实观测/缺失保持原值
    write = valid_mask & (fill_mask == 1)
    return np.where(write, filtered, data)
```

**效果**：
- 时间填充区更平滑、空间连续性更好；
- 真实观测像素逐位保持不变，不引入任何滤波偏差；
- 缺失像素在本阶段仍为缺失，交由阶段 3 处理。

### 2.3 阶段3：3D 因果渐进式 KNN 填充（3D Causal Progressive KNN）

#### 2.3.1 算法原理

阶段 3 是本项目预处理的核心算法。它在**时空 3D 域**进行 KNN 插值，替换了早期的逐帧 2D 版本。核心特点：

1. **因果约束**：只使用过去帧（t' < t），不使用未来帧；
2. **双层 KDTree**：Tier 1 为历史帧树（t−6…t−1，已完全填充，静态）；Tier 2 为当前帧树（随填充渐进更新）；
3. **渐进填充**：当前帧内按 2D 缺失密度升序处理，边缘（密度低）先填、中心（密度高）后填；
4. **偏差校正**：把历史帧的取值平移到当前帧的温度水平，抑制日间/帧间基线漂移。

```python
算法：3D 因果渐进式 KNN 填充
输入：阶段 2 输出的 T 帧序列（缺失处为 NaN）、missing_masks、land_mask
参数：k=20, radius=20, power=2, time_weight=0.9, space_weight=1.0,
      window_size=7（天）, batch_size=500
输出：海洋区域完全填充的 T 帧序列

预处理：逐帧计算观测区海洋均值 frame_obs_mean[t]（用于偏差校正）

外层：for t in 0..T-1（因果，按时间顺序）
  1. 当前帧海洋缺失像素坐标 M；按 2D 缺失密度升序排序（渐进顺序）

  2. Tier 1（历史帧）：收集 t_start=max(0, t-6) .. t-1 各帧已填充的海洋像素
     - 时空坐标 = (t_offset × time_weight, y × space_weight, x × space_weight)
       其中 t_offset = t - t_past ∈ {1..6}
     - 偏差校正后的值 = filled[t_past] + (curr_mean - past_mean)
     - 构建 past_tree = cKDTree(历史时空坐标)

  3. Tier 2（当前帧）：对当前帧已有效像素（原始有效 + 已填充）建 curr_tree
     - 时空坐标 = (0, y × space_weight, x × space_weight)（t_offset=0）
     - 每处理一个 batch 后若有新填充像素，则标记重建 curr_tree

  4. 分 batch 查询：
     - 对 batch 内查询点，分别在 past_tree、curr_tree 各取最近 k 个
     - 合并两层结果，按时空距离取全局最近的 k 个
     - IDW 插值：w = 1 / (d^power + ε)，filled = Σ(w·v) / Σw

  5. 写回填充值，更新当前帧有效像素集合

兜底：残余 NaN 用全局海洋均值填充；陆地保持 NaN
```

#### 2.3.2 关键设计

**为什么强制因果（只用过去帧）？**
- 与推理时的使用方式一致（推理时未来帧不可得）；
- 避免未来信息泄漏到填充结果，保证时间一致性。

**为什么按缺失密度升序渐进填充？**
- 边缘像素周围有更多真实/已填充值，插值更可靠；
- 逐步向中心推进，避免大面积缺失中心因缺乏邻居而无法填充。

**为什么做偏差校正？**
- 历史帧与当前帧可能存在整体温度水平差异（日变化、天气过程）；
- 用 `curr_mean − past_mean` 将历史值平移到当前基线，避免把过去偏冷/偏暖的整体状态引入当前帧。

**为什么使用反距离幂加权（power=2）？**
- 权重 w = 1/d²，距离越近权重越大，强调时空近邻；
- 符合海洋要素的时空自相关特性。

**时空缩放（time_weight=0.9, space_weight=1.0）**：
- 时间维乘以 0.9 使“1 天的时间距离”与约 0.9 个像素的空间距离等价，从而让近邻既包含空间邻居也包含时间邻居；平衡时空贡献。

#### 2.3.3 实现优化

- **批量向量化查询**：以 `batch_size=500` 为单位调用 `cKDTree.query`，避免逐像素 Python 循环；
- **两层结果合并**：将 Tier 1/Tier 2 的距离矩阵拼接后按行取最近 k 个（`np.argsort`），统一做 IDW；
- **渐进重建**：当前帧树在每个 batch 产生新填充后按需重建，使后填充像素能利用先填充结果；
- **并行**：不同小时/序列的预处理相互独立，可跨进程并行。

**效果**：
- 阶段 3 后海洋区域被完全填充（0 缺失）；
- 边缘区域主要借助真实观测，中心区域借助时间邻居与已填充邻居；
- 输出的 `sst_data` 为逐日、完全填充的 30 天可用序列的基础数据。

### 2.4 输出 H5 数据格式

每个（小时 h，序列 sid）产出一个 `jaxa_knn_filled_{sid:02d}.h5`，字段如下（形状以 series 00 为例，T=360）：

| 字段 | 形状 | 含义 |
|------|------|------|
| `sst_data` | (T, 451, 351) float32 | 三阶段后完全填充的 SST（Kelvin），海洋无缺失，陆地为 NaN |
| `land_mask` | (451, 351) uint8 | 1=陆地，0=海洋 |
| `original_obs_mask` | (T, 451, 351) uint8 | 1=**真实观测**像素（唯一的真值来源） |
| `temporal_fill_mask` | (T, 451, 351) uint8 | 1=阶段 1 时间填充像素 |
| `original_missing_mask` | (T, 451, 351) uint8 | 1=阶段 1 之后仍缺失、由阶段 3 KNN 填充的像素 |
| `latitude` | (451,) float32 | 24.0°N → 15.0°N |
| `longitude` | (351,) float32 | 111.0°E → 118.0°E |
| `timestamps` | (T,) S32 | 逐日 ISO 时间戳（固定小时 h） |

**掩码语义（关键）**：三类掩码在海洋上互补——真实观测（`original_obs_mask`）、时间填充（`temporal_fill_mask`）、KNN 填充（`original_missing_mask`）。以 h=00、series 00 为例，三者约占全网格 28% / 32% / 40%（其中 KNN 类包含约 16.6% 的陆地）。**只有 `original_obs_mask` 是真实测量值，其余均为重建值**——这一点决定了后续训练的损失掩码与推理的输出组合策略。

### 2.5 数据归一化

```python
# 使用 OSTIA 预训练时的归一化参数（迁移学习的基础）
mean = 299.9221 K   # 全局均值
std  = 2.6919 K     # 全局标准差
sst_normalized = (sst - mean) / std
```

**为什么使用 OSTIA 的归一化参数？**
- OSTIA 预训练模型已在该归一化空间中学习特征；
- JAXA 微调沿用相同的 mean/std，确保输入分布一致；
- 迁移学习的关键前提：保持输入分布与预训练一致。挖空位置以 `mean` 填充，归一化后恰为 0。


## 3. 模型架构

### 3.1 FNO_CBAM_SST_Temporal 模型

#### 3.1.1 整体架构

模型接收 **30 天** 的 SST 与 mask 双序列（概念上 60 通道：30×SST + 30×mask），输出第 30 天的重建 SST：

```
输入：
  - sst_seq  : [B, 30, H, W]  30 天 SST 序列（归一化空间）
  - mask_seq : [B, 30, H, W]  30 天缺失掩码（1=缺失，0=观测）

双线性 lifting（分别编码后拼接）：
  # 张量先 permute 为 [B, H, W, 30]
  - SST 编码器 :  Linear(30 → width//2 = 32)
  - Mask 编码器:  Linear(30 → width//2 = 32)   # mask 内部先取 (1 - mask)，令“观测=1”
  - 拼接融合  :  concat → [B, H, W, 64] → permute → [B, 64, H, W]

FNO 主干（depth = 6 个残差块），每块顺序为：
    residual = x
    x1 = SpectralConv2d(x)          # 傅里叶域全局卷积
    x2 = Conv2d_1x1(x)              # 空间域 1×1 局部卷积
    x  = x1 + x2
    x  = CBAM(x)                    # 通道→空间 注意力
    x  = LayerNorm(x)               # 对 [C, H, W] 归一化
    x  = residual + x               # 残差相加
    x  = GELU(x)

解码器：
  # permute 回 [B, H, W, 64]
  - Linear(64 → 128) → GELU → Linear(128 → 1)
  # permute → [B, 1, H, W]

输出：[B, 1, H, W] 第 30 天重建 SST（归一化空间）
```

> 说明：lifting 使用**双路 Linear**（沿时间维 30→32），而非 Conv2d(60→64)；输出解码同样使用 Linear(64→128→1)，而非卷积投影。mask 在前向中被反相为“观测=1”后再编码。

#### 3.1.2 SpectralConv2d（傅里叶神经算子）

**核心思想**：在频率域做卷积，天然具有全局感受野。

```python
def forward(self, x):                      # x: [B, C, H, W]
    x_ft = torch.fft.rfft2(x)              # [B, C, H, W//2+1] 复数
    out_ft = torch.zeros(B, C_out, H, W//2+1, dtype=cfloat)

    # 保留两个低频块（沿纬度方向取正、负频率各 modes1 个）
    out_ft[:, :, :modes1,  :modes2] = compl_mul2d(x_ft[:, :, :modes1,  :modes2], weights1)
    out_ft[:, :, -modes1:, :modes2] = compl_mul2d(x_ft[:, :, -modes1:, :modes2], weights2)

    x = torch.fft.irfft2(out_ft, s=(H, W))
    return x
```

- 复数权重通过 `torch.view_as_complex` 由实张量 `(in, out, modes1, modes2, 2)` 构造，用 `einsum("bixy,ioxy->boxy")` 与频谱相乘；
- 保留两个低频块（纬度方向的正、负低频），其余高频自动置零，等价于一种可学习的低通算子；
- **参数设置**：`modes1 = 80`（纬度方向，对应 H=451），`modes2 = 64`（经度方向，对应 W=351）。

**设计理由**：
1. **全局感受野**：适合捕获大尺度环流与锋面；
2. **多尺度建模**：低频保留大中尺度结构，高频（噪声）被抑制；
3. **分辨率鲁棒**：频域算子对输入分辨率相对不敏感。

#### 3.1.3 CBAM 注意力机制（串行：先通道，后空间）

**通道注意力**（共享 MLP，reduction ratio = 16）：

```python
avg = adaptive_avg_pool2d(x, 1).view(B, C)     # [B, C]
mx  = adaptive_max_pool2d(x, 1).view(B, C)     # [B, C]
# 共享两层 MLP：fc1: C → C/16, ReLU, fc2: C/16 → C（均无 bias）
ch  = sigmoid(fc2(relu(fc1(avg))) + fc2(relu(fc1(mx)))).view(B, C, 1, 1)
x   = x * ch
```

**空间注意力**（7×7 卷积）：

```python
avg = mean(x, dim=1, keepdim=True)             # [B, 1, H, W]
mx  = max(x,  dim=1, keepdim=True)             # [B, 1, H, W]
sp  = sigmoid(Conv2d(2→1, kernel=7, padding=3)(concat([avg, mx], dim=1)))
x   = x * sp
```

**设计理由**：
- 通道注意力自适应地强调重要特征通道；
- 空间注意力聚焦重要区域（锋面、涡旋、云边界）；
- 串行组合先决定“看什么”，再决定“看哪里”。

#### 3.1.4 模型参数

```
输出尺寸：(451, 351)  南海北部研究网格
width = 64, depth = 6, modes1 = 80, modes2 = 64, cbam_reduction_ratio = 16

总参数量：约 6.25 亿（≈ 624.9 M）
  - SpectralConv2d（6 层复数谱权重）：约 503.3 M（主导项）
  - LayerNorm（6 层，normalized_shape = [64, 451, 351]）：约 121.6 M
  - CBAM / 1×1 卷积 / 双编码器 / 解码器：合计约 0.03 M（很小）
```

> 谱权重与 LayerNorm 的仿射参数共同构成了模型的绝大部分参数量。

## 4. 损失函数设计

损失在**归一化空间**、且只在指定掩码区域内计算。训练分两阶段（第 5 节），两阶段使用的损失组合不同，本节给出各损失分量的定义。

### 4.1 输出组合（Output Composition）

无论训练还是推理，模型输出都先经过“输出组合”：**观测/非挖空区域直接使用输入值，仅缺失/挖空区域使用模型预测，陆地保持输入**。

```python
def output_composition(pred, sst_seq, mask_seq, land_mask=None):
    last_input = sst_seq[:, -1:, :, :]     # 第 30 天输入
    last_mask  = mask_seq[:, -1:, :, :]    # 第 30 天 mask（1=缺失/挖空）
    composed = last_input * (1 - last_mask) + pred * last_mask
    if land_mask is not None:
        lm = land_mask.unsqueeze(1)
        composed = composed * (1 - lm) + last_input * lm   # 陆地保持输入
    return composed
```

**意义**：保留真值、只重建缺失、简化学习任务、并让重建区与观测区自然衔接（配合边界平滑损失）。

### 4.2 掩码 MSE（重建主损失）

```python
def masked_mse_loss(pred, target, loss_mask):
    mask = loss_mask.unsqueeze(1)                        # [B, 1, H, W]
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
```

- 用乘法而非索引，保证梯度稳定；
- `loss_mask` 的定义随训练阶段而不同（见第 5 节）。

### 4.3 梯度一致性损失

```python
def masked_gradient_loss(pred, target, loss_mask):
    # 在 loss_mask 区域，约束 x/y 方向一阶差分与真值一致（L1）
    ...
    return (loss_x + loss_y) / 2
```

**设计理由**：逐点 MSE 对锋面位置的微小偏移敏感；梯度损失关注空间结构，提升重建场的物理合理性。

### 4.4 边界平滑损失

```python
def boundary_smoothness_loss(composed, target, mask_seq):
    last_mask = mask_seq[:, -1:, :, :]
    # 检测 mask 内/外相邻像素对（跨越缺失区边界）
    # 只在这些边界像素对上约束 composed 的梯度 ≈ 真值梯度
    ...
    return (loss_x + loss_y) / 2
```

**设计理由**：防止重建区与观测区之间出现接缝，使温度过渡平滑自然，提高视觉质量。**该项是部署配置（JAXA 微调）中启用的关键正则项**。

### 4.5 时间连续性损失（曾测试，微调阶段最终未启用）

```python
def temporal_consistency_penalty(pred, target, prev_day, loss_mask):
    expected  = target - prev_day        # 真实的 day30 - day29
    predicted = pred   - prev_day        # 预测的 day30 - day29
    penalty = F.relu(torch.abs(predicted - expected) - 0.5)   # 容忍带 0.5
    return (penalty * loss_mask.unsqueeze(1)).sum() / (loss_mask.sum() + 1e-8)
```

- 约束第 30 天相对第 29 天的变化不偏离真实变化过多；
- 在 OSTIA 预训练中以另一形式（相对历史日间变化统计）参与损失；
- 在 JAXA 微调/消融实验中经过测试：加入时间连续性项（`cbam_full`）相对不加（`cbam_boundary`）并无收益，因此**部署配置中不启用时间连续性损失**（详见第 7 节）。

### 4.6 温度范围损失（仅 OSTIA 预训练）

OSTIA 预训练额外加入一个很小的温度范围惩罚（权重 0.01），抑制预测越出合理海温范围。JAXA 微调阶段不使用该项。


## 5. 训练策略

### 5.1 两阶段训练与迁移学习

本项目采用两阶段训练：**OSTIA 预训练 → 逐小时 JAXA 微调**。这是典型的迁移学习范式。所有训练均在 **4 卡 DDP**（NCCL 后端）上进行。

#### 5.1.1 阶段1：OSTIA 监督预训练

**目标**：在高质量、无缺失的 OSTIA 分析数据上学习 SST 的基本时空规律，为微调提供良好初始化。

**数据与网格**：
- 使用 OSTIA SST 分析产品，重采样到与本项目一致的 **451×351、南海北部（15–24°N, 111–118°E, 0.02°）** 研究网格；
- OSTIA 无云缺失，可提供完整真值；
- 逐日序列，30 天为一个输入窗口。

**人工挖空（模拟云遮挡）**：在完整场上对第 30 天施加人工缺失掩码，训练模型重建被挖空区域。

**训练配置**：
```
硬件/并行 : 4 × GPU，DDP（NCCL）
Batch     : 4 / GPU × 4 = 16
优化器    : AdamW (lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
调度器    : StepLR(step_size=15, gamma=0.5)
Epochs    : 60
梯度裁剪  : max_norm = 1.0
归一化    : 由训练集统计得到（约 mean≈299.92 K, std≈2.69 K）

损失（output composition 之后）：
  L = 1.2 · L_missing_mse
    + 0.0 · L_observed        # 观测项权重为 0（输出组合已保证观测一致）
    + 0.2 · L_gradient
    + 0.15 · L_temporal       # 相对历史日间变化的时间连续性
    + 0.01 · L_range          # 温度范围惩罚
```

**预训练效果**：模型学到 SST 的基本空间模式（锋面、涡旋、环流）与时间演化，成为 24 个逐小时微调模型的共同基座。

#### 5.1.2 阶段2：逐小时 JAXA 微调（24 个模型）

**目标**：从 OSTIA 基座出发，适应 JAXA 数据特征并学习**逐小时日变化**规律。为每个小时 h=00…23 独立微调一个模型，共 **24 个逐小时模型**。

**为什么训练 24 个模型？**
- 逐小时 SST 的日变化显著，不同时刻热力状态差异大（凌晨稳定、正午偏暖、清晨/傍晚快速升降温）；
- 让每个模型专注特定小时的物理过程，更准确重建该时刻的 SST 场。

**数据划分**：
- 共 **9 个年度序列（00–08）**，均为固定小时的逐日序列；
- **训练：序列 0–7（8 年）**；**验证：序列 8（1 年）**。（无第 9 个序列。）

| 序列 | 起始日期 | 天数 |
|------|----------|------|
| 00 | 2017-07-06 | 360 |
| 01 | 2016-07-06 | 317 |
| 02 | 2021-07-05 | 365 |
| 03 | 2018-07-06 | 345 |
| 04 | 2019-07-07 | 356 |
| 05 | 2020-07-05 | 363 |
| 06 | 2022-07-05 | 365 |
| 07 | 2023-07-05 | 365 |
| 08 | 2024-07-04 | 267 |

**输入构造（`JAXAFinetuneDataset`）**：
- 每个样本取以目标日为末端的 30 天窗口；不足 30 天时用首帧向前 padding；
- 前 29 天的 `mask_seq` 使用 `original_missing_mask`（真实云缺失结构）；
- 第 30 天施加**人工方形挖空**（不是不规则云团）：
  - 挖空只落在**真实观测区域** `original_obs_mask` ∩ 海洋 上；
  - 目标挖空比例 `mask_ratio = 0.2`，方块边长 10–50 像素，随机叠加直到达到比例；
  - 第 30 天挖空位置以 `mean` 填充（归一化后为 0），`mask_seq[-1]` 设为该人工掩码。

**Loss Mask（关键细节）**：
```python
loss_mask = artificial_mask ∩ original_obs_mask ∩ ocean
```
- `artificial_mask`：本次人工挖空区域（模型需要重建）；
- `original_obs_mask`：该像素**原本是真实观测**（因此有可信真值）；
- 交集确保“既需要重建、又有真值”——避免用 KNN 填充值当作监督目标，防止模型学习 KNN 的插值误差。

**训练配置（每个小时一个模型）**：
```
硬件/并行 : 4 × GPU，DDP（NCCL）
Batch     : 2 / GPU × 4 = 8
优化器    : AdamW (lr=5e-4, weight_decay=1e-4)     # 比预训练更小的学习率
调度器    : CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
Epochs    : 50（消融实验中使用 100）
梯度裁剪  : max_norm = 1.0
归一化    : 固定使用 OSTIA 参数 mean=299.9221 K, std=2.6919 K

损失（部署配置，output composition 之后）：
  L = 1.0 · L_mse + 0.2 · L_gradient + 0.1 · L_boundary
  （在 loss_mask 区域计算；时间连续性项经消融测试后不启用）
```

**迁移学习流程**：
```python
model = FNO_CBAM_SST_Temporal(out_size=(451,351), modes1=80, modes2=64, width=64, depth=6)
ckpt  = torch.load('experiments/ostia_pretrain/best_model.pth')   # OSTIA 基座
model.load_state_dict(ckpt['model_state_dict'])
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
# 归一化沿用 OSTIA 的 mean/std，保持输入分布一致
```

**微调要点**：
- 训练与验证都执行 output composition，随后仅在 `loss_mask` 上计算损失与指标；
- 验证时不计算梯度/边界损失，只用 MSE 与去归一化后的 MAE/RMSE；
- 以验证 MAE 最优保存 `best_model.pth`（同时保存 `norm_mean/norm_std`）。

### 5.2 数据增强

- **人工挖空**：每个 epoch 随机生成方形挖空（`mask_ratio=0.2`，边长 10–50），带随机种子（训练与验证使用不同种子）；
- **时间窗口**：以 30 天滑动窗口构造样本；
- 不使用随机翻转/旋转（SST 具有明确地理与日变化结构，翻转会破坏物理含义）。

### 5.3 训练技巧

- **梯度裁剪** `clip_grad_norm_(max_norm=1.0)`：稳定训练、防止梯度爆炸；
- **学习率调度**：预训练用 StepLR（每 15 epoch 减半），微调用 CosineAnnealingLR（平滑衰减至 1e-6）；
- **最优模型保存**：以验证集 MAE 为准，另每 10 epoch 存一次 checkpoint。


## 6. 推理流程

### 6.1 完整推理 Pipeline

```
输入：JAXA 原始逐小时 nc（含云缺失）
      ↓
[阶段1: 时间加权填充]  同小时历史（48h 内、≤3 天）反距离加权
      ↓
[阶段2: 高斯低通滤波]  σ=1.5，仅平滑时间填充像素，观测不变
      ↓
[阶段3: 3D 因果渐进式 KNN]  时空双层 KDTree + 偏差校正，海洋填满
      ↓  （以上三阶段与训练数据完全一致，产出 jaxa_knn_filled_XX.h5）
[选择对应小时模型 h=00..23]
      ↓
[构造 30 天窗口输入]
  - sst_seq  : 30 天 KNN 填充序列（归一化）
  - mask_seq : 前 29 天用 original_missing_mask；第 30 天强制置 0
      ↓
[模型前向] → 第 30 天重建（归一化空间）→ 反归一化到 Kelvin
      ↓
[Output Composition]
  - 原始缺失（云）区 original_missing_mask==1 : 用模型预测 pred
  - 其余（真实观测 + 时间填充）             : 用 KNN 填充值 knn
  - 陆地                                     : 置 NaN
      ↓
[高斯滤波后处理 σ=1.0]
  - 只平滑“非真实观测”像素（重建区 = 云区 + 时间回填区）
  - 真实观测 (original_obs_mask==1) 逐位保留，绝不被滤波
  - 卷积输入含观测值 → 重建区借真值邻居平滑 → 无缝衔接
      ↓
输出：重建后的 SST 场（NetCDF，单变量 sea_surface_temperature，Kelvin）
```

### 6.2 高斯滤波后处理

```python
def apply_gaussian(sst, land_mask, fill_region, sigma=1.0):
    valid = ~np.isnan(sst) & (land_mask == 0)
    tmp = sst.copy(); tmp[~valid] = np.nanmean(sst)     # 卷积背景（含观测）
    filtered = gaussian_filter(tmp, sigma=sigma)
    write = valid & (fill_region == 1)                  # fill_region = 非真实观测
    return np.where(write, filtered, sst)               # 观测/陆地保留原值
```

**目的与保证**：
- σ=1.0 平滑模型重建区的高频起伏，改善视觉连续性；
- `fill_region = (original_obs_mask == 0)` 确保**真实观测像素永不被滤波**；
- 由于卷积输入仍包含真实观测，重建区在边界处能与观测无缝衔接。

### 6.3 输出 NetCDF 格式

```
维度：time=1, lat=451, lon=351
变量：
  lat (degrees_north, 24.0 → 15.0)
  lon (degrees_east, 111.0 → 118.0)
  time (seconds since 1981-01-01)
  sea_surface_temperature(time, lat, lon)  单位 kelvin，_FillValue=NaN
目录：/data/sst_data/SST_Data_Imputation/YYYYMM/DD/YYYYMMDDHHMMSS.nc
```

### 6.4 批量推理与断点续跑

```bash
python scripts/inference/batch/infer_jaxa_full.py --hours 1-23 --gpu 6 --batch_size 4
```
- 遍历 h=01…23（h=00 已单独产出）与所有序列（00–08），逐帧推理并保存；
- 分批（`batch_size`）在 GPU 上并行前向；
- **断点续跑**：若目标 nc 已存在则跳过，仅推理缺失帧。


## 7. 消融实验

在 **h=00** 上进行系统消融，逐步加入模块与损失项，隔离各自贡献。所有变体从同一 OSTIA 基座出发，4 卡 DDP、100 epochs、AdamW(lr=5e-4)、CosineAnnealing，数据划分与微调一致（序列 0–7 训练、8 验证）。

### 7.1 变体设计（递进）

| 变体 | 结构 | grad | boundary | temporal | 说明 |
|------|------|:---:|:---:|:---:|------|
| `fno_only` | 无 CBAM | 0 | 0 | 0 | 纯 FNO 基线 |
| `fno_grad` | 无 CBAM | 0.2 | 0 | 0 | 隔离梯度损失的作用 |
| `cbam_basic` | +CBAM | 0.2 | 0 | 0 | 加入 CBAM 注意力（保留梯度损失） |
| `cbam_boundary` | +CBAM | 0.2 | 0.1 | 0 | **加入边界平滑损失（部署配置）** |
| `cbam_full` | +CBAM | 0.2 | 0.1 | 0.1 | 再加入时间连续性损失 |

（`fno_only` 因关闭 CBAM，需在 DDP 中设 `find_unused_parameters=True`。）

### 7.2 结论

- **物理损失（梯度、边界）** 主要作用是**锐化锋面、消除重建区与观测区之间的接缝**，对整体 MAE 的代价可忽略；
- **时间连续性损失**（`cbam_full`）相对 `cbam_boundary` 未带来收益，故**部署配置为 `cbam_boundary`：MSE + 梯度(0.2) + 边界(0.1)，不含时间连续性**；
- 最终 24 个逐小时模型均采用该 `cbam_boundary` 配置。


## 8. 评估

### 8.1 评估协议

- **人工挖空评估**：在真实观测像素上人为制造缺失，再用模型重建，与被挖空处的真值比较；
- 挖空形状包含**小方块**与**大团块（large blob）**两类，覆盖不同缺失几何；
- 挖空强度分三档：低（`target_ratio ≈ 0.30`）、中（`≈ 0.55`）、高（`≈ 0.75`）；
- 指标：在 `挖空 ∩ 真实观测` 区域计算 **MAE、RMSE、Max Error**，并额外统计**边界处 MAE**（评估接缝质量）。

### 8.2 指标定义

```python
MAE  = mean(|pred - gt|)                    # 仅在 挖空∩观测 区域
RMSE = sqrt(mean((pred - gt)^2))
Max  = max(|pred - gt|)
```

### 8.3 结果（h=00，与 KNN-IDW 基线对比）

在 h=00 的人工挖空评估中（每档 12 个样本），FNO-CBAM 在各挖空强度下均显著优于 KNN 基线（单位：K）：

| 缺失强度 | FNO MAE | FNO RMSE | KNN MAE | KNN RMSE |
|----------|:------:|:------:|:------:|:------:|
| 低 (≈0.30) | 0.081 | 0.119 | 0.174 | 0.275 |
| 中 (≈0.55) | 0.133 | 0.186 | 0.197 | 0.295 |
| 高 (≈0.75) | 0.216 | 0.287 | 0.258 | 0.371 |
| 全体 | 0.143 | — | 0.210 | — |

- 缺失越少、可用邻居越多，重建越准；即使在 75% 高挖空下模型仍优于 KNN；
- 模型的优势在低/中缺失下最明显（MAE 约为 KNN 的 45%–67%）。

### 8.4 基线方法

对照基线包括：**KNN-IDW**（反距离加权）、**双线性/线性插值**、**三次(cubic)插值**、以及 **DINEOF**（经验正交函数重建）。其中 **DINEOF 在本数据的高缺失、逐小时设置下重建失败**（无法收敛到合理场），因此不作为有效对照；KNN-IDW 是最稳健的传统基线，故作为主要比较对象。


## 9. 技术创新点

### 9.1 三阶段预处理 Pipeline
- 时间加权填充 → 仅平滑填充区的高斯滤波 → 3D 因果渐进式 KNN；
- 分层处理不同类型缺失，且**全程保护真实观测不被改动**。

### 9.2 3D 因果渐进式 KNN
- 时空双层 KDTree（历史帧 + 当前帧渐进更新）、因果约束、缺失密度排序、偏差校正；
- 相比逐帧 2D KNN，利用了时间邻居，填充更稳定、更连续。

### 9.3 FNO-CBAM 融合架构
- 傅里叶算子（全局）+ CBAM（通道/空间精细化）+ 双路 Linear 时序编码（30 天）。

### 9.4 Output Composition 策略
- 观测区保留原值、仅重建缺失区，训练与推理一致，避免在观测区引入误差。

### 9.5 逐小时微调策略
- 为每个小时训练独立模型（24 个），针对性捕获日变化。

### 9.6 迁移学习范式
- OSTIA 预训练 → JAXA 微调，统一归一化、较小学习率。

### 9.7 物理正则（梯度 + 边界平滑）
- 锐化锋面、消除重建区接缝，以可忽略的整体误差代价提升物理与视觉质量。

### 9.8 真实观测保护 + 仅重建区滤波
- 阶段 2 与推理后处理都只平滑重建像素，真实观测逐位保留，最终得到无缝拼接的连续场。


## 10. 应用场景

### 10.1 海洋学研究
- 海表温度时空演化分析（完整逐小时序列）；
- 锋面/涡旋识别（高分辨率、无缺失）；
- 海气相互作用与日变化研究。

### 10.2 气候与环境监测
- 海洋热含量、海洋热浪检测（逐小时分辨率利于捕获极端事件）；
- 长时间序列的区域气候分析。

### 10.3 数值模式支撑
- 模式初始化与数据同化提供完整场；
- 作为独立数据集用于模式验证。

### 10.4 渔业与航运
- 渔场预报、航线规划、海洋灾害预警的 SST 输入。


## 11. 局限性与未来工作

### 11.1 当前局限
- **计算成本**：24 个逐小时模型，单模型约 6.25 亿参数，训练与存储开销较大；推理需按小时加载对应模型；
- **时间依赖**：需要 30 天历史窗口，序列开头需 padding，实时性受限；
- **极端天气**：台风等快速变化事件的重建仍具挑战；
- **空间泛化**：模型在南海北部 451×351 网格上训练，迁移到其他海域需重新微调。

### 11.2 未来方向
- **统一模型 + 小时嵌入**：以 hour embedding 取代 24 个模型，降低成本；
- **更长时序输入**：探索 60/90 天窗口；
- **多源融合**：结合 MODIS/VIIRS/AMSR 及现场观测提升鲁棒性；
- **不确定性量化**：为重建给出置信区间；
- **实时化**：模型压缩/量化与流式推理；
- **物理约束**：引入海洋动力学约束（PINN 思路）。


## 12. 参考文献

1. **Fourier Neural Operator**: Li, Z., et al. (2020). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR.
2. **CBAM Attention**: Woo, S., et al. (2018). *CBAM: Convolutional Block Attention Module.* ECCV.
3. **JAXA Himawari SST**: https://www.eorc.jaxa.jp/ptree/
4. **OSTIA SST Analysis**: Good, S., et al. (2020). *The Operational Sea Surface Temperature and Sea Ice Analysis (OSTIA).* Remote Sensing.
5. **DINEOF**: Beckers, J.-M. & Rixen, M. (2003). *EOF Calculations and Data Filling from Incomplete Oceanographic Datasets.* JAOT.
6. **Transfer Learning**: Yosinski, J., et al. (2014). *How transferable are features in deep neural networks?* NIPS.

---

**文档版本**：v3.0（与当前代码对齐）
**最后更新**：2026-07-12
**作者**：Leizheng
