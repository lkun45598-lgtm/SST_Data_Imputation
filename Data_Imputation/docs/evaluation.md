# 评估指标与方法

本文档描述 SST 填充模型（FNO-CBAM）的评估协议、指标定义、掩码策略、
基线对比与相关脚本。**以代码为准**，各节均标注对应源码路径。

---

## 1. 评估哲学：在"观测像素"上人工挖空

真实的缺测像素**没有真值**，无法直接评估。因此本项目的评估统一采用
**人工掩码（artificial mask）** 策略：

1. 取某一帧（滑动窗口的第 30 天 / day-30）的**原始观测区域**
   `original_obs_mask`（1 = 该像素被卫星真实观测到）。
2. 只在"观测区域 ∩ 海洋"范围内人为挖空一部分像素，得到 `artificial_mask`。
3. 被挖空像素在挖空前的真值 = 该像素的真实观测值，可作为 Ground Truth。
4. 模型在挖空后的输入上重建，最终**只在 `artificial ∩ observed ∩ ocean`
   像素上**计算误差。

> **术语约定**：这里的 "Ground Truth" 指被挖空的观测像素在挖空前的真实观测值
> （代码中即 KNN 补全场 `sst_data` 第 30 天在观测像素处的取值——在观测像素上
> KNN 场 = 原始观测）。**不要**把模型的原始输入笼统称作 "Ground Truth"。

相关代码：`inference/evaluate.py`、
`visualization/paper_figs/gen_eval_data.py` 等脚本中的
`eval_mask = artificial_mask * original_obs_mask * (1 - land_mask)`。

---

## 2. 评估指标

误差在开尔文（K）下计算；由于是差值，K 与 °C 数值相等（1 K 差 = 1 °C 差）。
陆地像素不参与任何计算。

### 2.1 MAE（平均绝对误差）

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$

主指标，单位 K（=°C）。

### 2.2 RMSE（均方根误差）

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$

单位 K（=°C），对大误差更敏感。

### 2.3 边界区 MAE（Boundary MAE）

在掩码/缺测区域的**边界带**上单独计算 MAE，用来考察模型在填充区与观测区
交界处（以及温度锋面附近）的过渡是否平滑。

边界带由掩码的膨胀与腐蚀之差得到（`scipy.ndimage.binary_dilation` /
`binary_erosion`，`dilation=2`），见各脚本中的 `boundary_pixels()`。
该指标出现在 `gen_fig8_stats.py`、`gen_fig8_largeblob.py`、
`baselines/eval_baselines.py`（字段 `bnd_mae`）。

### 2.4 VRMSE（方差缩放 RMSE，可选）

$$\text{VRMSE} = \frac{\text{RMSE}}{\text{std}(y)}$$

无量纲，衡量相对于"猜均值"基线的改进：`<1` 优于猜均值，`=1` 相当，`>1` 更差。
仅在 `inference/evaluate.py` 的 `calculate_metrics()` 中计算与打印。

### 2.5 Max Error（最大绝对误差）

$$\text{Max Error} = \max_i |\hat{y}_i - y_i|$$

反映最坏局部失败，见 `inference/evaluate.py` 与 fig8 相关脚本（字段 `max`）。

> 相关性（correlation）为可选指标，当前脚本未默认计算。

---

## 3. 掩码策略

评估提供两种掩码生成器与多档缺失比例，以覆盖不同难度：

### 3.1 小尺度散点方块（`SquareMaskGenerator`）

- 由随机方块（边长 **10–50 像素**）拼成，散布在观测区域内。
- 用于 `inference/evaluate.py`、`gen_eval_data.py`、`gen_fig6_stats.py`、
  `gen_fig7_stats.py`、`gen_fig8_stats.py`、`baselines/eval_baselines.py`。

### 3.2 大尺度连续块（`LargeBlobMaskGenerator`，fig8 large-blob）

- 由较大方块（边长 **80–200 像素，约 4°–10°**）组成，块少而大。
- **模拟真实云带**：大块内部远离任何观测锚点，是对插值类方法的真正压力测试。
- 用于 `visualization/paper_figs/gen_fig8_largeblob.py`。

### 3.3 缺失比例档位

论文图使用三档目标缺失率（相对观测区域）：

| 档位 | target ratio |
|------|--------------|
| low  | ~0.30 |
| mid  | ~0.55 |
| high | ~0.75 |

`inference/evaluate.py` 单独使用 `MASK_RATIO = 0.20`；
`gen_fig7_stats.py` 的逐小时评估使用固定 `0.50`。

---

## 4. 评估流程

### 4.1 输入构建（与训练一致）

见各脚本的 `inference_like_training()` / `predict_fno()` / `predict_model()`：

1. 取以 day-30 为末帧的 **30 天 KNN 补全序列**（`WINDOW_SIZE = 30`）。
2. 第 30 天在 `artificial_mask > 0` 的位置用归一化均值 `norm_mean` 填充。
3. 掩码通道：前 29 天用各自的 `original_missing_mask`，第 30 天用 `artificial_mask`。
4. 归一化 `(x - norm_mean) / norm_std`，NaN 置 0，送入模型。
5. 反归一化回 K。

模型：`models/fno_cbam_temporal.FNO_CBAM_SST_Temporal`
（`out_size=(451,351)`, `modes1=80`, `modes2=64`, `width=64`, `depth=6`,
`cbam_reduction_ratio=16`），权重取自
`experiments/jaxa_finetune/best_model.pth`，
`norm_mean≈299.92 K`、`norm_std≈2.69 K`。

### 4.2 输出合成（Output Composition）

**观测像素保留原值，只有被挖空的像素写入模型预测**：

```python
filled = sst_seq[-1].copy()                 # day-30 观测/KNN 场
filled = np.where(artificial_mask > 0, pred_kelvin, filled)
```

### 4.3 高斯后处理（σ = 1.0，仅作用于填充区）

后处理高斯滤波 **只写回填充/挖空区域**；**真实观测保持原值不被平滑，陆地保持 NaN**。
见各脚本 `gauss_filter(..., fill_region=(mask > 0))` 与
`inference/evaluate.py: apply_gaussian_filter_sst(..., fill_region=...)`：

```python
write = valid & (fill_region == 1)          # 仅填充区写回滤波值
result = np.where(write, filtered, np.where(valid, sst, np.nan))
```

因为指标只在挖空像素上计算，此举保证观测值永不被改动。

---

## 5. 基线对比

| 基线 | 说明 | 代码 |
|------|------|------|
| **KNN-IDW** | 2D 反距离加权插值（`k=20`, `power=2`） | `knn_inpaint_2d()` |
| **Linear 2D** | 2D 线性插值 | `baselines/simple_interp.linear_interp_2d` |
| **Cubic 2D** | 2D 三次样条插值 | `baselines/simple_interp.cubic_interp_2d` |
| **DINEOF** | 30 天序列上的时序 SVD 填充（`k=20` 模态，均值中心化） | `baselines/dineof.dineof_fill` |

所有基线在**与深度模型完全相同的样本与掩码**（相同随机种子）上评估，
并施加相同的高斯后处理，结果可直接对比。

> **DINEOF 是失败基线**：在大块掩码下误差远超其他方法（见下表，MAE ~0.8 K，
> bndMAE ~0.65 K），说明纯时序 SVD 无法处理大面积连续缺失。

### 5.1 实测结果（当前缓存）

**小尺度方块掩码（`cache/fig8_stats.npz`，45 样本，K）**

| 方法 | MAE | RMSE | 边界 MAE |
|------|-----|------|----------|
| KNN-IDW | 0.206 | 0.311 | 0.122 |
| fno_only | 0.130 | 0.185 | 0.091 |
| cbam_basic | 0.127 | 0.179 | 0.080 |
| cbam_boundary | 0.131 | 0.188 | 0.073 |
| cbam_full | 0.131 | 0.188 | 0.074 |
| **main（部署模型）** | **0.130** | **0.182** | 0.084 |

**大尺度连续块掩码（`cache/fig8_largeblob_stats.npz`，45 样本，K）**

| 方法 | MAE | RMSE | 边界 MAE |
|------|-----|------|----------|
| KNN-IDW | 0.276 | 0.396 | 0.142 |
| Linear 2D | 0.219 | 0.311 | 0.103 |
| Cubic 2D | 0.243 | 0.428 | 0.064 |
| DINEOF | 0.843 | 0.991 | 0.657 |
| fno_only | 0.199 | 0.273 | 0.105 |
| cbam_basic | 0.189 | 0.262 | 0.090 |
| **main** | **0.193** | **0.265** | 0.094 |

> 数值由当前缓存计算，随重新运行脚本刷新；单位 K（=°C）。

---

## 6. 逐小时泛化

系统是 **24 个逐小时微调模型（h00–h23）** 的集合；主图中以 **hour-00 作为代表**
（`experiments/jaxa_finetune`），其余小时为
`experiments/jaxa_finetune_h{HH}/best_model.pth`。

`gen_fig7_stats.py` 对全部 24 个小时分别评估：

- **Part A**：对每个小时用 `mask_ratio=0.50` 的方块掩码、每小时 10 个样本
  （`SERIES_TO_USE = 0`）计算 MAE/RMSE，考察逐小时精度稳定性。
- **Part B**：从重建归档（`202407`）统计 24 小时**日变化（diurnal cycle）**，
  比较模型输出与观测像素处原始观测的逐小时均值，并给出定点
  （19°N, 115°E）时序。

---

## 7. 脚本与输出

数据来源：`/data1/.../FNO_CBAM/jaxa_knn_filled/jaxa_knn_filled_00.h5`
（`SERIES_ID = 0`；同目录另有 series 01–08）。窗口 `WINDOW_SIZE = 30`，种子
`SEED = 42`。

| 脚本 | 作用 | 输出 |
|------|------|------|
| `inference/evaluate.py` | 单模型评估（方块掩码 20%，30 样本），打印 MAE/RMSE/VRMSE/Max，出 4 连图 | `experiments/evaluation_results/evaluation_results.json` + `eval_*.png` |
| `visualization/paper_figs/gen_eval_data.py` | fig4/6 用例：FNO-CBAM vs KNN-IDW，三档掩码各 12 样本，保存代表性用例与逐样本统计 | `cache/eval_cache.npz`, `cache/eval_stats.json` |
| `.../gen_fig6_stats.py` | fig6 聚合统计：逐像素误差图、误差直方图、pred-vs-GT 散点，三档各 20 样本 | `cache/fig6_stats.npz` |
| `.../gen_fig7_stats.py` | fig7：逐小时 MAE/RMSE（24 小时）+ 日变化曲线 | `cache/fig7_stats.npz` |
| `.../gen_fig8_stats.py` | fig8 消融：main + 各消融变体 + KNN，小方块掩码，三档各 15 样本 | `cache/fig8_stats.npz` |
| `.../gen_fig8_largeblob.py` | fig8 大块掩码：上述全部 + Linear/Cubic/DINEOF 传统基线 | `cache/fig8_largeblob_stats.npz` |
| `baselines/eval_baselines.py` | 传统基线（Linear/Cubic/DINEOF）在小方块掩码上评估，与 fig8 对齐 | `cache/baseline_stats.npz` |

消融变体（`gen_fig8_stats.py`）：`fno_only`（无 CBAM）、`cbam_basic`、
`cbam_boundary`、`cbam_full`，及主模型 `main`。

### 运行示例

```bash
cd /data1/user/lz/SST_Data_Imputation/Data_Imputation
python inference/evaluate.py
python visualization/paper_figs/gen_eval_data.py
python visualization/paper_figs/gen_fig8_largeblob.py
python baselines/eval_baselines.py
```

### `inference/evaluate.py` 可视化（4 连图）

```
┌───────────────┬──────────────────┬─────────────────────┬──────────────────────┐
│ Input SST     │ Ground Truth SST │ FNO-CBAM Prediction │ Absolute Error       │
│ (挖空区透明)  │ (day-30 观测场)  │ (+Gaussian σ=1.0)   │ (仅挖空区)           │
└───────────────┴──────────────────┴─────────────────────┴──────────────────────┘
```

---

## 8. FNO-CBAM 实测精度（当前缓存）

由 `cache/eval_stats.json`（`gen_eval_data.py`，三档各 12 样本）聚合，单位 K（=°C）：

| 缺失档位 | 实际缺失率 | FNO-CBAM MAE | FNO-CBAM RMSE | KNN-IDW MAE |
|----------|-----------|--------------|---------------|-------------|
| low  | ~0.31 | 0.081 | 0.119 | 0.174 |
| mid  | ~0.56 | 0.133 | 0.186 | 0.197 |
| high | ~0.76 | 0.216 | 0.287 | 0.258 |
| 全部 | — | 0.143 | 0.197 | 0.210 |

**结论**：缺失比例越大误差越高；各档 FNO-CBAM 的 MAE 均显著低于 KNN-IDW 基线，
高缺失档优势相对收窄。数值随脚本重跑刷新。

---

## 9. 评估注意事项

1. **评估区域**：仅在 `artificial ∩ observed ∩ ocean` 像素上计算指标（此处有真值）。
2. **陆地排除**：陆地像素不参与任何计算，滤波后保持 NaN。
3. **观测保护**：输出合成与高斯后处理都不改动真实观测；后处理只作用于填充区。
4. **单位**：误差在 K 下计算，与 °C 数值相等；VRMSE 无量纲。
5. **一致性**：基线与深度模型使用相同样本、相同掩码（相同随机种子）、相同后处理，
   确保对比公平。
6. **代表模型**：主图中的 "Ours" 为 hour-00 部署模型，逐小时结果见 fig7。
</content>
</invoke>
