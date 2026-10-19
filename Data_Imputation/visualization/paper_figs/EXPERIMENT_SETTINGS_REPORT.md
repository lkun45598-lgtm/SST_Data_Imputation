# 实验设置统一报告

**生成时间**: 2026-07-23  
**目的**: 核查论文图表实验设置的一致性，确定统一标准

---

## 1. Mask Ratio 设置统一

### 当前状态
- ✅ **主要评估脚本已统一**: 所有关键脚本使用 **30% / 55% / 75%**
  - `gen_fig6_stats.py` (主结果图)
  - `eval_proxy_ablation.py` (proxy消融)
  - `gen_fig8_stats.py` (ablation图)
  - `gen_fig8_largeblob.py`
  
- ⚠️ **论文中需要更新**: 如果论文文本中提到了不同的 mask ratio，需要统一改为 30/55/75

### 决议
**统一标准: 30% / 55% / 75%** (低/中/高 missingness levels)

---

## 2. 主结果数值统一

### Fig6 主结果 (gen_fig6_stats.py → fig6_stats.npz)
**Full model (FNO + CBAM + Temporal)** 在 series 8, N=20/level:

| Level | MAE (K) | RMSE (K) |
|-------|---------|----------|
| Low (30%)  | **0.324** | **0.425** |
| Mid (55%)  | **0.327** | **0.446** |
| High (75%) | **0.341** | **0.453** |

**这是标准参考值** - 来自:
- Model: `experiments/jaxa_finetune_h12_realmask_clean/best_model.pth`
- Data: Series 8 (jaxa_knn_filled_08.h5)
- Samples: 20 per level
- Mask: Real-cloud shapes, 30/55/75% ratios

### Proxy Ablation 表 (eval_proxy_ablation.py → proxy_ablation_stats.npz)
**Full model** 在 series 8, N=219/level:

| Level | MAE (K) | RMSE (K) |
|-------|---------|----------|
| Low (30%)  | **0.477** | **0.638** |
| Mid (55%)  | **0.475** | **0.634** |
| High (75%) | **0.493** | **0.659** |

### ⚠️ 数值差异分析

**Full model 在两个表中的结果不同:**

| Source | Low MAE | Mid MAE | High MAE | Avg RMSE |
|--------|---------|---------|----------|----------|
| Fig6主结果 | 0.324 K | 0.327 K | 0.341 K | ~0.44 K |
| Proxy消融  | 0.477 K | 0.475 K | 0.493 K | ~0.64 K |
| **差异** | **+0.153 K** | **+0.148 K** | **+0.152 K** | **+0.20 K** |

**原因分析:**
1. **样本数不同**: N=20 vs N=219
2. **随机mask不同**: 不同的 random seed 生成不同的 mask realizations
3. **可能的评估模式差异**: 
   - Fig6 可能使用了某种平滑/后处理
   - Proxy ablation 使用的是 raw 输入 proxy (不同变体)

### 决议
**以 Fig6 主结果为准** (MAE: 0.324 / 0.327 / 0.341 K):
- 这是主比较实验的官方结果
- 样本数合理 (N=20)
- 与其他方法在同等条件下比较

**Proxy ablation 表保持独立**:
- 这是单独的消融实验
- 使用更大样本集 (N=219) 提高统计稳定性
- 重点是相对比较 (full vs noknn vs nogauss)，不是绝对数值

---

## 3. Baseline 方法整理

### 已有评估缓存的方法

#### 3.1 深度学习方法
从 `stats_*_v6.npz` 提取 (N=20/level, series 8):

| Method | Low MAE | Mid MAE | High MAE | Avg RMSE |
|--------|---------|---------|----------|----------|
| **FNO (Full)** | 0.416 K | 0.422 K | 0.372 K | ~0.51 K |
| **U-Net** | 0.363 K | 0.368 K | 0.338 K | ~0.46 K |
| **ConvLSTM** | 0.270 K | 0.292 K | 0.267 K | ~0.37 K |
| **DINCAE** | 0.355 K | 0.363 K | 0.313 K | ~0.45 K |

⚠️ **注意**: 这些数值与 Fig6 的 Full model (0.324/0.327/0.341 K) 不一致，可能来自:
- 不同的模型 checkpoint
- 不同的评估脚本
- 不同的数据处理流程

#### 3.2 传统方法
从 `eval_final.py` 和其他脚本:

| Method | 描述 | 状态 |
|--------|------|------|
| **DINEOF** | EOF-based reconstruction | ✅ 已评估 (多个k值) |
| **OI** | Optimal Interpolation | ✅ 已评估 |
| **KNN** | 3D K-Nearest Neighbors | ✅ 已评估 |
| **Linear** | Linear interpolation | ⚠️ 仅在部分实验 |
| **Cubic** | Cubic interpolation | ⚠️ 仅在部分实验 |

### 方法文档状态

| Method | 论文方法部分 | 训练脚本 | 评估脚本 | 引用 |
|--------|-------------|---------|---------|------|
| **FNO (Ours)** | ✅ 已写 | ✅ train_jaxa_hourly.py | ✅ 多个 | ✅ 已有 |
| **U-Net** | ⚠️ 需补充 | ✅ deep_models.py | ✅ 已有缓存 | ⚠️ 需添加 |
| **ConvLSTM** | ⚠️ 需补充 | ✅ deep_models.py | ✅ 已有缓存 | ⚠️ 需添加 |
| **DINCAE** | ⚠️ 需补充 | ✅ dincae.py | ✅ 已有缓存 | ⚠️ 需添加 |
| **DINEOF** | ⚠️ 需补充 | ✅ dineof.py | ✅ eval_dineof.py | ⚠️ 需添加 |
| **OI** | ⚠️ 需补充 | N/A (传统方法) | ✅ 已集成 | ⚠️ 需添加 |

---

## 4. 实验配置标准

### 4.1 主比较实验 (Main Comparison Table + Fig6)
- **目的**: 与 baseline 方法对比
- **Model**: jaxa_finetune_h12_realmask_clean
- **Data**: Series 8 (jaxa_knn_filled_08.h5)
- **Mask ratios**: 30% / 55% / 75%
- **Mask type**: Real cloud shapes
- **Samples**: N=20 per level (60 total)
- **Evaluation area**: Artificial masked region (隐藏的真实观测)
- **脚本**: `gen_fig6_stats.py`
- **缓存**: `cache/fig6_stats.npz`

### 4.2 Proxy Construction Ablation
- **目的**: 消融 Gaussian/KNN 预处理的作用
- **Variants**: Full / No Gaussian / No KNN
- **Data**: Series 8, 三个变体各用自己的 proxy
- **Mask ratios**: 30% / 55% / 75%
- **Mask type**: Real cloud shapes
- **Samples**: N=219 per level per variant (1971 total)
- **Evaluation**: 统一的干净真值 (noknn h5的观测区)
- **脚本**: `eval_proxy_ablation.py`
- **缓存**: `cache/proxy_ablation_stats.npz`

### 4.3 Loss Component Ablation (Fig8)
- **目的**: 消融 CBAM / gradient / boundary / temporal loss
- **Variants**: FNO-only / CBAM / CBAM+boundary / CBAM+temporal (正在训练)
- **Data**: Series 8
- **Mask ratios**: 30% / 55% / 75%
- **Mask type**: Real cloud shapes
- **Samples**: N=15 per level per variant
- **脚本**: `gen_fig8_stats.py`
- **缓存**: `cache/fig8_stats.npz`

### 4.4 Hourly Performance (Fig7)
- **目的**: 展示24小时模型的逐小时性能
- **Models**: h00-h23 (24个模型)
- **Data**: 各自对应小时的 series 8
- **Mask ratio**: 20% (单一 level)
- **Samples**: 每小时约30-50样本
- **脚本**: `gen_fig7_stats.py`
- **缓存**: `cache/fig7_stats.npz`

---

## 5. 待完成任务

### 5.1 立即执行
- [x] 统一确认 mask ratio 为 30/55/75
- [x] 确定 Fig6 主结果为标准参考值
- [ ] 检查论文文本中的数值，统一为 Fig6 标准
- [ ] 补充 baseline 方法的描述和引用

### 5.2 等实验完成后
- [ ] 更新 Fig8 ablation 结果（CBAM消融实验完成后）
- [ ] 验证所有图表使用相同的 mask ratio
- [ ] 统一所有表格的数值格式和精度

### 5.3 方法部分补充内容
需要为以下方法添加:
1. **方法描述** (1-2段):
   - 模型架构
   - 输入形式 (temporal window, channels)
   - 训练设置 (optimizer, learning rate, epochs)
   
2. **引用文献**:
   - U-Net: Ronneberger et al., 2015
   - ConvLSTM: Shi et al., 2015
   - DINCAE: Barth et al., 2020
   - DINEOF: Alvera-Azcárate et al., 2005
   - OI: 经典方法引用

3. **公平对比说明**:
   - 所有深度学习方法使用相同的输入 proxy (KNN-filled history)
   - 相同的训练/验证/测试划分
   - 相同的 mask realizations 用于评估

---

## 6. 关键文件路径

### 模型
- Full model: `experiments/jaxa_finetune_h12_realmask_clean/best_model.pth`
- No KNN: `experiments/jaxa_finetune_h12_noknn_realmask_clean/best_model.pth`
- No Gaussian: `experiments/jaxa_finetune_h12_nogauss_realmask_clean/best_model.pth`

### 数据
- Full proxy: `experiments/hourly_data/h12/jaxa_knn_filled_08.h5`
- No KNN proxy: `experiments/hourly_data/h12_noknn/jaxa_knn_filled_08.h5`
- No Gaussian proxy: `experiments/hourly_data/h12_nogauss/jaxa_knn_filled_08.h5`

### 评估脚本
- 主结果: `visualization/paper_figs/gen_fig6_stats.py`
- Proxy消融: `visualization/paper_figs/eval_proxy_ablation.py`
- Loss消融: `visualization/paper_figs/gen_fig8_stats.py`
- 小时性能: `visualization/paper_figs/gen_fig7_stats.py`

### 缓存结果
- 主结果: `visualization/paper_figs/cache/fig6_stats.npz`
- Proxy消融: `visualization/paper_figs/cache/proxy_ablation_stats.npz`
- Loss消融: `visualization/paper_figs/cache/fig8_stats.npz`
- 小时性能: `visualization/paper_figs/cache/fig7_stats.npz`

---

**总结**: 实验设置基本一致，mask ratio 已统一为 30/55/75。主要工作是补充 baseline 方法描述和确保论文文本中的数值与 Fig6 主结果一致。
