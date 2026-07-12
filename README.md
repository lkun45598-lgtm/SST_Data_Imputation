# FNO-CBAM 逐小时 SST 缺失值重建

基于 **傅里叶神经算子 (Fourier Neural Operator, FNO)** 与 **卷积块注意力模块 (CBAM)** 的海表温度 (SST) 缺失值重建系统。

## 项目概述

本项目针对 **南海北部海域** JAXA 卫星逐小时 SST 数据中由云层大面积遮挡造成的缺失，采用深度学习方法进行时空重建。核心特点：

- **目标区域与网格**：南海北部，规则网格 **451 × 351**（纬度 × 经度）。
- **时序输入**：每个样本使用 **30 天** 同小时 SST 序列 + 对应的缺失掩码序列（共 60 通道）。
- **FNO-CBAM 架构**：傅里叶谱卷积捕捉全局空间模式，CBAM 注意力增强关键特征。
- **两阶段迁移学习**：OSTIA 预训练 → 逐小时 JAXA 微调（共 24 个逐小时模型 h00–h23，每个独立微调）。
- **观测保真的输出组合 (Output Composition)**：真实观测像素原值保留，模型仅重建非观测（云遮挡）像素。
- **高斯后处理**：最终 σ=1.0 高斯滤波 **仅作用于重建像素**，观测真值保持不变 → 得到无缝、无缺口且保留真实观测的 SST 场。

📖 详细方法论：[METHODOLOGY.md](METHODOLOGY.md)

## 目录结构

```
SST_Data_Imputation/
├── README.md                       # 本文件（项目总览）
├── METHODOLOGY.md                  # 详细方法论文档
├── manuscript.tex                  # 论文稿
│
├── Data_Imputation/                # 核心代码库（见其内部 README.md）
│   ├── preprocessing/              # 三阶段预处理
│   ├── models/                     # FNO-CBAM 模型定义
│   ├── losses/                     # 组合损失
│   ├── datasets/                   # OSTIA / JAXA 数据集
│   ├── training/                   # 训练脚本（OSTIA 预训练 + 逐小时微调）
│   ├── inference/                  # 推理与评估
│   ├── postprocessing/             # 高斯滤波后处理
│   ├── ablation/                   # 消融实验（自包含，不改动主模型）
│   ├── baselines/                  # 传统基线（linear/cubic/DINEOF）
│   ├── visualization/paper_figs/   # 论文图件生成脚本
│   ├── sst_pipeline/               # 统一 Pipeline 封装
│   ├── docs/                       # 分模块文档
│   └── experiments/                # 预处理数据与模型检查点（gitignored）
│
└── scripts/                        # 生产/批量脚本
    ├── inference/
    │   ├── batch/                  # 批量与全量推理
    │   │   ├── infer_jaxa_full.py         # 全量逐小时推理 (h01–h23, 全 series)
    │   │   ├── batch_infer_hourly.py
    │   │   └── batch_infer_original_hourly.py
    │   ├── single/                 # 单样本推理（调试）
    │   └── visualization/          # 结果可视化（时序图等）
    ├── training/
    │   ├── batch_preprocess_hourly.sh     # 批量预处理 h00–h23
    │   └── batch_train_hourly.sh          # 批量逐小时微调
    └── testing/
```

## 数据流程

### 1. 数据预处理（JAXA 原始 NC → 模型输入 H5）

由 `Data_Imputation/preprocessing/hourly_preprocess.py` 对每个小时 `h`（0–23）执行统一的三阶段处理：

```
JAXA 逐小时 NC 文件（按小时抽帧成跨天序列）
        ↓
Stage 1  时间加权填充：同一小时、向前回溯的历史帧，权重 = 1/Δ天 加权平均
        ↓
Stage 2  高斯低通滤波 (σ=1.5)：仅平滑“时间填充像素”，真实观测像素保持不变
        ↓
Stage 3  3D 因果渐进式 KNN 填充 (k=20)：按缺失密度递进，兼顾时间与空间邻域
        ↓
experiments/hourly_data/hXX/jaxa_knn_filled_SS.h5   ← 模型输入
```

输出 H5 字段：`sst_data`、`land_mask`、`original_obs_mask`、`temporal_fill_mask`、
`original_missing_mask`、`latitude`、`longitude`、`timestamps`。

### 2. 训练流程（两阶段迁移学习）

```
Stage 1: OSTIA 预训练
    - 数据: OSTIA SST（南海北部）
    - 输入: 30 天 SST 序列 + 30 天掩码序列
    - 目的: 学习 SST 空间模式与时间动态
    - 脚本: Data_Imputation/training/train_ostia.py

Stage 2: 逐小时 JAXA 微调（h00–h23，共 24 个模型）
    - 每个小时从同一个 OSTIA 预训练权重独立微调
    - 训练/验证: 9 个年度序列 00–08；序列 0–7 训练，序列 8 验证（无序列 9）
    - 配置: 50 epochs, 4-GPU DDP, lr 5e-4, AdamW + CosineAnnealing
    - 脚本: Data_Imputation/training/train_jaxa_hourly.py
    - 批量: scripts/training/batch_train_hourly.sh
```

### 3. 推理与产品生成

```
30 天 KNN 填充序列（原值不替换）
        ↓
FNO-CBAM 预测
        ↓
Output Composition：观测区保留原值，云区用模型预测（陆地保持不变）
        ↓
高斯滤波 σ=1.0：仅作用于重建像素
        ↓
无缝、无缺口且保留真实观测的 SST 场
```

全量推理（`scripts/inference/batch/infer_jaxa_full.py`）对每个小时的全部 series 逐帧推理并写出
NC 文件，目录结构：`/data/sst_data/SST_Data_Imputation/YYYYMM/DD/YYYYMMDDHHMMSS.nc`，
含 `sst_original`（原始观测，云区 NaN）、`sst_knn_filled`（KNN 粗填）、
`sst_model_filled`（FNO-CBAM 重建）、`original_missing_mask`（云掩码）。

## 快速开始

### 预处理

```bash
# 单小时、单序列
python Data_Imputation/preprocessing/hourly_preprocess.py --hour 1 --series 0 --workers 32
# 单小时、全部序列
python Data_Imputation/preprocessing/hourly_preprocess.py --hour 1 --series all --workers 64
# 批量 h00–h23
bash scripts/training/batch_preprocess_hourly.sh
```

### 训练

```bash
# 1. OSTIA 预训练
python Data_Imputation/training/train_ostia.py

# 2. 逐小时微调（单个小时；50 epochs, 4-GPU DDP）
python Data_Imputation/training/train_jaxa_hourly.py --hour 1 --epochs 50 --batch-size 2

# 3. 批量微调 h00–h23
bash scripts/training/batch_train_hourly.sh
```

### 推理

```bash
# 全量逐小时推理 (h01–h23, 全部 series)
python scripts/inference/batch/infer_jaxa_full.py --hours 1-23 --gpu 6

# 时序图可视化
python scripts/inference/visualization/plot_point_timeseries.py \
    --lat 19.0 --lon 115.0 --year 2024 --month 7
```

## 模型架构

```
输入: [B, 60, 451, 351]
      ├─ 30 通道: SST 序列（30 天）
      └─ 30 通道: 缺失掩码序列（30 天）

    ├─ 编码: SST 与 mask 分别经 Linear 编码后融合 → width
    ├─ FNO Block × depth(6)
    │   ├─ SpectralConv2d（傅里叶谱卷积）
    │   ├─ Conv2d（局部空间卷积）
    │   ├─ CBAM 注意力（通道 + 空间）
    │   ├─ LayerNorm + 残差
    │   └─ GELU
    └─ 解码: Linear(width→128) → GELU → Linear(128→1)

输出: [B, 1, 451, 351]（第 30 天重建 SST）
```

配置：`out_size=(451, 351)`, `modes1=80`, `modes2=64`, `width=64`, `depth=6`。

## 损失函数

训练损失为组合损失，仅在评估区域（人工掩码 ∩ 原始观测）计算：

```
L_total = α_mse·L_mse + α_grad·L_grad + α_boundary·L_boundary
```

- `L_mse`：重建（云区）像素的 MSE。
- `L_grad`：空间梯度一致性损失。
- `L_boundary`：掩码边界平滑损失（约束云区↔观测区边界处梯度）。
- 时间连续性损失经实验测试后**未纳入**最终模型。

## 评估

评估通过在**真实观测像素**上施加人工掩码，比较重建值与被遮挡的真值；指标包括 RMSE、MAE、
VRMSE、Max Error。基线对比方法：KNN、2D 线性、2D 三次样条、DINEOF
（见 `Data_Imputation/baselines/`）。

| 指标 | 说明 |
|------|------|
| RMSE | 均方根误差 (K) |
| MAE | 平均绝对误差 (K) |
| VRMSE | RMSE / std(真值) |
| Max Error | 最大绝对误差 (K) |

## 依赖环境

```
torch >= 1.10
numpy, scipy, h5py, netCDF4, xarray, matplotlib, tqdm
```

## 备注

1. **归一化参数**：JAXA 微调沿用 OSTIA 预训练的归一化参数（mean=299.92K, std=2.69K），保证输入分布一致。
2. **观测保真**：推理与后处理均不改动真实观测像素，模型只重建云遮挡区域。
3. **高斯滤波**：推荐 σ=1.0，仅作用于重建像素。

## 作者

Leizheng
