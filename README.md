# FNO-CBAM SST缺失值重建项目

基于Fourier Neural Operator (FNO) 和 Convolutional Block Attention Module (CBAM) 的海表温度 (SST) 缺失值重建系统。

## 项目概述

本项目针对JAXA卫星SST数据中由于云层遮挡导致的缺失值问题，采用深度学习方法进行重建。主要特点：

- **两阶段训练策略**：OSTIA预训练 → JAXA逐小时微调（H=00~H=23）
- **30天时间序列输入**：利用时间连续性信息
- **FNO-CBAM架构**：结合傅里叶神经算子和注意力机制
- **Output Composition**：观测区域保留原值，仅对缺失区域进行预测
- **高斯滤波后处理**：平滑重建结果，减少噪声
- **全量数据重建**：已完成2016-07至2025-03共73,004个逐小时SST文件重建

📖 **详细方法论文档**：[METHODOLOGY.md](METHODOLOGY.md) - 包含完整的算法原理、模型架构、训练策略和技术创新点

## 目录结构

```
SST_Data_Imputation/
├── METHODOLOGY.md              # 📖 详细方法论文档
├── README.md                   # 项目说明
│
├── Data_Imputation/
│   ├── models/                 # 模型定义
│   │   └── fno_cbam_temporal.py    # FNO-CBAM时序模型（~606M参数）
│   │
│   ├── losses/                 # 损失函数
│   │   └── temporal_loss.py    # 组合损失 (MSE + 梯度 + 边界平滑)
│   │
│   ├── datasets/               # 数据集定义
│   │   ├── ostia_dataset.py    # OSTIA预训练数据集
│   │   ├── ostia_dataset_filled.py # OSTIA预填充数据集
│   │   └── jaxa_dataset.py     # JAXA微调数据集
│   │
│   ├── preprocessing/          # 数据预处理Pipeline
│   │   ├── temporal_weighted_fill.py   # Step1: 时间加权填充
│   │   ├── lowpass_filter.py           # Step2: 低通滤波
│   │   ├── knn_fill.py                 # Step3: 2D KNN填充
│   │   └── knn_fill_3d.py              # Step3: 3D渐进式KNN填充（推荐）
│   │
│   ├── training/               # 训练脚本
│   │   ├── train_ostia.py      # OSTIA预训练 (8 GPU DDP)
│   │   └── train_jaxa_hourly.py # JAXA逐小时微调 (8 GPU DDP, H=00~H=23)
│   │
│   ├── inference/              # 推理与评估
│   │   ├── fill_jaxa.py        # JAXA数据填充
│   │   ├── evaluate.py         # 模型评估 (VRMSE, MAE, RMSE)
│   │   ├── jaxa_inference_dataset.py   # 推理数据集
│   │   ├── infer_and_visualize_h01.py  # 单小时推理+可视化模板
│   │   └── infer_with_original_missing_h01.py  # 原始缺失推理模板
│   │
│   ├── postprocessing/         # 后处理
│   │   └── gaussian_filter.py  # 高斯滤波平滑（σ=1.0）
│   │
│   ├── visualization/          # 可视化
│   │   ├── plot_reconstruction.py  # 重建结果可视化
│   │   └── compare_sigma.py    # 高斯滤波sigma对比
│   │
│   └── utils/                  # 工具脚本
│
└── scripts/                    # 生产脚本
    ├── inference/
    │   ├── batch/              # 批量推理（生产）
    │   │   ├── batch_infer_hourly.py
    │   │   ├── batch_infer_original_hourly.py
    │   │   └── infer_jaxa_full.py  # 全量推理（73,004文件）
    │   ├── single/             # 单样本推理（调试）
    │   └── visualization/      # 数据可视化
    │       └── plot_point_timeseries.py  # 时序图绘制
    ├── training/
    │   ├── batch_preprocess_hourly.sh  # 批量预处理
    │   └── batch_train_hourly.sh       # 批量训练H=00~H=23
    └── testing/                # 测试脚本
```

## 数据流程

### 1. 数据预处理 (JAXA原始数据 → 模型输入)

```
JAXA hourly NC files
        ↓
[temporal_weighted_fill.py] 时间加权填充，hourly→daily
        ↓
    jaxa_weighted_aligned/*.h5
        ↓
[lowpass_filter.py] 低通滤波去噪
        ↓
    jaxa_filtered/*.h5
        ↓
[knn_fill.py] 3D KNN时空填充（考虑时间连续性）
        ↓
    jaxa_knn_filled/*.h5  ← 模型输入数据
```

### 2. 训练流程

```
Stage 1: OSTIA预训练
    - 数据: OSTIA SST (南海区域)
    - 输入: 30天SST序列（缺失区域用最近邻插值填充）+ 30天Mask序列
    - 目的: 学习SST的空间模式和时间动态
    - 脚本: Data_Imputation/training/train_ostia.py

Stage 2: JAXA逐小时微调 (H=00~H=23)
    - 数据: JAXA hourly SST (24个小时分别训练24个模型)
    - 输入: 30天SST序列（缺失区域用3D KNN填充）+ 30天Mask序列
    - 目的: 适应目标区域特征，捕捉日变化规律
    - 脚本: Data_Imputation/training/train_jaxa_hourly.py
    - 批量训练: scripts/training/batch_train_hourly.sh
```

### 3. 推理流程

```
30天KNN填充数据
        ↓
[模型推理] FNO-CBAM预测
        ↓
[Output Composition] 观测区域保留原值
        ↓
[高斯滤波] sigma=1.0平滑
        ↓
    最终重建结果
```

### 4. 全量数据重建

已完成2016-07至2025-03共73,004个逐小时SST文件的重建：

```bash
# 批量推理脚本（H=01~H=23）
python scripts/inference/batch/infer_jaxa_full.py

# 输出目录结构
/data/sst_data/SST_Data_Imputation/
├── YYYYMM/
│   └── DD/
│       └── YYYYMMDDHHMMSS.nc  # 单变量格式，包含Gaussian σ=1.0滤波
```

详细说明见 `/data/sst_data/SST_Data_Imputation/README.md`

## 快速开始

### 训练模型

```bash
# 1. OSTIA预训练（8 GPU）
cd Data_Imputation/training
python train_ostia.py

# 2. JAXA逐小时微调（批量训练H=00~H=23）
cd ../../scripts/training
bash batch_train_hourly.sh
```

### 批量推理

```bash
# 推理所有JAXA数据（H=01~H=23）
python scripts/inference/batch/infer_jaxa_full.py --gpu 6

# 可视化推理结果（原始缺失模式）
python scripts/inference/batch/batch_infer_original_hourly.py --hours 1-23 --gpu 6
```

### 数据可视化

```bash
# 绘制指定点的时序图（年度/月度/周度）
python scripts/inference/visualization/plot_point_timeseries.py \
    --lat 19.0 --lon 115.0 --year 2024 --month 7
```

## 模型架构

### FNO-CBAM-Temporal

```
输入: [B, 60, H, W]
      ├─ 30 channels: SST序列 (30天)
      └─ 30 channels: Mask序列 (30天)

架构:
    ├─ Lifting: Conv2d (60 → 64)
    ├─ FNO Blocks × 6
    │   ├─ Spectral Conv (傅里叶域卷积)
    │   ├─ Local Conv (空间域卷积)
    │   ├─ CBAM Attention
    │   └─ GELU + Residual
    └─ Projection: Conv2d (64 → 1)

输出: [B, 1, H, W] (第30天SST预测)
```

### 损失函数

```
L_total = α₁·L_mse + α₂·L_gradient + α₃·L_temporal

其中:
- L_mse: 缺失区域MSE损失
- L_gradient: 空间梯度一致性损失
- L_temporal: 时间连续性损失
```

## 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| RMSE | √(mean((pred-gt)²)) | 均方根误差 |
| MAE | mean(\|pred-gt\|) | 平均绝对误差 |
| VRMSE | RMSE / std(gt) | 相对于标准差的RMSE |
| Max Error | max(\|pred-gt\|) | 最大误差 |

### 当前模型性能 (JAXA测试集)

| 指标 | 值 |
|------|------|
| VRMSE | 0.215 ± 0.062 |
| MAE | 0.095 ± 0.014 K |
| RMSE | 0.126 ± 0.020 K |
| Max Error | 0.620 ± 0.153 K |

## 配置说明

### 模型配置

```python
# models/fno_cbam_temporal.py
FNO_CBAM_SST_Temporal(
    out_size=(451, 351),  # 输出尺寸 (H, W)
    modes1=80,            # 傅里叶模式数 (纬度方向)
    modes2=64,            # 傅里叶模式数 (经度方向)
    width=64,             # 网络宽度
    depth=6               # FNO块数量
)
```

### 训练配置

```python
# OSTIA预训练
batch_size = 4 × 8 GPUs = 32
learning_rate = 1e-3
epochs = 60
optimizer = AdamW
scheduler = StepLR(step=15, gamma=0.5)

# JAXA微调
batch_size = 2 × 8 GPUs = 16
learning_rate = 5e-4
epochs = 100
optimizer = AdamW
scheduler = CosineAnnealing
```

### 数据路径配置

```python
# sst_pipeline/config.py
PathConfig(
    jaxa_knn_filled_dir = '/path/to/jaxa_knn_filled',
    model_path = '/path/to/best_model.pth',
    output_dir = '/path/to/output'
)
```

## 依赖环境

```
torch >= 1.10
numpy
scipy
h5py
netCDF4
matplotlib
tqdm
```

## 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `models/fno_cbam_temporal.py` | FNO-CBAM模型定义，包含SpectralConv2d和CBAM模块 |
| `losses/temporal_loss.py` | 组合损失函数，支持缺失区域加权 |
| `training/train_ostia.py` | OSTIA预训练，8卡DDP，Output Composition |
| `training/train_jaxa.py` | JAXA微调，加载预训练权重 |
| `inference/evaluate.py` | 评估脚本，计算VRMSE等指标 |
| `sst_pipeline/pipeline.py` | 统一Pipeline，一键处理 |

### 预处理文件

| 文件 | 输入 | 输出 |
|------|------|------|
| `temporal_weighted_fill.py` | JAXA hourly NC | jaxa_weighted_aligned/*.h5 |
| `lowpass_filter.py` | jaxa_weighted_aligned/*.h5 | jaxa_filtered/*.h5 |
| `knn_fill.py` | jaxa_filtered/*.h5 | jaxa_knn_filled/*.h5 |

## 注意事项

1. **归一化参数**：JAXA微调使用OSTIA预训练的归一化参数 (mean=299.92K, std=2.69K)
2. **Output Composition**：推理时观测区域直接使用输入值，模型只预测缺失区域
3. **高斯滤波**：推荐sigma=1.0，过大会模糊细节，过小效果不明显
4. **GPU内存**：单卡推理需要约8GB显存

## 文档

- **[METHODOLOGY.md](METHODOLOGY.md)** - 详细方法论文档
  - 3D渐进式KNN填充算法原理
  - FNO-CBAM模型架构详解
  - 损失函数设计与数学推导
  - 两阶段训练策略
  - 技术创新点与应用场景

- **[/data/sst_data/SST_Data_Imputation/README.md](/data/sst_data/SST_Data_Imputation/README.md)** - 输出数据集说明
  - 数据格式与变量定义
  - 读取示例代码
  - 数据统计信息

## 引用

如果您使用本项目的代码或数据，请引用：

```bibtex
@software{sst_reconstruction_2026,
  title = {FNO-CBAM SST缺失值重建系统},
  author = {Leizheng},
  year = {2026},
  url = {https://github.com/lkun45598-lgtm/SST_Data_Imputation}
}
```

## 作者

Leizheng

## 更新日志

- 2026-04-20: 添加详细方法论文档（METHODOLOGY.md）
- 2026-01-24: 完成全量数据重建（73,004文件）
- 2026-01-23: 项目重构，模块化整理
- 2026-01-22: JAXA逐小时微调完成（H=00~H=23）
- 2026-01-20: 添加高斯滤波后处理
- 2026-01-19: OSTIA预训练完成
