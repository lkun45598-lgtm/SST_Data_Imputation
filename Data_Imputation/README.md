# Data_Imputation — 核心代码库

FNO-CBAM 逐小时 SST 缺失值重建的核心实现：预处理、模型、训练、推理、评估、消融与可视化。
目标区域为 **南海北部**，网格 **451 × 351**。项目总览见上级 [../README.md](../README.md)。

## 项目要点

- **两阶段迁移学习**：OSTIA 预训练 → 逐小时 JAXA 微调（h00–h23，共 24 个独立模型）。
- **30 天时序输入**：30 天 SST 序列 + 30 天缺失掩码序列（60 通道）。
- **FNO-CBAM 架构**：傅里叶谱卷积 + CBAM 注意力。
- **Output Composition**：观测像素保留原值，仅重建云遮挡像素。
- **观测保真的高斯后处理**：σ=1.0 仅作用于重建像素。

## 目录结构

```
Data_Imputation/
├── preprocessing/                  # 三阶段预处理
│   ├── hourly_preprocess.py        # 主入口：对指定小时执行三阶段处理
│   ├── temporal_weighted_fill.py   # Stage 1 组件：时间加权填充
│   ├── lowpass_filter.py           # Stage 2 组件：高斯低通滤波
│   ├── knn_fill.py                 # 2D KNN 填充（旧版）
│   └── knn_fill_3d.py              # Stage 3：3D 因果渐进式 KNN 填充
│
├── models/
│   └── fno_cbam_temporal.py        # FNO_CBAM_SST_Temporal（谱卷积 + CBAM）
│
├── losses/
│   └── temporal_loss.py            # 组合损失组件
│
├── datasets/
│   ├── ostia_dataset.py            # OSTIA 预训练数据集
│   ├── ostia_dataset_filled.py     # OSTIA 预填充数据集
│   └── jaxa_dataset.py             # JAXA 数据集
│
├── training/
│   ├── train_ostia.py              # OSTIA 预训练
│   ├── train_jaxa_hourly.py        # 逐小时微调（4-GPU DDP，主用）
│   └── train_jaxa.py               # 早期单模型微调脚本
│
├── inference/
│   ├── jaxa_inference_dataset.py   # 微调/推理数据集（人工掩码生成）
│   ├── evaluate.py                 # 评估（RMSE / MAE / VRMSE 等）
│   ├── fill_jaxa.py / fill_ostia.py
│   └── infer_*_h01.py              # 单小时推理 + 可视化模板
│
├── postprocessing/
│   └── gaussian_filter.py          # 高斯滤波（仅平滑重建像素）
│
├── ablation/                       # 消融实验（自包含，不改动主模型）
│   ├── models/fno_cbam_ablation.py # 带 use_cbam 开关的变体
│   ├── train_ablation_ddp.py       # 4-GPU DDP，逐变体训练
│   └── README.md
│
├── baselines/                      # 传统基线
│   ├── simple_interp.py            # 2D 线性 / 三次样条
│   ├── dineof.py                   # DINEOF（时序 SVD）
│   └── eval_baselines.py           # 与 DL 结果同掩码、同样本对比
│
├── visualization/
│   ├── plot_reconstruction.py, compare_sigma.py
│   └── paper_figs/                 # 论文图件（fig1–fig9）与统计缓存
│
├── sst_pipeline/                   # 统一 Pipeline 封装
│   ├── config.py, pipeline.py, run.py
│   ├── data/loader.py, model/wrapper.py
│   ├── inference/predictor.py, postprocess/gaussian.py, visualization/plotter.py
│
├── docs/                           # architecture / preprocessing / training / evaluation
└── experiments/                    # 预处理数据与模型检查点（gitignored）
    ├── hourly_data/hXX/            # 各小时 KNN 填充 H5
    ├── ostia_pretrain/             # OSTIA 预训练权重
    └── jaxa_finetune_hXX/          # 逐小时微调权重
```

## 数据预处理

`preprocessing/hourly_preprocess.py` 对指定小时 `h`（0–23）与年度序列执行统一三阶段处理：

```
JAXA 逐小时 NC（同小时抽帧成跨天序列）
        ↓
Stage 1  时间加权填充：向前回溯的同小时历史帧，权重 = 1/Δ天 加权平均
        ↓
Stage 2  高斯低通滤波 (σ=1.5)：仅平滑“时间填充像素”，真实观测保持不变
        ↓
Stage 3  3D 因果渐进式 KNN 填充 (k=20)：按缺失密度递进，兼顾时间(7 天窗)与空间邻域
        ↓
experiments/hourly_data/hXX/jaxa_knn_filled_SS.h5
```

输出 H5 字段：

| 字段 | 含义 |
|------|------|
| `sst_data` | 三阶段填充后的 SST 序列 |
| `land_mask` | 陆地掩码（全序列恒 NaN 的像素） |
| `original_obs_mask` | 原始真实观测像素（1=观测） |
| `temporal_fill_mask` | Stage 1 时间填充像素（1=被填充） |
| `original_missing_mask` | Stage 1 之后仍缺失的像素 |
| `latitude` / `longitude` | 坐标 |
| `timestamps` | 每帧时间戳 |

```bash
python preprocessing/hourly_preprocess.py --hour 1 --series 0 --workers 32
python preprocessing/hourly_preprocess.py --hour 1 --series all --workers 64
```

## 训练流程

```
Stage 1: OSTIA 预训练
    - 输入: 30 天 SST 序列 + 30 天掩码序列
    - 脚本: training/train_ostia.py

Stage 2: 逐小时 JAXA 微调（h00–h23）
    - 每个小时从同一 OSTIA 权重独立微调
    - 序列 0–7 训练、序列 8 验证（共 9 个年度序列 00–08，无序列 9）
    - 50 epochs, 4-GPU DDP, lr 5e-4, batch 2/GPU, AdamW + CosineAnnealing
    - 脚本: training/train_jaxa_hourly.py
```

```bash
# OSTIA 预训练
python training/train_ostia.py

# 单个小时微调
python training/train_jaxa_hourly.py --hour 1 --epochs 50 --batch-size 2
```

## 推理流程

```
30 天 KNN 填充序列
        ↓
FNO-CBAM 预测
        ↓
Output Composition：观测区保留原值，云区用模型预测（陆地保持不变）
        ↓
高斯滤波 σ=1.0（仅重建像素）
        ↓
无缝重建结果（观测真值不变）
```

统一 Pipeline：

```python
from sst_pipeline import Pipeline
pipeline = Pipeline()
result = pipeline.process(date="2017-08-08", apply_gaussian=True, sigma=1.0, visualize=True)
```

```bash
python -m sst_pipeline.run --date 2017-08-08 --sigma 1.0
python -m sst_pipeline.run --start-date 2017-08-01 --end-date 2017-08-10
python -m sst_pipeline.run --list-dates
```

## 模型架构

```
输入: [B, 60, 451, 351] = 30 天 SST + 30 天掩码
    ├─ SST/mask 分别 Linear 编码后融合 → width
    ├─ FNO Block × depth(6)
    │   ├─ SpectralConv2d（傅里叶谱卷积）
    │   ├─ Conv2d（局部卷积）+ CBAM 注意力
    │   └─ LayerNorm + 残差 + GELU
    └─ Linear(width→128) → GELU → Linear(128→1)
输出: [B, 1, 451, 351]（第 30 天重建）
```

配置：`out_size=(451, 351)`, `modes1=80`, `modes2=64`, `width=64`, `depth=6`。

## 损失函数

组合损失，仅在评估区域（人工掩码 ∩ 原始观测）计算：

```
L_total = α_mse·L_mse + α_grad·L_grad + α_boundary·L_boundary
```

- `L_mse`：云区重建像素的 MSE。
- `L_grad`：空间梯度一致性。
- `L_boundary`：掩码边界平滑（云区↔观测区边界梯度约束）。
- 时间连续性损失经测试后**未纳入**最终模型。

## 评估

在**真实观测像素**上施加人工方块掩码，将重建值与被遮挡的真值比较：

| 指标 | 说明 |
|------|------|
| RMSE | 均方根误差 (K) |
| MAE | 平均绝对误差 (K) |
| VRMSE | RMSE / std(真值) |
| Max Error | 最大绝对误差 (K) |

基线对比（`baselines/`）：KNN、2D 线性、2D 三次样条、DINEOF；消融实验见 `ablation/README.md`。

## 依赖环境

```
torch >= 1.10
numpy, scipy, h5py, netCDF4, xarray, matplotlib, tqdm
```

## 备注

1. **归一化参数**：微调沿用 OSTIA 归一化参数（mean=299.92K, std=2.69K）。
2. **Output Composition**：推理时观测区直接用输入值，模型只预测缺失区。
3. **高斯滤波**：推荐 σ=1.0，仅作用于重建像素。
