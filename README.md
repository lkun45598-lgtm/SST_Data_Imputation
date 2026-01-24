FNO-CBAM SST缺失值重建项目
基于Fourier Neural Operator (FNO) 和 Convolutional Block Attention Module (CBAM) 的海表温度 (SST) 缺失值重建系统。

项目概述
本项目针对JAXA卫星SST数据中由于云层遮挡导致的缺失值问题，采用深度学习方法进行重建。主要特点：

两阶段训练策略：OSTIA预训练 → JAXA微调
30天时间序列输入：利用时间连续性信息
FNO-CBAM架构：结合傅里叶神经算子和注意力机制
Output Composition：观测区域保留原值，仅对缺失区域进行预测
高斯滤波后处理：平滑重建结果，减少噪声
目录结构
Data_Imputation/
├── models/                     # 模型定义
│   └── fno_cbam_temporal.py    # FNO-CBAM时序模型
│
├── losses/                     # 损失函数
│   └── temporal_loss.py        # 组合损失 (MSE + 梯度 + 时间连续性)
│
├── datasets/                   # 数据集定义
│   ├── ostia_dataset.py        # OSTIA预训练数据集
│   ├── ostia_dataset_filled.py # OSTIA预填充数据集 (加速加载)
│   └── jaxa_dataset.py         # JAXA微调数据集
│
├── preprocessing/              # 数据预处理Pipeline
│   ├── temporal_weighted_fill.py   # Step1: 时间加权填充 (hourly→daily)
│   ├── lowpass_filter.py           # Step2: 低通滤波
│   └── knn_fill.py                 # Step3: KNN空间填充
│
├── training/                   # 训练脚本
│   ├── train_ostia.py          # OSTIA预训练 (8 GPU DDP)
│   └── train_jaxa.py           # JAXA微调 (8 GPU DDP)
│
├── inference/                  # 推理与评估
│   ├── fill_jaxa.py            # JAXA数据填充
│   ├── evaluate.py             # 模型评估 (VRMSE, MAE, RMSE)
│   └── jaxa_inference_dataset.py   # 推理数据集
│
├── postprocessing/             # 后处理
│   └── gaussian_filter.py      # 高斯滤波平滑
│
├── visualization/              # 可视化
│   ├── plot_reconstruction.py  # 重建结果可视化
│   └── compare_sigma.py        # 高斯滤波sigma对比
│
├── sst_pipeline/               # 统一Pipeline模块
│   ├── config.py               # 配置管理
│   ├── pipeline.py             # 主Pipeline类
│   ├── run.py                  # 命令行入口
│   ├── data/loader.py          # 数据加载器
│   ├── model/wrapper.py        # 模型封装
│   ├── inference/predictor.py  # 推理器
│   ├── postprocess/gaussian.py # 高斯后处理
│   └── visualization/plotter.py # 可视化绘图
│
├── scripts/                    # Shell脚本
└── docs/                       # 文档
数据流程
1. 数据预处理 (JAXA原始数据 → 模型输入)
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
[knn_fill.py] KNN空间填充
        ↓
    jaxa_knn_filled/*.h5  ← 模型输入数据
2. 训练流程
Stage 1: OSTIA预训练
    - 数据: OSTIA SST (南海区域)
    - 输入: 30天SST序列（缺失区域用最近邻插值填充）+ 30天Mask序列
    - 目的: 学习SST的空间模式和时间动态
    - 脚本: training/train_ostia.py

Stage 2: JAXA微调
    - 数据: JAXA SST (南海区域, 高分辨率)
    - 输入: 30天SST序列（缺失区域用KNN填充）+ 30天Mask序列
    - 目的: 适应目标区域的特征
    - 脚本: training/train_jaxa.py
3. 推理流程
30天KNN填充数据
        ↓
[模型推理] FNO-CBAM预测
        ↓
[Output Composition] 观测区域保留原值
        ↓
[高斯滤波] sigma=1.0平滑
        ↓
    最终重建结果
快速开始
使用Pipeline API
from sst_pipeline import Pipeline

# 初始化Pipeline
pipeline = Pipeline()

# 处理单个日期
result = pipeline.process(
    date="2017-08-08",
    apply_gaussian=True,
    sigma=1.0,
    visualize=True
)

# 查看可用日期
dates = pipeline.get_available_dates()
使用命令行
# 处理单个日期
python -m sst_pipeline.run --date 2017-08-08 --sigma 1.0

# 处理日期范围
python -m sst_pipeline.run --start-date 2017-08-01 --end-date 2017-08-10

# 查看可用日期
python -m sst_pipeline.run --list-dates

# 自定义参数
python -m sst_pipeline.run --date 2017-08-08 \
    --sigma 1.5 \
    --mask-ratio 0.3 \
    --no-visualize
模型架构
FNO-CBAM-Temporal
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
损失函数
L_total = α₁·L_mse + α₂·L_gradient + α₃·L_temporal

其中:
- L_mse: 缺失区域MSE损失
- L_gradient: 空间梯度一致性损失
- L_temporal: 时间连续性损失
评估指标
指标	公式	说明
RMSE	√(mean((pred-gt)²))	均方根误差
MAE	mean(|pred-gt|)	平均绝对误差
VRMSE	RMSE / std(gt)	相对于标准差的RMSE
Max Error	max(|pred-gt|)	最大误差
当前模型性能 (JAXA测试集)
指标	值
VRMSE	0.215 ± 0.062
MAE	0.095 ± 0.014 K
RMSE	0.126 ± 0.020 K
Max Error	0.620 ± 0.153 K
配置说明
模型配置
# models/fno_cbam_temporal.py
FNO_CBAM_SST_Temporal(
    out_size=(451, 351),  # 输出尺寸 (H, W)
    modes1=80,            # 傅里叶模式数 (纬度方向)
    modes2=64,            # 傅里叶模式数 (经度方向)
    width=64,             # 网络宽度
    depth=6               # FNO块数量
)
训练配置
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
数据路径配置
# sst_pipeline/config.py
PathConfig(
    jaxa_knn_filled_dir = '/path/to/jaxa_knn_filled',
    model_path = '/path/to/best_model.pth',
    output_dir = '/path/to/output'
)
依赖环境
torch >= 1.10
numpy
scipy
h5py
netCDF4
matplotlib
tqdm
文件说明
核心文件
文件	说明
models/fno_cbam_temporal.py	FNO-CBAM模型定义，包含SpectralConv2d和CBAM模块
losses/temporal_loss.py	组合损失函数，支持缺失区域加权
training/train_ostia.py	OSTIA预训练，8卡DDP，Output Composition
training/train_jaxa.py	JAXA微调，加载预训练权重
inference/evaluate.py	评估脚本，计算VRMSE等指标
sst_pipeline/pipeline.py	统一Pipeline，一键处理
预处理文件
文件	输入	输出
temporal_weighted_fill.py	JAXA hourly NC	jaxa_weighted_aligned/*.h5
lowpass_filter.py	jaxa_weighted_aligned/*.h5	jaxa_filtered/*.h5
knn_fill.py	jaxa_filtered/*.h5	jaxa_knn_filled/*.h5
注意事项
归一化参数：JAXA微调使用OSTIA预训练的归一化参数 (mean=299.92K, std=2.69K)
Output Composition：推理时观测区域直接使用输入值，模型只预测缺失区域
高斯滤波：推荐sigma=1.0，过大会模糊细节，过小效果不明显
GPU内存：单卡推理需要约8GB显存
作者
Claude Code

更新日志
2026-01-23: 项目重构，模块化整理
2026-01-22: JAXA 8年数据微调完成
2026-01-20: 添加高斯滤波后处理
2026-01-19: OSTIA预训练完成
