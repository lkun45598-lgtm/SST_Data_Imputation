# 模型训练指南

## 概述

FNO-CBAM模型采用两阶段训练策略：
1. **OSTIA预训练**: 在全球OSTIA数据上学习SST时空模式
2. **JAXA微调**: 在目标区域JAXA数据上进行领域适应

---

## 训练流程图

```
┌─────────────────────────────────────┐
│  Stage 1: OSTIA预训练               │
│  training/train_ostia.py            │
│  - 数据: OSTIA全球SST               │
│  - 目的: 学习SST时空模式            │
│  - 输出: pretrained_model.pth       │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 2: JAXA微调                  │
│  training/train_jaxa.py             │
│  - 数据: JAXA日本海域SST            │
│  - 目的: 领域适应                   │
│  - 输出: best_model.pth             │
└─────────────────────────────────────┘
```

---

## Stage 1: OSTIA预训练

**脚本**: `training/train_ostia.py`

### 数据配置

| 配置项 | 值 |
|--------|-----|
| 数据源 | OSTIA L4 SST |
| 空间范围 | 15°N-24°N, 111°E-118°E (南海) |
| 时间窗口 | 30天 |
| 输入通道 | 60 (30×SST + 30×Mask) |
| **输入填充方式** | **最近邻插值** |

### 训练配置

```python
# 模型参数
FNO_CBAM_SST_Temporal(
    out_size=(451, 351),
    modes1=80,      # 傅里叶模式数 (纬度)
    modes2=64,      # 傅里叶模式数 (经度)
    width=64,       # 网络宽度
    depth=6         # FNO块数量
)

# 训练参数
batch_size = 4 × 8 GPUs = 32
learning_rate = 1e-3
epochs = 60
optimizer = AdamW(weight_decay=1e-4)
scheduler = StepLR(step_size=15, gamma=0.5)
```

### 损失函数

```python
L_total = 1.2×L_missing + 0.0×L_observed + 0.2×L_gradient + 0.15×L_temporal + 0.01×L_range

# 各项说明:
# L_missing:  缺失区域MSE损失
# L_observed: 观测区域MSE损失 (使用Output Composition后设为0)
# L_gradient: 空间梯度一致性
# L_temporal: 时间连续性约束
# L_range:    温度范围约束
```

### Output Composition

```python
# 关键技术: 观测区域直接使用输入值，模型只预测缺失区域
final = input × (1 - mask) + prediction × mask
```

### 运行命令

```bash
cd /path/to/project
python training/train_ostia.py
```

---

## Stage 2: JAXA微调

**脚本**: `training/train_jaxa.py`

### 数据配置

| 配置项 | 值 |
|--------|-----|
| 数据源 | JAXA KNN填充后数据 |
| 空间范围 | 15°N-24°N, 111°E-118°E (南海) |
| 训练集 | Series 0-7 (8年) |
| 验证集 | Series 8 (1年) |
| **输入填充方式** | **KNN空间填充** |
| Mask策略 | 20%人工挖空 |

### 训练配置

```python
# 加载预训练权重
pretrained_path = 'experiments/ostia_pretrained/best_model.pth'

# 训练参数
batch_size = 2 × 8 GPUs = 16
learning_rate = 5e-4      # 较小学习率
epochs = 100
optimizer = AdamW(weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=100, eta_min=1e-6)
```

### 归一化参数

**重要**: JAXA微调必须使用OSTIA预训练的归一化参数！

```python
# OSTIA归一化参数
ostia_mean = 299.9221  # Kelvin
ostia_std = 2.6919     # Kelvin
```

### 损失函数

```python
L_total = 1.0×L_mse + 0.02×L_gradient + 0.1×L_temporal

# 只在人工挖空区域计算损失
# loss_mask = artificial_mask ∩ original_obs_mask
```

### 运行命令

```bash
cd /path/to/project
python training/train_jaxa.py
```

---

## 多GPU训练

### 环境要求

- 8× NVIDIA GPU (推荐H20/A100/V100)
- 每卡显存 ≥ 40GB
- PyTorch ≥ 1.10 (支持DDP)

### DDP配置

```python
# 自动配置8卡DDP
world_size = 8
mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
```

### 显存占用

| 阶段 | 单卡显存 |
|------|---------|
| OSTIA预训练 (bs=4) | ~35GB |
| JAXA微调 (bs=2) | ~25GB |

---

## 训练监控

### 查看训练日志

```bash
# 实时查看
tail -f logs/training_*.log

# 查看关键指标
grep "Valid - Loss" logs/training_*.log
grep "最优模型已保存" logs/training_*.log
```

### GPU监控

```bash
watch -n 1 nvidia-smi
```

### 训练曲线

训练完成后查看 `experiments/*/training_history.json`:

```python
import json
import matplotlib.pyplot as plt

with open('training_history.json') as f:
    history = json.load(f)

plt.plot(history['train_loss'], label='Train')
plt.plot(history['valid_loss'], label='Valid')
plt.legend()
plt.savefig('loss_curve.png')
```

---

## 输出文件

```
experiments/jaxa_finetune_8years/
├── best_model.pth           # 最优模型 (按验证MAE)
├── final_model.pth          # 最终模型
├── checkpoint_epoch_*.pth   # 定期检查点
├── training_history.json    # 训练历史
└── config.json              # 训练配置
```

---

## 常见问题

### 1. CUDA Out of Memory

**解决方案**: 减小batch_size

```python
batch_size = 2  # 从4减到2
```

### 2. Loss不收敛

**解决方案**:
- 检查数据归一化是否正确
- 降低学习率: `lr = 1e-4`
- 检查数据是否有NaN

### 3. 验证MAE不下降

**解决方案**:
- 增加训练epochs
- 调整损失权重
- 检查是否过拟合

---

## 预计训练时间

| 阶段 | GPU | 时间 |
|------|-----|------|
| OSTIA预训练 | 8×H20 | ~8小时 |
| JAXA微调 | 8×H20 | ~12小时 |
| **总计** | - | **~20小时** |
