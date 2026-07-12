# 模型训练指南

## 概述

FNO-CBAM 模型采用**两阶段迁移学习**策略：

1. **Stage 1 — OSTIA 预训练**：在无缺失（gap-free）的 OSTIA 再分析数据上学习 SST 的时空模式，得到一个通用基座模型。
2. **Stage 2 — JAXA 逐小时微调**：以 OSTIA 基座为起点，针对目标海域的 JAXA 卫星数据做领域适应。**每个整点小时（00–23）各自独立微调出一个模型**，全部从同一份 OSTIA 基座出发。

---

## 训练流程图

```
┌──────────────────────────────────────────────┐
│  Stage 1: OSTIA 预训练                         │
│  training/train_ostia.py                       │
│  - 数据: OSTIA L4 全球再分析 SST (无缺失)      │
│  - 目的: 学习 SST 时空模式                     │
│  - 输出: experiments/ostia_pretrain/best_model.pth │
└──────────────────────────────────────────────┘
                        │
        （同一份基座，24 个小时共用）
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐   ...   ┌───────────────┐
│ Stage 2: h00  │ │ Stage 2: h01  │         │ Stage 2: h23  │
│ train_jaxa_   │ │ train_jaxa_   │         │ train_jaxa_   │
│ hourly.py     │ │ hourly.py     │         │ hourly.py     │
│ --hour 0      │ │ --hour 1      │         │ --hour 23     │
│ jaxa_finetune │ │ jaxa_finetune │         │ jaxa_finetune │
│               │ │ _h01          │         │ _h23          │
└───────────────┘ └───────────────┘         └───────────────┘
```

- h00 输出目录为 `experiments/jaxa_finetune`；h01–h23 输出目录为 `experiments/jaxa_finetune_h{hh}`。
- 24 个模型互相独立，每个都从 `ostia_pretrain/best_model.pth` 微调而来。

---

## Stage 1: OSTIA 预训练

**脚本**: `training/train_ostia.py`

### 数据配置

| 配置项 | 值 |
|--------|-----|
| 数据源 | OSTIA L4 SST（无缺失再分析场） |
| 输入格式 | 30 天序列，双输入 `sst_seq [30,H,W]` + `mask_seq [30,H,W]` |
| 输入通道 | 60 (30×SST + 30×Mask) |
| 空间尺寸 | 451 × 351 |
| 训练/验证 | `processed_sst_train.h5` / `processed_sst_valid.h5` |

### 模型与训练配置

```python
# 模型
FNO_CBAM_SST_Temporal(
    out_size=(451, 351),
    modes1=80,      # 傅里叶模式数 (纬度)
    modes2=64,      # 傅里叶模式数 (经度)
    width=64,       # 网络宽度
    depth=6         # FNO+CBAM 块数量
)

# 训练 (4 卡 DDP)
world_size    = 4
batch_size    = 4  per GPU × 4 GPUs = 16
learning_rate = 1e-3
epochs        = 60
optimizer     = AdamW(weight_decay=1e-4, betas=(0.9, 0.999))
scheduler     = StepLR(step_size=15, gamma=0.5)
```

### 损失函数

```python
L_total = 1.2×L_missing + 0.0×L_observed + 0.2×L_gradient
        + 0.15×L_temporal + 0.01×L_range

# L_missing:  缺失区域 MSE
# L_observed: 观测区域 MSE（因使用 Output Composition，权重设为 0）
# L_gradient: 空间梯度一致性
# L_temporal: 时间连续性约束
# L_range:    温度范围约束
```

### Output Composition

```python
# 观测区域直接使用输入值，模型只预测缺失区域
final = input × (1 - mask) + pred × mask
```

### 运行命令

```bash
python training/train_ostia.py
```

输出保存至 `experiments/ostia_pretrain/`（`best_model.pth` 按验证 MAE 选优）。

---

## Stage 2: JAXA 逐小时微调

**脚本**: `training/train_jaxa_hourly.py`

每个整点小时独立运行一次，`--hour` 指定小时（0–23）。

### 数据配置

| 配置项 | 值 |
|--------|-----|
| 数据源 | JAXA KNN 填充后的逐小时数据 |
| 输入格式 | 30 天序列，`sst_seq [30,H,W]` + `mask_seq [30,H,W]` |
| 训练集 | Series 0–7（8 年） |
| 验证集 | Series 8（1 年） |
| 人工挖空比例 | 20%（方形，边长 10–50） |
| 归一化 | 复用 OSTIA 基座参数（见下） |

### 数据集与挖空机制（`inference/jaxa_inference_dataset.py`）

- **30 天时间窗口**：取目标日及其前 29 天。
- **人工方形挖空只在"原始观测像素"上生成**：`valid_for_mask = original_obs_mask × (1 - land_mask)`（即海洋且原本有观测的像素），从而模拟真实缺失。
- **第 30 天（目标日）的 `mask_seq` 被替换为人工挖空掩码 `artificial_mask`**；前 29 天沿用各自的 `original_missing_mask`。
- **Loss 计算区域**：`loss_mask = artificial_mask ∩ original_obs_mask ∩ ocean`，即"被人工挖掉、原本有观测、且是海洋"的像素。
- **Ground Truth**：目标日的 `sst_data`（KNN 填充后的完整参考场）。在被挖空的观测像素处，参考值就是该像素真实的观测值——因此在 `loss_mask` 区域比对的是模型预测与真实观测。
- 被挖空的输入像素用均值填充（归一化后为 0）。

### 训练配置

```python
# 加载 OSTIA 基座
pretrained_path = 'experiments/ostia_pretrain/best_model.pth'

# 训练 (4 卡 DDP)
world_size    = 4
batch_size    = 2  per GPU × 4 GPUs = 8
learning_rate = 5e-4
epochs        = 50            # train_jaxa_hourly.py 默认值 (h01–h23)；h00 基座训练了 100 epoch
optimizer     = AdamW(weight_decay=1e-4)
scheduler     = CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
grad_clip     = 1.0
```

（以上超参可在 `experiments/jaxa_finetune*/config.json` 中核对：`batch_size=2`、`lr=5e-4`；h01 为 50 epoch，h00 为 100 epoch。）

### 归一化参数

**重要**：JAXA 微调必须复用 OSTIA 预训练的归一化参数，保证输入分布一致。

```python
ostia_mean = 299.9221  # Kelvin
ostia_std  = 2.6919     # Kelvin
```

### Output Composition（含陆地处理）

```python
# 非挖空区域用输入值，挖空区域用预测，陆地强制保持输入
composed = last_input × (1 - last_mask) + pred × last_mask
# land 区域再覆盖回 last_input
```

### 损失函数

最终部署目标为 **MSE + 梯度损失 + 边界（平滑）损失**：

```python
L_total = 1.0×L_mse + 0.2×L_gradient + 0.1×L_boundary

# 全部只在 loss_mask 区域计算
# L_mse:      挖空区域重建误差
# L_gradient: 挖空区域梯度一致性
# L_boundary: mask 边界处（洞内↔洞外相邻像素对）的梯度一致性，
#             用于锐化掩码/锋面边界
```

> 说明：曾额外测试过一项**时间连续性损失（temporal loss）**，实验表明它对整体精度无可测收益，因此不纳入最终目标函数（详见 `ablation/README.md` 的消融结论）。

### 运行命令

```bash
# 单个小时
python training/train_jaxa_hourly.py --hour 0
python training/train_jaxa_hourly.py --hour 1
# ... 直到 --hour 23

# 可选参数
python training/train_jaxa_hourly.py --hour 5 --epochs 50 --batch-size 2
```

---

## 多 GPU 训练

### 环境要求

- 4× NVIDIA GPU（如 H20/A100）
- PyTorch ≥ 1.10（支持 DDP）

### DDP 配置

```python
# 两个阶段均使用 4 卡 DDP
world_size = 4
mp.spawn(train_worker, args=(world_size, ...), nprocs=world_size, join=True)
```

---

## 训练监控

### 关键指标

```bash
# 验证损失 / 最优模型
grep "Valid - Loss" <日志>
grep "最优模型已保存" <日志>
```

### GPU 监控

```bash
watch -n 1 nvidia-smi
```

### 训练曲线

训练完成后查看 `experiments/*/training_history.json`（含每个 epoch 的 train/valid 指标）：

```python
import json, matplotlib.pyplot as plt

with open('training_history.json') as f:
    history = json.load(f)

train_mae = [e['mae'] for e in history['train']]
valid_mae = [e['mae'] for e in history['valid']]
plt.plot(train_mae, label='Train MAE')
plt.plot(valid_mae, label='Valid MAE')
plt.legend(); plt.savefig('mae_curve.png')
```

---

## 输出文件

```
experiments/ostia_pretrain/          # Stage 1 基座
├── best_model.pth                   # 最优 (按验证 MAE)
├── final_model.pth
├── checkpoint_epoch_*.pth
└── training_history.json

experiments/jaxa_finetune/           # Stage 2, h00
experiments/jaxa_finetune_h01/       # Stage 2, h01
...                                  # ... 直到 h23
experiments/jaxa_finetune_h23/
每个目录含:
├── best_model.pth                   # 最优 (含 norm_mean/norm_std)
├── final_model.pth
├── checkpoint_epoch_*.pth
├── training_history.json
└── config.json
```

---

## 常见问题

### 1. CUDA Out of Memory

减小每卡 batch size：

```bash
python training/train_jaxa_hourly.py --hour 0 --batch-size 1
```

### 2. Loss 不收敛

- 确认使用了正确的 OSTIA 归一化参数（`mean=299.9221, std=2.6919`）
- 检查基座权重是否成功加载（日志应打印"预训练权重加载成功"）
- 检查数据是否含 NaN

### 3. 验证 MAE 不下降

- 适当增加 epochs
- 检查是否过拟合（train/valid MAE 差距）
- 核对损失权重

---

## 预计训练时间（参考）

| 阶段 | GPU | 约耗时 |
|------|-----|--------|
| OSTIA 预训练 (60 epoch) | 4 卡 | ~8 小时 |
| JAXA 单小时微调 (50 epoch) | 4 卡 | ~4 小时 |
| JAXA 全部 24 小时 | 4 卡 | 逐个串行约数天，可按小时分批调度 |
</content>
</invoke>
