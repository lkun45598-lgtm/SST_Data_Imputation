# 消融实验（Ablation）

用于验证各组件贡献的独立训练目录。**不修改**主模型代码
（`Data_Imputation/models/fno_cbam_temporal.py`）与生产训练脚本——主结果所用代码保持冻结。

消融训练在**整点 00（hour 00）** 上进行，数据为 `jaxa_knn_filled`，从 OSTIA 基座
（`experiments/ostia_pretrain/best_model.pth`）微调；所有变体使用完全相同的数据、
batch size、学习率调度与 epoch 数，以保证公平对比。

```
Data_Imputation/ablation/
├── models/
│   └── fno_cbam_ablation.py     # FNO_CBAM_Ablation，带 use_cbam 开关
├── train_ablation_ddp.py         # 4 卡 DDP，每次训练一个变体
├── run_all_variants.sh           # 顺序训练多个变体
├── experiments/                  # 各变体 checkpoint + history (gitignored)
│   ├── fno_only/
│   ├── fno_grad/
│   ├── cbam_basic/
│   ├── cbam_boundary/
│   └── cbam_full/
└── README.md
```

## 变体（Variants）

| 名称            | use_cbam | α_grad | α_temporal | α_boundary | 说明                                          |
|-----------------|----------|--------|------------|------------|-----------------------------------------------|
| `fno_only`      | False    | 0.0    | 0.0        | 0.0        | 纯 FNO，仅 MSE                                |
| `fno_grad`      | False    | 0.2    | 0.0        | 0.0        | 纯 FNO + 梯度损失（隔离梯度损失的单独贡献）   |
| `cbam_basic`    | True     | 0.2    | 0.0        | 0.0        | + CBAM 注意力（+ 梯度损失）                   |
| `cbam_boundary` | True     | 0.2    | 0.0        | 0.1        | + 边界平滑损失　**（部署配置 / "Ours"）**     |
| `cbam_full`     | True     | 0.2    | 0.1        | 0.1        | + 时间连续性损失（用于验证其是否有益）        |

> 变体定义见 `train_ablation_ddp.py` 中的 `VARIANTS` 字典，与上表一致。

## 结论

- **物理启发的损失（梯度损失、边界平滑损失）**在几乎不增加整体 MAE 的前提下，
  显著**锐化了掩码/锋面边界**，改善了填补结果的视觉与结构质量。
- **时间连续性损失（temporal loss，即 `cbam_full`）无可测收益**，因此从最终目标函数中排除。
- 最终**部署 / "Ours" 采用 `cbam_boundary` 配置**：CBAM + 梯度损失 + 边界平滑损失。

## 训练细节

- 模型：`FNO_CBAM_Ablation(out_size=(451,351), modes1=80, modes2=64, width=64, depth=6)`，
  通过 `use_cbam` 开关切换是否启用 CBAM（关闭时参数仍分配，以便从 CBAM 基座加载权重，
  DDP 需 `find_unused_parameters=True`）。
- 数据：`jaxa_knn_filled`，训练 Series 0–7，验证 Series 8；30 天窗口，20% 方形挖空。
- Output Composition + 仅在 `loss_mask` 区域计算损失（与生产脚本一致）。
- 优化器 `AdamW(lr=5e-4, weight_decay=1e-4)`，`CosineAnnealingLR(eta_min=1e-6)`，
  4 卡 DDP，每卡 batch size 2（有效 batch = 8），默认 100 epoch。
- 归一化复用 OSTIA 参数：`mean=299.9221, std=2.6919`。

## 用法

```bash
# 训练单个变体（默认 GPU 4-7，100 epoch，约 19 小时）
python ablation/train_ablation_ddp.py --variant cbam_boundary \
    --gpu-ids 4,5,6,7 --port 29610 --epochs 100

# 每个变体使用不同的 DDP 端口，可在两组互不重叠的 GPU 上并行训练
# （例如 0,1,2,3 与 4,5,6,7）。

# 顺序训练多个变体
VARIANTS="fno_only fno_grad cbam_basic cbam_boundary cbam_full" \
    GPUS="4,5,6,7" bash ablation/run_all_variants.sh
```

`run_all_variants.sh` 默认训练 `fno_only cbam_basic cbam_boundary cbam_full`，
可通过 `VARIANTS`、`GPUS`、`EPOCHS`、`PORT_START` 等环境变量覆盖。
</content>
