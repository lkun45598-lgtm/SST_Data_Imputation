# Ablation experiments

Self-contained directory for ablation training. **Does not modify** the main
model code in `Data_Imputation/models/fno_cbam_temporal.py` or the production
training script — those are frozen for the main result.

```
Data_Imputation/ablation/
├── models/
│   └── fno_cbam_ablation.py     # FNO_CBAM_Ablation with use_cbam toggle
├── train_ablation_ddp.py         # 4-GPU DDP training, one variant at a time
├── experiments/                  # variant checkpoints + history (gitignored)
│   ├── fno_only/
│   ├── cbam_basic/
│   ├── cbam_boundary/
│   └── cbam_full/
└── README.md
```

## Variants

| Name            | use_cbam | α_grad | α_temporal | α_boundary | Description                              |
|-----------------|----------|--------|------------|------------|------------------------------------------|
| `fno_only`      | False    | 0.0    | 0.0        | 0.0        | Pure FNO, MSE only                       |
| `cbam_basic`    | True     | 0.2    | 0.0        | 0.0        | + CBAM attention, + gradient loss        |
| `cbam_boundary` | True     | 0.2    | 0.0        | 0.1        | + boundary smoothness loss               |
| `cbam_full`     | True     | 0.2    | 0.1        | 0.1        | + temporal consistency (== main model)   |

All variants train from the same OSTIA pretrain checkpoint with identical data,
batch size, LR schedule, and epoch count for fair comparison.

## Usage

```bash
# Train one variant on GPUs 4-7 (100 epochs, ~19h)
python ablation/train_ablation_ddp.py --variant cbam_full \
    --gpu-ids 4,5,6,7 --port 29610 --epochs 100

# Each variant uses a different DDP port to allow parallel training on
# two non-overlapping GPU groups (e.g. 0,1,2,3 and 4,5,6,7).
```
