#!/usr/bin/env bash
# Run all 4 ablation variants sequentially on GPUs 4,5,6,7.
# Each variant: 100 epochs, ~13h. Total: ~52h.
# Logs and checkpoints land in ablation/experiments/{variant}/

set -e
cd "$(dirname "$0")/.."

source /home/lz/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

GPUS="4,5,6,7"
EPOCHS=100
PORT_START=29620
LOG_DIR="ablation/experiments"
mkdir -p "$LOG_DIR"

VARIANTS=(fno_only cbam_basic cbam_boundary cbam_full)

for i in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$i]}"
    port=$((PORT_START + i))
    echo
    echo "================================================"
    echo "[$(date '+%F %T')] Starting variant: $variant"
    echo "================================================"
    python -u ablation/train_ablation_ddp.py \
        --variant "$variant" \
        --gpu-ids "$GPUS" \
        --epochs "$EPOCHS" \
        --port "$port" \
        2>&1 | tee "$LOG_DIR/${variant}/train.log"
    echo "[$(date '+%F %T')] DONE variant: $variant"
done

echo
echo "All 4 variants complete!"
