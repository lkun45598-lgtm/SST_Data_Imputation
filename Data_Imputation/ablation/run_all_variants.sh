#!/usr/bin/env bash
# Run ablation variants sequentially on 4 GPUs.
#
# Defaults: train all 4 variants. Override with VARIANTS env var, e.g.:
#   VARIANTS="cbam_basic cbam_boundary cbam_full" bash run_all_variants.sh
# Override GPUs with:
#   GPUS="0,1,2,3" bash run_all_variants.sh

set -uo pipefail           # NOTE: no -e — we want to keep going on per-variant errors
cd "$(dirname "$0")/.."

source /home/lz/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

GPUS="${GPUS:-4,5,6,7}"
EPOCHS="${EPOCHS:-100}"
PORT_START="${PORT_START:-29620}"
LOG_DIR="ablation/experiments"
DEFAULT_VARIANTS=(fno_only cbam_basic cbam_boundary cbam_full)
VARIANTS=(${VARIANTS:-${DEFAULT_VARIANTS[@]}})

mkdir -p "$LOG_DIR"

echo "================================================"
echo "[$(date '+%F %T')] Ablation launcher starting"
echo "  GPUS=$GPUS  EPOCHS=$EPOCHS  variants=${VARIANTS[*]}"
echo "================================================"

for i in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$i]}"
    port=$((PORT_START + i))
    variant_dir="$LOG_DIR/$variant"
    mkdir -p "$variant_dir"   # prevent tee path errors

    echo
    echo "================================================"
    echo "[$(date '+%F %T')] Starting variant: $variant  (port=$port)"
    echo "================================================"

    python -u ablation/train_ablation_ddp.py \
        --variant "$variant" \
        --gpu-ids "$GPUS" \
        --epochs "$EPOCHS" \
        --port "$port" \
        2>&1 | tee "$variant_dir/train.log"
    rc=${PIPESTATUS[0]}

    if [[ $rc -eq 0 ]]; then
        echo "[$(date '+%F %T')] ✓ DONE variant: $variant (rc=0)"
    else
        echo "[$(date '+%F %T')] ✗ FAILED variant: $variant (rc=$rc) — continuing to next"
    fi
done

echo
echo "================================================"
echo "[$(date '+%F %T')] All requested variants finished"
echo "================================================"
