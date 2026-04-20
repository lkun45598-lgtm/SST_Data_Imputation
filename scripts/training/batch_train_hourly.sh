#!/bin/bash
# 批量微调 h=2 到 h=23 的JAXA逐时模型
# 每个小时用4张GPU (1,3,4,5) 训练50个epoch，串行执行

export CUDA_VISIBLE_DEVICES=1,3,4,5
PYTHON=/home/lz/miniconda3/envs/pytorch/bin/python
SCRIPT=/data1/user/lz/SST_Data_Imputation/Data_Imputation/training/train_jaxa_hourly.py
LOG_DIR=/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data

EPOCHS=50
BATCH_SIZE=2

echo "=========================================="
echo "批量微调: h=2 到 h=23"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}/GPU"
echo "开始时间: $(date)"
echo "=========================================="

for h in $(seq 2 23); do
    hh=$(printf "%02d" $h)
    DATA_DIR="${LOG_DIR}/h${hh}"
    SAVE_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h${hh}"
    LOG_FILE="${LOG_DIR}/h${hh}_train.log"

    # 检查数据是否存在
    H5_COUNT=$(ls "${DATA_DIR}"/*.h5 2>/dev/null | wc -l)
    if [ "$H5_COUNT" -lt 9 ]; then
        echo "[$(date '+%m-%d %H:%M')] h=${hh}: 数据不完整 (${H5_COUNT}/9)，跳过"
        continue
    fi

    # 检查是否已经训练过
    if [ -f "${SAVE_DIR}/best_model.pth" ]; then
        echo "[$(date '+%m-%d %H:%M')] h=${hh}: 已有best_model.pth，跳过"
        continue
    fi

    echo "[$(date '+%m-%d %H:%M')] h=${hh}: 开始训练 -> ${LOG_FILE}"
    $PYTHON $SCRIPT --hour $h --epochs $EPOCHS --batch-size $BATCH_SIZE > "$LOG_FILE" 2>&1
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        # 提取最优MAE
        BEST_MAE=$(grep "最优模型已保存" "$LOG_FILE" | tail -1 | grep -oP 'MAE: \K[0-9.]+')
        echo "[$(date '+%m-%d %H:%M')] h=${hh}: 完成! Best MAE=${BEST_MAE}K"
    else
        echo "[$(date '+%m-%d %H:%M')] h=${hh}: 训练失败 (exit=$STATUS)"
    fi
done

echo ""
echo "=========================================="
echo "全部训练完成! $(date)"
echo "=========================================="

# 汇总
echo ""
echo "模型汇总:"
for h in $(seq 2 23); do
    hh=$(printf "%02d" $h)
    MODEL="/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h${hh}/best_model.pth"
    if [ -f "$MODEL" ]; then
        SIZE=$(du -h "$MODEL" | cut -f1)
        echo "  h${hh}: ${SIZE} ✓"
    else
        echo "  h${hh}: 未生成"
    fi
done
