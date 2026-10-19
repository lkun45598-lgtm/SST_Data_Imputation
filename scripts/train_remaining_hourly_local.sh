#!/bin/bash
# 在本地4090服务器训练剩余hourly模型
# GPU 4-7 (4张卡)
# 需要训练: h03-h11, h13-h23 (共19个, 用正确的Jul 17 OSTIA权重)

set -e
REPO=/data1/user/lz/SST_Data_Imputation/Data_Imputation
PY=/home/lz/miniconda3/envs/pytorch/bin/python
SCRIPT=$REPO/training/train_jaxa_hourly.py

# 需要训练的小时 (跳过已有正确权重的 h00,h01,h02,h12)
HOURS=(05 06 07 08 09 10 11 13 14 15 16 17 18 19 20 21 22 23)

# GPU配置
GPUS="0,1,6,7"
NUM_GPUS=4

# 训练参数
EPOCHS=60
BATCH_SIZE=2
USE_CBAM="--use-cbam"
ALPHA_GRAD=0.2
ALPHA_BOUNDARY=0.1
ALPHA_TEMPORAL=0.0

LOG_DIR=$REPO/experiments/hourly_data
mkdir -p $LOG_DIR

echo "[$(date)] ===== 开始训练剩余hourly模型 ====="
echo "GPU: $GPUS ($NUM_GPUS 张卡)"
echo "共 ${#HOURS[@]} 个模型"
echo "OSTIA权重: $REPO/experiments/ostia_pretrain_clean/best_model.pth"
ls -lh $REPO/experiments/ostia_pretrain_clean/best_model.pth

for H in "${HOURS[@]}"; do
    HOUR=$(echo $H | sed 's/^0*//')  # 去前导0: 03->3
    EXP=$REPO/experiments/jaxa_finetune_h${H}_realmask_clean
    LOG=$LOG_DIR/h${H}_cbam_train.log

    # 已完整训练(有final_model)则跳过, 保护已完成的模型
    if [ -f "$EXP/final_model.pth" ]; then
        echo "[$(date +%H:%M)] h${H} 已完整训练(有final_model), 跳过"
        continue
    fi

    # 删除不完整的旧模型(只有checkpoint没final)
    if [ -d "$EXP" ]; then
        echo "[$(date +%H:%M)] 删除不完整的旧 h${H}"
        rm -rf "$EXP"
    fi

    echo "[$(date +%H:%M)] ===== 开始训练 h${H} (hour=$HOUR) ====="

    cd $REPO/training
    CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$((29500 + HOUR)) \
        $PY -u $SCRIPT \
        --hour $HOUR \
        --mask-mode realcloud \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --gpus $NUM_GPUS \
        $USE_CBAM \
        --alpha-grad $ALPHA_GRAD \
        --alpha-boundary $ALPHA_BOUNDARY \
        --alpha-temporal $ALPHA_TEMPORAL \
        > ${LOG} 2>&1

    if [ ! -f "${EXP}/best_model.pth" ]; then
        echo "[$(date +%H:%M)] h${H} 训练失败, 退出"
        exit 1
    fi

    # 清理checkpoint (省磁盘)
    rm -f ${EXP}/checkpoint_epoch_*.pth

    # 复制训练日志
    cp ${LOG} ${EXP}/train.log 2>/dev/null || true

    VAL_MAE=$(grep -oP '最优MAE: \K[0-9.]+' ${LOG} 2>/dev/null || echo 'N/A')
    echo "[$(date +%H:%M)] h${H} 完成, val_mae=$VAL_MAE"
done

echo "[$(date)] ===== 全部完成 ====="
