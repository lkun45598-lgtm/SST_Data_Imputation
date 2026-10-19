#!/bin/bash
# 在 A100 上批量训练 23 个小时的 clean 版微调模型 (h00-h11, h13-h23, 跳过 h12)
# 方案: 每模型 8 卡 DDP, 串行训练 23 个
# clean 配置: --mask-mode realcloud --epochs 60 (save_dir 自动 = jaxa_finetune_hHH_realmask_clean)

REPO=/home/lz/SST_Data_Imputation/Data_Imputation
PY=/home/lz/miniconda3/envs/pytorch/bin/python
SCRIPT=$REPO/training/train_jaxa_hourly.py
LOGDIR=$REPO/experiments/hourly_data
EPOCHS=60
BATCH=2
NGPU=8

cd $REPO

echo "=========================================="
echo "A100 批量训练 clean 版 (23 小时, 跳过 h12)"
echo "每模型 ${NGPU} 卡, 串行, ${EPOCHS} epochs"
echo "开始: $(date)"
echo "=========================================="

for h in $(seq 0 23); do
    [ $h -eq 12 ] && continue          # h12 已有, 跳过
    hh=$(printf "%02d" $h)
    save=$REPO/experiments/jaxa_finetune_h${hh}_realmask_clean
    log=$LOGDIR/h${hh}_clean_train.log

    if [ -f "$save/best_model.pth" ]; then
        echo "[$(date '+%m-%d %H:%M')] h${hh}: 已有 best_model.pth, 跳过"
        continue
    fi

    echo "[$(date '+%m-%d %H:%M')] h${hh}: 开始训练 (${NGPU}卡) -> $log"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=29509 \
        $PY $SCRIPT --hour $h --mask-mode realcloud --epochs $EPOCHS \
        --batch-size $BATCH --gpus $NGPU > "$log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M')] h${hh}: 完成 ✓"
    else
        echo "[$(date '+%m-%d %H:%M')] h${hh}: 失败 (exit=$?)"
    fi
done

echo "=========================================="
echo "全部训练完成: $(date)"
echo "=========================================="
for h in $(seq 0 23); do
    [ $h -eq 12 ] && continue
    hh=$(printf "%02d" $h)
    m=$REPO/experiments/jaxa_finetune_h${hh}_realmask_clean/best_model.pth
    [ -f "$m" ] && echo "  h${hh}: ✓ $(du -h $m|cut -f1)" || echo "  h${hh}: ✗ 未生成"
done
