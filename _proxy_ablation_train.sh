#!/usr/bin/env bash
# Proxy construction 消融训练自动启动器
# 数据齐(9 series)后自动开训, 两个模型顺序共用 GPU 1,5,6,7, 各 60 epoch, batch2×4卡=有效8
set -u
REPO=/data1/user/lz/SST_Data_Imputation/Data_Imputation
cd "$REPO/training"
GPUS_STR="1,5,6,7"
NGPU=4
BATCH=2
EPOCHS=60

wait_data () {
  local sfx="$1"
  local dir="$REPO/experiments/hourly_data/h12${sfx}"
  echo "[$(date '+%F %T')] 等待数据 $dir (需9个h5)..."
  while true; do
    local n; n=$(ls "$dir"/jaxa_knn_filled_*.h5 2>/dev/null | wc -l)
    if [ "$n" -ge 9 ]; then echo "[$(date '+%F %T')] $dir 就绪 ($n/9)"; sleep 15; return; fi
    sleep 60
  done
}

train () {
  local sfx="$1" port="$2"
  echo "[$(date '+%F %T')] 开始训练 h12${sfx} (GPU $GPUS_STR, batch$BATCH, ${EPOCHS}ep, port $port)"
  CUDA_VISIBLE_DEVICES="$GPUS_STR" MASTER_PORT="$port" \
    python3 train_jaxa_hourly.py --hour 12 --data-suffix "$sfx" \
      --mask-mode realcloud --epochs "$EPOCHS" --gpus "$NGPU" --batch-size "$BATCH"
  echo "[$(date '+%F %T')] 训练结束 h12${sfx} (exit=$?)"
  echo "  -> best: $REPO/experiments/jaxa_finetune_h12${sfx}_realmask_clean/best_model.pth"
  ls -lh "$REPO/experiments/jaxa_finetune_h12${sfx}_realmask_clean/best_model.pth" 2>/dev/null
}

echo "===== Proxy 消融训练启动器 $(date '+%F %T') ====="
wait_data "_noknn"
train "_noknn" 29531
wait_data "_nogauss"
train "_nogauss" 29532
echo "===== 两个消融模型全部训练完成 $(date '+%F %T') ====="
