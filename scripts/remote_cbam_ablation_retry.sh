#!/bin/bash
# 远程 A100：依次训练 4 个 CBAM 消融实验
# 每个完成后自动传回本地并删除远程文件

set -e

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
HOUR=12
BASE_PORT=29900  # 避开 h09=默认端口

# 4 个 CBAM 消融变体
declare -a VARIANTS=(
  "fno_only:false:0:0:0"
  "cbam_basic:true:0.2:0:0"
  "cbam_boundary:true:0.2:0.1:0"
  "cbam_temporal:true:0.2:0.1:0.1"
)

echo "=========================================="
echo "远程 A100 - CBAM 消融实验（4 个变体）"
echo "=========================================="
echo ""

mkdir -p ${LOCAL_DIR}/logs
MASTER_LOG="${LOCAL_DIR}/logs/cbam_ablation_remote.log"
exec > >(tee -a ${MASTER_LOG}) 2>&1

START_TIME=$(date +%s)

for i in "${!VARIANTS[@]}"; do
  IFS=':' read -r NAME USE_CBAM ALPHA_GRAD ALPHA_BOUNDARY ALPHA_TEMPORAL <<< "${VARIANTS[$i]}"

  EXP_NAME="jaxa_finetune_h${HOUR}_${NAME}_realmask_clean"
  REMOTE_LOG="${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}_train.log"
  PORT=$((BASE_PORT + i))

  echo ""
  echo "=========================================="
  echo "[$((i+1))/4] 启动: ${NAME}"
  echo "=========================================="
  echo "配置："
  echo "  - use_cbam: ${USE_CBAM}"
  echo "  - alpha_grad: ${ALPHA_GRAD}"
  echo "  - alpha_boundary: ${ALPHA_BOUNDARY}"
  echo "  - alpha_temporal: ${ALPHA_TEMPORAL}"
  echo "  - DDP port: ${PORT}"
  echo ""

  # 构建命令
  CBAM_FLAG=""
  if [ "$USE_CBAM" = "true" ]; then
    CBAM_FLAG="--use-cbam"
  fi

  # 启动远程训练
  echo "[$(date '+%H:%M:%S')] 启动远程训练..."
  ssh ${REMOTE_HOST} "cd ${REMOTE_DIR}/training && \
MASTER_PORT=${PORT} nohup /home/lz/miniconda3/envs/pytorch/bin/python -u train_jaxa_hourly.py \
  --hour ${HOUR} \
  --mask-mode realcloud \
  --epochs 60 \
  --batch-size 2 \
  --gpus 8 \
  --save-suffix _${NAME}_realmask \
  ${CBAM_FLAG} \
  --alpha-grad ${ALPHA_GRAD} \
  --alpha-boundary ${ALPHA_BOUNDARY} \
  --alpha-temporal ${ALPHA_TEMPORAL} \
  > ${REMOTE_LOG} 2>&1 &
echo \$! > ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid
"

  sleep 5
  REMOTE_PID=$(ssh ${REMOTE_HOST} "cat ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid 2>/dev/null")

  if [ -z "$REMOTE_PID" ]; then
    echo "✗ 启动失败：无法获取 PID"
    exit 1
  fi

  echo "✓ 远程训练已启动，PID=${REMOTE_PID}"

  # 等待训练完成
  echo "[$(date '+%H:%M:%S')] 等待训练完成..."
  ELAPSED=0
  while true; do
    RUNNING=$(ssh ${REMOTE_HOST} "ps -p ${REMOTE_PID} > /dev/null 2>&1 && echo 1 || echo 0" 2>/dev/null)

    if [ "$RUNNING" != "1" ]; then
      echo "[$(date '+%H:%M:%S')] 训练已完成！"
      break
    fi

    # 每 10 分钟打印进度
    if [ $((ELAPSED % 600)) -eq 0 ]; then
      EPOCH_INFO=$(ssh ${REMOTE_HOST} "tail -1 ${REMOTE_LOG} 2>/dev/null | grep -oP 'Epoch \K\d+'" || echo "?")
      echo "[$(date '+%H:%M:%S')] 训练进行中... (Epoch ${EPOCH_INFO}/60, 已耗时 $((ELAPSED/60)) 分钟)"
    fi

    sleep 60
    ELAPSED=$((ELAPSED + 60))
  done

  # 传回文件
  echo "[$(date '+%H:%M:%S')] 传回文件到本地..."
  mkdir -p ${LOCAL_DIR}/ablation/experiments/${NAME}

  rsync -avz --progress \
    ${REMOTE_HOST}:${REMOTE_DIR}/experiments/${EXP_NAME}/ \
    ${LOCAL_DIR}/ablation/experiments/${NAME}/ || echo "  ⚠️ rsync 失败，尝试继续"

  rsync -az \
    ${REMOTE_HOST}:${REMOTE_LOG} \
    ${LOCAL_DIR}/ablation/experiments/${NAME}/ || true

  # 清理远程
  echo "[$(date '+%H:%M:%S')] 清理远程文件..."
  ssh ${REMOTE_HOST} "rm -rf ${REMOTE_DIR}/experiments/${EXP_NAME}" || true
  ssh ${REMOTE_HOST} "rm -f ${REMOTE_LOG}" || true
  ssh ${REMOTE_HOST} "rm -f ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid" || true

  echo "✓ [$((i+1))/4] ${NAME} 完成！"

  if [ -f "${LOCAL_DIR}/ablation/experiments/${NAME}/best_model.pth" ]; then
    SIZE=$(ls -lh ${LOCAL_DIR}/ablation/experiments/${NAME}/best_model.pth | awk '{print $5}')
    echo "  模型大小: ${SIZE}"
  else
    echo "  ⚠️ 未找到 best_model.pth"
  fi
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "=========================================="
echo "✅ 全部 CBAM 消融完成！"
echo "=========================================="
echo "总耗时: ${HOURS} 小时 ${MINUTES} 分钟"
echo "结果位置: ${LOCAL_DIR}/ablation/experiments/"
echo "=========================================="
