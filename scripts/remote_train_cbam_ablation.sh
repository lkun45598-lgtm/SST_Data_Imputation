#!/bin/bash
# CBAM 消融：在 A100 远程服务器依次训练 4 个变体
# 每个完成后自动传回本地并删除远程文件

set -e

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
HOUR=12
BASE_PORT=29700  # 起始端口（避开 h05=29509, h12_nopretrain=29600）

# 4 个消融变体配置：名称 use_cbam alpha_grad alpha_boundary alpha_temporal
declare -a VARIANTS=(
  "fno_only:false:0:0:0"
  "cbam_basic:true:0.2:0:0"
  "cbam_boundary:true:0.2:0.1:0"
  "cbam_temporal:true:0.2:0.1:0.1"
)

echo "=========================================="
echo "CBAM 消融实验：4 个变体（realcloud + 60ep）"
echo "远程: ${REMOTE_HOST}"
echo "预计总时间: ~16 小时（每个 ~4 小时）"
echo "=========================================="
echo ""

mkdir -p ${LOCAL_DIR}/logs
MASTER_LOG="${LOCAL_DIR}/logs/cbam_ablation_master.log"
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
  echo "  - 实验目录: ${EXP_NAME}"
  echo ""

  # 构建命令行参数
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
  ${CBAM_FLAG} \
  --alpha-grad ${ALPHA_GRAD} \
  --alpha-boundary ${ALPHA_BOUNDARY} \
  --alpha-temporal ${ALPHA_TEMPORAL} \
  > ${REMOTE_LOG} 2>&1 &
echo \$! > ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid
"

  sleep 3
  REMOTE_PID=$(ssh ${REMOTE_HOST} "cat ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid 2>/dev/null")

  if [ -z "$REMOTE_PID" ]; then
    echo "✗ 启动失败：无法获取 PID"
    exit 1
  fi

  echo "✓ 远程训练已启动，PID=${REMOTE_PID}"

  # 等待训练完成（每分钟检查一次）
  echo "[$(date '+%H:%M:%S')] 等待训练完成..."
  ELAPSED=0
  while true; do
    RUNNING=$(ssh ${REMOTE_HOST} "ps -p ${REMOTE_PID} > /dev/null 2>&1 && echo 1 || echo 0" 2>/dev/null)

    if [ "$RUNNING" != "1" ]; then
      echo "[$(date '+%H:%M:%S')] 训练已完成！"
      break
    fi

    # 每 10 分钟打印一次进度
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
    ${LOCAL_DIR}/ablation/experiments/${NAME}/

  rsync -az \
    ${REMOTE_HOST}:${REMOTE_LOG} \
    ${LOCAL_DIR}/ablation/experiments/${NAME}/

  # 清理远程
  echo "[$(date '+%H:%M:%S')] 清理远程文件..."
  ssh ${REMOTE_HOST} "rm -rf ${REMOTE_DIR}/experiments/${EXP_NAME}" || true
  ssh ${REMOTE_HOST} "rm -f ${REMOTE_LOG}" || true
  ssh ${REMOTE_HOST} "rm -f ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid" || true

  echo "✓ [$((i+1))/4] ${NAME} 完成！"
  echo "  本地路径: ${LOCAL_DIR}/ablation/experiments/${NAME}/"
  ls -lh ${LOCAL_DIR}/ablation/experiments/${NAME}/best_model.pth 2>/dev/null || echo "  (未找到 best_model.pth)"
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "=========================================="
echo "✅ CBAM 消融全部完成！"
echo "=========================================="
echo "总耗时: ${HOURS} 小时 ${MINUTES} 分钟"
echo "结果位置: ${LOCAL_DIR}/ablation/experiments/"
echo ""
echo "下一步："
echo "  1. 评估 4 个模型（Series 8, 3 levels）"
echo "  2. 生成 Table 并写入论文"
echo "=========================================="
