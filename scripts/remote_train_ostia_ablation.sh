#!/bin/bash
# 在 A100 远程服务器（10.16.46.215）训练 OSTIA 消融模型（w/o pretrain）
# 训练完成后自动传回本地，删除远程文件

set -e

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
EXP_NAME="jaxa_finetune_h12_nopretrain_realmask_clean"
LOG_FILE="${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}_train.log"

echo "=========================================="
echo "OSTIA Ablation: w/o pretrain (从零训练)"
echo "远程: ${REMOTE_HOST}"
echo "实验: ${EXP_NAME}"
echo "=========================================="

# 1. 在远程启动训练（后台运行，8卡DDP）
echo "[1/4] 启动远程训练..."
ssh ${REMOTE_HOST} "cd ${REMOTE_DIR}/training && \
nohup python -u train_jaxa_hourly.py \
  --hour 12 \
  --mask-mode realcloud \
  --epochs 60 \
  --batch-size 2 \
  --gpus 8 \
  --output-dir ${REMOTE_DIR}/experiments/${EXP_NAME} \
  --no-pretrain \
  > ${LOG_FILE} 2>&1 &
echo \"训练进程已启动，PID=\$!\"
echo \$! > ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid
"

PID=$(ssh ${REMOTE_HOST} "cat ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid 2>/dev/null || echo 'unknown'")
echo "远程 PID: ${PID}"
echo ""

# 2. 启动监控脚本（本地运行，定期检查远程进程）
echo "[2/4] 启动本地监控..."
echo "监控脚本将每 30 分钟检查一次远程进程并同步文件"
echo "按 Ctrl+C 停止监控（不影响远程训练）"
echo ""

# 创建本地实验目录
mkdir -p ${LOCAL_DIR}/experiments/${EXP_NAME}

# 监控循环
SYNC_INTERVAL=1800  # 30分钟
CHECK_INTERVAL=60   # 1分钟检查进程状态

while true; do
  # 检查远程进程是否还在运行
  RUNNING=$(ssh ${REMOTE_HOST} "ps -p ${PID} > /dev/null 2>&1 && echo 1 || echo 0")

  if [ "$RUNNING" == "1" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练进行中..."

    # 每 30 分钟同步一次日志（不传模型，避免频繁传大文件）
    if [ $(($(date +%s) % SYNC_INTERVAL)) -lt CHECK_INTERVAL ]; then
      echo "  → 同步日志..."
      rsync -avz --progress \
        ${REMOTE_HOST}:${LOG_FILE} \
        ${LOCAL_DIR}/experiments/${EXP_NAME}/ 2>/dev/null || true
    fi

    sleep ${CHECK_INTERVAL}
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练已完成！"
    break
  fi
done

# 3. 训练完成，传回文件并清理远程
echo ""
echo "[3/4] 传回文件到本地..."
rsync -avz --progress \
  ${REMOTE_HOST}:${REMOTE_DIR}/experiments/${EXP_NAME}/ \
  ${LOCAL_DIR}/experiments/${EXP_NAME}/

echo ""
echo "[4/4] 删除远程文件（释放 A100 空间）..."
ssh ${REMOTE_HOST} "rm -rf ${REMOTE_DIR}/experiments/${EXP_NAME}"
ssh ${REMOTE_HOST} "rm -f ${LOG_FILE}"
ssh ${REMOTE_HOST} "rm -f ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid"

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "本地路径: ${LOCAL_DIR}/experiments/${EXP_NAME}/"
echo "文件列表:"
ls -lh ${LOCAL_DIR}/experiments/${EXP_NAME}/
echo "=========================================="
