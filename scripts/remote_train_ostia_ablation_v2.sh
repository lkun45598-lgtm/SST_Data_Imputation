#!/bin/bash
# OSTIA 消融：在 A100 远程服务器启动训练 + 后台监控传输
# 使用方法：bash remote_train_ostia_ablation_v2.sh

set -e

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
EXP_NAME="jaxa_finetune_h12_nopretrain_realmask_clean"
REMOTE_LOG="${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}_train.log"
LOCAL_LOG="${LOCAL_DIR}/logs/ostia_ablation_monitor.log"

mkdir -p ${LOCAL_DIR}/experiments/${EXP_NAME}
mkdir -p ${LOCAL_DIR}/logs

echo "=========================================="
echo "OSTIA 消融实验：w/o pretrain (从零训练)"
echo "远程: ${REMOTE_HOST}"
echo "实验: ${EXP_NAME}"
echo "=========================================="

# 1. 启动远程训练
echo "[1/3] 启动远程训练（8卡 DDP）..."
ssh ${REMOTE_HOST} "cd ${REMOTE_DIR}/training && \
MASTER_PORT=29600 nohup /home/lz/miniconda3/envs/pytorch/bin/python -u train_jaxa_hourly.py \
  --hour 12 \
  --mask-mode realcloud \
  --epochs 60 \
  --batch-size 2 \
  --gpus 8 \
  --no-pretrain \
  > ${REMOTE_LOG} 2>&1 &
echo \$! > ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid
"

sleep 2
REMOTE_PID=$(ssh ${REMOTE_HOST} "cat ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid 2>/dev/null")
echo "✓ 远程训练已启动，PID=${REMOTE_PID}"

# 2. 启动本地监控脚本（后台运行）
echo "[2/3] 启动本地监控脚本（后台）..."

cat > /tmp/ostia_ablation_monitor.sh << 'MONITOR_EOF'
#!/bin/bash
REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
EXP_NAME="jaxa_finetune_h12_nopretrain_realmask_clean"
REMOTE_LOG="${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}_train.log"
LOCAL_LOG="${LOCAL_DIR}/logs/ostia_ablation_monitor.log"
REMOTE_PID=$(ssh ${REMOTE_HOST} "cat ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid 2>/dev/null")

exec > ${LOCAL_LOG} 2>&1

echo "[$(date)] 监控启动，远程 PID=${REMOTE_PID}"

SYNC_INTERVAL=1800  # 30分钟同步一次日志
LAST_SYNC=0

while true; do
  RUNNING=$(ssh ${REMOTE_HOST} "ps -p ${REMOTE_PID} > /dev/null 2>&1 && echo 1 || echo 0" 2>/dev/null)

  if [ "$RUNNING" != "1" ]; then
    echo "[$(date)] 训练已完成！"
    break
  fi

  NOW=$(date +%s)
  if [ $((NOW - LAST_SYNC)) -ge ${SYNC_INTERVAL} ]; then
    echo "[$(date)] 同步日志..."
    rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_DIR}/experiments/${EXP_NAME}/ 2>/dev/null || true
    LAST_SYNC=$NOW
  fi

  sleep 60
done

# 训练完成，最终传输
echo "[$(date)] 开始最终文件传输..."
rsync -avz --progress \
  ${REMOTE_HOST}:${REMOTE_DIR}/experiments/${EXP_NAME}/ \
  ${LOCAL_DIR}/experiments/${EXP_NAME}/ 2>&1

echo "[$(date)] 同步日志文件..."
rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_DIR}/experiments/${EXP_NAME}/ 2>&1

echo "[$(date)] 清理远程文件..."
ssh ${REMOTE_HOST} "rm -rf ${REMOTE_DIR}/experiments/${EXP_NAME}" 2>&1
ssh ${REMOTE_HOST} "rm -f ${REMOTE_LOG}" 2>&1
ssh ${REMOTE_HOST} "rm -f ${REMOTE_DIR}/experiments/hourly_data/${EXP_NAME}.pid" 2>&1

echo "[$(date)] ✅ 完成！文件已保存到: ${LOCAL_DIR}/experiments/${EXP_NAME}/"
ls -lh ${LOCAL_DIR}/experiments/${EXP_NAME}/ >> ${LOCAL_LOG}
MONITOR_EOF

chmod +x /tmp/ostia_ablation_monitor.sh
nohup bash /tmp/ostia_ablation_monitor.sh > /dev/null 2>&1 &
MONITOR_PID=$!

echo "✓ 监控脚本已启动（后台），PID=${MONITOR_PID}"
echo "  - 每 30 分钟同步日志"
echo "  - 训练完成后自动传回文件并清理远程"
echo "  - 监控日志: ${LOCAL_LOG}"

# 3. 显示状态
echo ""
echo "[3/3] 当前状态"
echo "=========================================="
echo "远程训练 PID: ${REMOTE_PID}"
echo "本地监控 PID: ${MONITOR_PID}"
echo ""
echo "查看远程训练日志："
echo "  ssh ${REMOTE_HOST} \"tail -f ${REMOTE_LOG}\""
echo ""
echo "查看本地监控日志："
echo "  tail -f ${LOCAL_LOG}"
echo ""
echo "查看远程进程："
echo "  ssh ${REMOTE_HOST} \"ps -p ${REMOTE_PID} -o pid,etime,cmd\""
echo ""
echo "停止监控（不影响远程训练）："
echo "  kill ${MONITOR_PID}"
echo "=========================================="

# 立即同步一次日志
sleep 5
echo ""
echo "首次日志同步..."
rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_DIR}/experiments/${EXP_NAME}/ 2>/dev/null && \
tail -20 ${LOCAL_DIR}/experiments/${EXP_NAME}/$(basename ${REMOTE_LOG}) 2>/dev/null || \
echo "(日志还未生成)"
