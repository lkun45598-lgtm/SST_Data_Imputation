#!/bin/bash
# 在 A100 上批量重训 23 个小时模型 (h00-h11, h13-h23, 跳过h12) 为【有CBAM完整配置】
# 配置: --use-cbam --alpha-grad 0.2 --alpha-boundary 0.1 --alpha-temporal 0.0 (与主结果h12一致)
# 每个模型训完立即 rsync 回本地并清理远程该模型的中间文件
# 会等当前 h12 训练结束后再开始 (避免抢8卡)
set -u

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
PY="/home/lz/miniconda3/envs/pytorch/bin/python"
SCRIPT="${REMOTE_DIR}/training/train_jaxa_hourly.py"
LOGDIR="${REMOTE_DIR}/experiments/hourly_data"

HOURS="00 01 02 03 04 05 06 07 08 09 10 11 13 14 15 16 17 18 19 20 21 22 23"  # 跳过12
EPOCHS=60
BATCH=2
NGPU=8

echo "========================================"
echo "[$(date '+%m-%d %H:%M')] A100 批量重训 23 小时模型 (有CBAM)"
echo "  配置: use-cbam + grad0.2 + boundary0.1 + temporal0.0"
echo "========================================"

# ---- 0. 等 h12 当前训练结束 (避免抢卡) ----
# 用 ps+grep 双重排除, 避免 SSH 命令自身被匹配 (pgrep -f 会匹配到含相同字符串的ssh命令)
echo "[$(date '+%H:%M')] 检查 A100 是否有训练进程..."
while ssh ${REMOTE_HOST} "ps aux | grep 'python.*train_jaxa_hourly' | grep -v grep | grep -qv 'bash -c'"; do
  echo "[$(date '+%H:%M')] A100 仍有训练进程, 等待 10 分钟..."
  sleep 600
done
echo "[$(date '+%H:%M')] A100 空闲, 开始批量训练"

# ---- 1. 逐个训练 ----
for h in ${HOURS}; do
  hh=$(printf "%02d" $((10#$h)))
  EXP_NAME="jaxa_finetune_h${hh}_realmask_clean"
  REMOTE_EXP="${REMOTE_DIR}/experiments/${EXP_NAME}"
  REMOTE_LOG="${LOGDIR}/h${hh}_cbam_train.log"
  LOCAL_EXP="${LOCAL_DIR}/experiments/${EXP_NAME}"

  echo ""
  echo "[$(date '+%m-%d %H:%M')] ===== h${hh} 开始训练 ====="

  # 训练 (前台阻塞直到该模型训完)
  ssh ${REMOTE_HOST} "cd ${REMOTE_DIR}/training && MASTER_PORT=29520 ${PY} -u ${SCRIPT} \
    --hour ${hh} --mask-mode realcloud --epochs ${EPOCHS} \
    --batch-size ${BATCH} --gpus ${NGPU} \
    --use-cbam --alpha-grad 0.2 --alpha-boundary 0.1 --alpha-temporal 0.0 \
    > ${REMOTE_LOG} 2>&1"

  # 验证远程 best_model.pth
  if ! ssh ${REMOTE_HOST} "test -f ${REMOTE_EXP}/best_model.pth"; then
    echo "[$(date '+%H:%M')] ❌ h${hh} 未生成 best_model.pth, 跳过回传 (保留日志)"
    rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_DIR}/experiments/hourly_data/ || true
    continue
  fi

  # 清理远程中间 checkpoint (只留 best/final/config/history)
  ssh ${REMOTE_HOST} "rm -f ${REMOTE_EXP}/checkpoint_epoch_*.pth" 2>/dev/null || true

  # 回传本地 (覆盖无CBAM旧版)
  echo "[$(date '+%H:%M')] h${hh} 回传本地..."
  mkdir -p ${LOCAL_EXP}
  rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/best_model.pth ${LOCAL_EXP}/
  rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/config.json ${LOCAL_EXP}/ 2>/dev/null || true
  rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/training_history.json ${LOCAL_EXP}/ 2>/dev/null || true
  rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_DIR}/experiments/hourly_data/ || true

  # 验证本地 + 大小合理性
  BSIZE=$(stat -c%s "${LOCAL_EXP}/best_model.pth" 2>/dev/null || echo 0)
  if [ -f "${LOCAL_EXP}/best_model.pth" ] && [ "${BSIZE}" -gt 1000000000 ]; then
    CBAM=$(${PY:-python3} -c "import json;print(json.load(open('${LOCAL_EXP}/config.json')).get('use_cbam'))" 2>/dev/null || echo "?")
    echo "[$(date '+%H:%M')] ✓ h${hh} 本地OK ($(ls -lh ${LOCAL_EXP}/best_model.pth|awk '{print $5}'), cbam=${CBAM}), 清理远程"
    ssh ${REMOTE_HOST} "rm -rf ${REMOTE_EXP}" || true
    ssh ${REMOTE_HOST} "rm -f ${REMOTE_LOG}" || true
  else
    echo "[$(date '+%H:%M')] ⚠️ h${hh} 本地验证失败, 保留远程文件"
  fi
done

echo ""
echo "========================================"
echo "[$(date '+%m-%d %H:%M')] 全部完成! 本地检查:"
for h in ${HOURS}; do
  hh=$(printf "%02d" $((10#$h)))
  m="${LOCAL_DIR}/experiments/jaxa_finetune_h${hh}_realmask_clean/best_model.pth"
  [ -f "$m" ] && echo "  h${hh}: ✓ $(ls -lh $m|awk '{print $5,$6,$7}')" || echo "  h${hh}: ✗"
done
echo "========================================"
