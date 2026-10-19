#!/bin/bash
# 监控远程 A100 上的 h12 训练:
#   1. 训练中: 周期性删除 checkpoint_epoch_*.pth (只保留 best_model.pth), 省远程空间
#   2. 训练完成后: 回传 best_model.pth + log + config + history 到本地
#   3. 回传验证成功后: 删除 A100 上的整个实验目录 + 日志
set -u

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"

EXP_NAME="jaxa_finetune_h12_realmask_clean"
REMOTE_EXP="${REMOTE_DIR}/experiments/${EXP_NAME}"
REMOTE_LOG="${REMOTE_DIR}/experiments/hourly_data/jaxa_finetune_h12_retrain.log"
LOCAL_EXP="${LOCAL_DIR}/experiments/${EXP_NAME}"
LOCAL_LOGDIR="${LOCAL_DIR}/experiments/hourly_data"

CHECK_INTERVAL=300   # 每 5 分钟检查一次

echo "========================================"
echo "[$(date '+%H:%M:%S')] 开始监控 h12 训练"
echo "  远程: ${REMOTE_EXP}"
echo "  本地: ${LOCAL_EXP}"
echo "  策略: 训练中清理中间checkpoint, 完成后回传best+log, 再清理远程"
echo "========================================"

# ---------- 阶段 1+2: 循环监控 ----------
while true; do
  # 检查训练进程是否还在
  TRAIN_RUNNING=$(ssh ${REMOTE_HOST} "pgrep -f 'train_jaxa_hourly.py --hour 12' | head -1" 2>/dev/null)

  # 检查是否出现"训练完成"标志
  DONE_FLAG=$(ssh ${REMOTE_HOST} "grep -a -c '训练完成' ${REMOTE_LOG} 2>/dev/null || echo 0")

  # 当前进度
  CUR_EPOCH=$(ssh ${REMOTE_HOST} "grep -a -oE 'Epoch [0-9]+/60' ${REMOTE_LOG} 2>/dev/null | tail -1")
  BEST_MAE=$(ssh ${REMOTE_HOST} "grep -a -oE '最优模型已保存! MAE: [0-9.]+K' ${REMOTE_LOG} 2>/dev/null | tail -1")

  if [ "${DONE_FLAG}" != "0" ] && [ -z "${TRAIN_RUNNING}" ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 检测到训练完成! (${CUR_EPOCH}, ${BEST_MAE})"
    break
  fi

  # 阶段1: 训练中清理中间 checkpoint (保留 best_model.pth / final / config / history)
  echo "[$(date '+%H:%M:%S')] 训练中... ${CUR_EPOCH} ${BEST_MAE}; 清理远程中间checkpoint"
  ssh ${REMOTE_HOST} "rm -f ${REMOTE_EXP}/checkpoint_epoch_*.pth" 2>/dev/null || true

  sleep ${CHECK_INTERVAL}
done

# ---------- 阶段 3: 回传 ----------
# 完成后再清理一次中间 checkpoint (final阶段可能又写了 checkpoint_epoch_60)
echo "[$(date '+%H:%M:%S')] 清理远程中间checkpoint (保留best/final/config/history)..."
ssh ${REMOTE_HOST} "rm -f ${REMOTE_EXP}/checkpoint_epoch_*.pth" 2>/dev/null || true

# 验证远程 best_model.pth 存在
if ! ssh ${REMOTE_HOST} "test -f ${REMOTE_EXP}/best_model.pth"; then
  echo "❌ 远程无 best_model.pth, 中止回传! 检查日志: ${REMOTE_LOG}"
  exit 1
fi

mkdir -p ${LOCAL_EXP}
echo "[$(date '+%H:%M:%S')] 回传 best_model.pth ..."
rsync -az --progress ${REMOTE_HOST}:${REMOTE_EXP}/best_model.pth ${LOCAL_EXP}/

echo "[$(date '+%H:%M:%S')] 回传 config.json / training_history.json / final_model.pth ..."
rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/config.json ${LOCAL_EXP}/ 2>/dev/null || true
rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/training_history.json ${LOCAL_EXP}/ 2>/dev/null || true
rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/final_model.pth ${LOCAL_EXP}/ 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] 回传训练日志 ..."
rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_LOGDIR}/

# ---------- 阶段 4: 验证 + 清理远程 ----------
echo "[$(date '+%H:%M:%S')] 验证本地文件..."
OK=1
for f in best_model.pth config.json training_history.json; do
  if [ -f "${LOCAL_EXP}/${f}" ]; then
    echo "  ✓ ${f} ($(ls -lh ${LOCAL_EXP}/${f} | awk '{print $5}'))"
  else
    echo "  ✗ ${f} 缺失!"
    OK=0
  fi
done

# 确认 config 是正常模型
echo "[$(date '+%H:%M:%S')] 确认模型配置:"
python3 -c "import json; c=json.load(open('${LOCAL_EXP}/config.json')); print('  use_cbam:', c.get('use_cbam'), '| no_pretrain:', c.get('no_pretrain', False), '| alpha_grad:', c.get('alpha_grad'), '| alpha_boundary:', c.get('alpha_boundary'))" 2>/dev/null || OK=0

# best_model.pth 大小合理性检查 (应 ~7G)
BSIZE=$(stat -c%s "${LOCAL_EXP}/best_model.pth" 2>/dev/null || echo 0)
if [ "${BSIZE}" -lt 1000000000 ]; then
  echo "  ⚠️ best_model.pth 只有 ${BSIZE} 字节, 可能不完整, 不清理远程!"
  OK=0
fi

if [ "${OK}" = "1" ]; then
  echo "[$(date '+%H:%M:%S')] ✓ 本地验证通过, 清理远程 A100 文件..."
  ssh ${REMOTE_HOST} "rm -rf ${REMOTE_EXP}" || true
  ssh ${REMOTE_HOST} "rm -f ${REMOTE_LOG}" || true
  echo "  ✓ 远程已清理"
else
  echo "[$(date '+%H:%M:%S')] ⚠️ 本地验证未通过, 保留远程文件以便重传!"
fi

echo "========================================"
echo "[$(date '+%H:%M:%S')] ✓ h12 监控+回传流程结束"
echo "  本地模型: ${LOCAL_EXP}/best_model.pth"
echo "========================================"
