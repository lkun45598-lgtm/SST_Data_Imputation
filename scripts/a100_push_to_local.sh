#!/bin/bash
# 【A100 端推送脚本】在 A100 上独立 tmux 常驻运行
# 职责: 检查训练脚本写的完成标记(.done), 把训完的模型 rsync 推送到本地4090,
#       验证成功后删除 A100 上该模型目录(训练脚本据此继续下一个)
# 与训练脚本解耦; A100 主动推送, 不依赖本地轮询, 不依赖本地SSH稳定
set -u

LOCAL_HOST="lz@172.18.28.63"       # 本地 4090
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"    # A100 本地路径
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"  # 4090 上的路径
DONEDIR="${REMOTE_DIR}/experiments/_cbam_done"
LOGDIR_REMOTE="${REMOTE_DIR}/experiments/hourly_data"

HOURS="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
INTERVAL=180
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"

echo "[$(date '+%m-%d %H:%M')] A100端推送脚本启动 (推送到 ${LOCAL_HOST})"

while true; do
  all_done=1
  for h in ${HOURS}; do
    hh=$(printf "%02d" $((10#$h)))
    EXP_NAME="jaxa_finetune_h${hh}_realmask_clean"
    REMOTE_EXP="${REMOTE_DIR}/experiments/${EXP_NAME}"
    LOCAL_EXP="${LOCAL_DIR}/experiments/${EXP_NAME}"
    FLAG="${DONEDIR}/h${hh}.done"
    PUSHED="${DONEDIR}/h${hh}.pushed"   # 本脚本自己的"已推送"标记

    # 已推送过 -> 跳过
    [ -f "${PUSHED}" ] && continue

    # 该模型尚未训完 -> 标记未全部完成, 跳过
    if [ ! -f "${FLAG}" ] || [ ! -f "${REMOTE_EXP}/best_model.pth" ]; then
      all_done=0
      continue
    fi

    # 训完且未推送 -> 推送到本地
    echo "[$(date '+%m-%d %H:%M')] h${hh} 训完, 推送到本地..."
    ${SSH} ${LOCAL_HOST} "mkdir -p ${LOCAL_EXP}" 2>/dev/null
    rsync -az -e "${SSH}" ${REMOTE_EXP}/best_model.pth ${LOCAL_HOST}:${LOCAL_EXP}/
    rc1=$?
    rsync -az -e "${SSH}" ${REMOTE_EXP}/config.json ${LOCAL_HOST}:${LOCAL_EXP}/ 2>/dev/null
    rsync -az -e "${SSH}" ${REMOTE_EXP}/training_history.json ${LOCAL_HOST}:${LOCAL_EXP}/ 2>/dev/null
    rsync -az -e "${SSH}" ${REMOTE_EXP}/train.log ${LOCAL_HOST}:${LOCAL_DIR}/experiments/hourly_data/h${hh}_cbam_train.log 2>/dev/null

    # 验证本地文件大小 (远程询问本地)
    if [ $rc1 -eq 0 ]; then
      LSIZE=$(${SSH} ${LOCAL_HOST} "stat -c%s ${LOCAL_EXP}/best_model.pth 2>/dev/null" || echo 0)
      if [ "${LSIZE}" -gt 1000000000 ]; then
        echo "[$(date '+%m-%d %H:%M')] ✓ h${hh} 推送成功 (本地 ${LSIZE} 字节), 删A100目录"
        touch "${PUSHED}"
        rm -rf "${REMOTE_EXP}"     # 删A100目录 -> 训练脚本据此继续下一个
      else
        echo "[$(date '+%m-%d %H:%M')] ⚠️ h${hh} 本地大小异常(${LSIZE}), 保留重传"
        all_done=0
      fi
    else
      echo "[$(date '+%m-%d %H:%M')] ⚠️ h${hh} rsync失败(rc=$rc1), 下轮重试"
      all_done=0
    fi
  done

  if [ "${all_done}" = "1" ]; then
    echo "[$(date '+%m-%d %H:%M')] ===== 全部24个已推送到本地 ====="
    break
  fi
  sleep ${INTERVAL}
done
