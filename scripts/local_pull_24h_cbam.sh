#!/bin/bash
# 【本地端脚本】后台轮询, 把 A100 训完的模型拉回本地
# 逻辑: 每 5 分钟检查 A100 的完成标记 (_cbam_done/hHH.done);
#   发现新完成的 -> rsync 拉回 -> 验证(>1G, cbam=true) -> 删远程目录(通知A100可训下一个)
# 本地断了重启即可继续; 不影响 A100 训练
set -u

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
DONEDIR="${REMOTE_DIR}/experiments/_cbam_done"
LOGDIR_LOCAL="${LOCAL_DIR}/experiments/hourly_data"

HOURS="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
INTERVAL=300

echo "[$(date '+%m-%d %H:%M')] 本地轮询拉取脚本启动"

while true; do
  all_done=1
  for h in ${HOURS}; do
    hh=$(printf "%02d" $((10#$h)))
    EXP_NAME="jaxa_finetune_h${hh}_realmask_clean"
    REMOTE_EXP="${REMOTE_DIR}/experiments/${EXP_NAME}"
    LOCAL_EXP="${LOCAL_DIR}/experiments/${EXP_NAME}"
    LOCAL_MODEL="${LOCAL_EXP}/best_model.pth"
    FLAG="${DONEDIR}/h${hh}.done"

    # 本地已成功拉取过 (有CBAM模型且>1G) -> 跳过
    if [ -f "${LOCAL_MODEL}" ]; then
      BSIZE=$(stat -c%s "${LOCAL_MODEL}" 2>/dev/null || echo 0)
      CBAM=$(python3 -c "import json;print(json.load(open('${LOCAL_EXP}/config.json')).get('use_cbam'))" 2>/dev/null || echo "?")
      if [ "${BSIZE}" -gt 1000000000 ] && [ "${CBAM}" = "True" ]; then
        continue
      fi
    fi

    all_done=0

    # 检查远程是否训完 (标记存在 + 模型文件存在)
    if ssh ${REMOTE_HOST} "test -f ${FLAG} && test -f ${REMOTE_EXP}/best_model.pth" 2>/dev/null; then
      echo "[$(date '+%m-%d %H:%M')] h${hh} 远程已完成, 开始拉取..."
      mkdir -p "${LOCAL_EXP}"
      rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/best_model.pth "${LOCAL_EXP}/"
      rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/config.json "${LOCAL_EXP}/" 2>/dev/null || true
      rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/training_history.json "${LOCAL_EXP}/" 2>/dev/null || true
      rsync -az ${REMOTE_HOST}:${REMOTE_EXP}/train.log "${LOGDIR_LOCAL}/h${hh}_cbam_train.log" 2>/dev/null || true

      # 验证
      BSIZE=$(stat -c%s "${LOCAL_MODEL}" 2>/dev/null || echo 0)
      CBAM=$(python3 -c "import json;print(json.load(open('${LOCAL_EXP}/config.json')).get('use_cbam'))" 2>/dev/null || echo "?")
      if [ "${BSIZE}" -gt 1000000000 ] && [ "${CBAM}" = "True" ]; then
        echo "[$(date '+%m-%d %H:%M')] ✓ h${hh} 拉取成功 ($(ls -lh ${LOCAL_MODEL}|awk '{print $5}'), cbam=${CBAM}), 删远程"
        ssh ${REMOTE_HOST} "rm -rf ${REMOTE_EXP}" || true
      else
        echo "[$(date '+%m-%d %H:%M')] ⚠️ h${hh} 拉取验证失败 (size=${BSIZE}, cbam=${CBAM}), 保留远程, 下轮重试"
      fi
    fi
  done

  if [ "${all_done}" = "1" ]; then
    echo "[$(date '+%m-%d %H:%M')] ===== 全部 23 个已拉回本地 ====="
    break
  fi
  sleep ${INTERVAL}
done

# 最终汇总
echo ""
echo "本地最终检查:"
for h in ${HOURS}; do
  hh=$(printf "%02d" $((10#$h)))
  m="${LOCAL_DIR}/experiments/jaxa_finetune_h${hh}_realmask_clean/best_model.pth"
  [ -f "$m" ] && echo "  h${hh}: ✓ $(ls -lh $m|awk '{print $5,$6,$7}')" || echo "  h${hh}: ✗"
done
