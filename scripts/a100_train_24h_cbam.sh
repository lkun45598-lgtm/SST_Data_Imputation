#!/bin/bash
# 【A100 端脚本】在 A100 本地用 tmux 常驻运行, 串行训练 23 个有CBAM模型
# 训完一个 -> 清理中间checkpoint -> 写完成标记 -> 等本地拉走(目录被删)才训下一个
# 这样 A100 上最多只占 1 个模型的空间 (避免爆盘, /home 只剩 32G)
# 不依赖任何反向连接, A100 自己就能跑, 绝对抗断连
set -u

REPO=/home/lz/SST_Data_Imputation/Data_Imputation
PY=/home/lz/miniconda3/envs/pytorch/bin/python
SCRIPT=$REPO/training/train_jaxa_hourly.py
LOGDIR=$REPO/experiments/hourly_data
DONEDIR=$REPO/experiments/_cbam_done   # 完成标记目录
mkdir -p $DONEDIR

HOURS="00 01 02 03 04 05 06 07 08 09 10 11 13 14 15 16 17 18 19 20 21 22 23"
EPOCHS=60; BATCH=2; NGPU=8

echo "[$(date '+%m-%d %H:%M')] A100 批量训练启动 (23个有CBAM模型)"

for h in ${HOURS}; do
  hh=$(printf "%02d" $((10#$h)))
  EXP=$REPO/experiments/jaxa_finetune_h${hh}_realmask_clean
  LOG=$LOGDIR/h${hh}_cbam_train.log
  FLAG=$DONEDIR/h${hh}.done

  # 若本地已确认拉走 (标记存在且目录已被删), 跳过
  if [ -f "$FLAG" ] && [ ! -d "$EXP" ]; then
    echo "[$(date '+%H:%M')] h${hh} 已完成且已拉走, 跳过"
    continue
  fi

  echo "[$(date '+%m-%d %H:%M')] ===== h${hh} 训练开始 ====="
  cd $REPO/training
  MASTER_PORT=29520 $PY -u $SCRIPT \
    --hour ${hh} --mask-mode realcloud --epochs ${EPOCHS} \
    --batch-size ${BATCH} --gpus ${NGPU} \
    --use-cbam --alpha-grad 0.2 --alpha-boundary 0.1 --alpha-temporal 0.0 \
    > ${LOG} 2>&1

  if [ ! -f "${EXP}/best_model.pth" ]; then
    echo "[$(date '+%H:%M')] ❌ h${hh} 未生成模型, 跳过"
    continue
  fi

  # 清理中间 checkpoint, 只留 best/final/config/history
  rm -f ${EXP}/checkpoint_epoch_*.pth
  # 复制日志进目录, 方便本地一并拉取
  cp ${LOG} ${EXP}/train.log 2>/dev/null || true
  # 写完成标记 (本地轮询靠它判断可以拉取)
  touch "$FLAG"
  echo "[$(date '+%m-%d %H:%M')] ✓ h${hh} 训练完成, 等本地拉取..."

  # 等本地拉走: 本地拉完会删掉 EXP 目录; 这里最多等 6 小时, 超时也继续(防卡死)
  waited=0
  while [ -d "$EXP" ] && [ $waited -lt 21600 ]; do
    sleep 120; waited=$((waited+120))
  done
  if [ -d "$EXP" ]; then
    echo "[$(date '+%H:%M')] ⚠️ h${hh} 等待本地拉取超时(6h), 强制继续下一个 (磁盘可能吃紧)"
  else
    echo "[$(date '+%H:%M')] h${hh} 已被本地拉走, 继续下一个"
  fi
done

echo "[$(date '+%m-%d %H:%M')] ===== 全部 23 个训练完成 ====="
