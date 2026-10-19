#!/bin/bash
# 在远程 A100 用 8 卡重新训练正常的 h12 模型 (CBAM + 预训练 + 完整loss)
# 训练完成后自动 rsync 权重/日志/config 回本地
set -u

REMOTE_HOST="lz@10.16.46.215"
REMOTE_DIR="/home/lz/SST_Data_Imputation/Data_Imputation"
LOCAL_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation"
PY="/home/lz/miniconda3/envs/pytorch/bin/python"

EXP_NAME="jaxa_finetune_h12_realmask_clean"   # 正常模型目录名 (suffix=_realmask)
REMOTE_EXP="${REMOTE_DIR}/experiments/${EXP_NAME}"
REMOTE_LOG="${REMOTE_DIR}/experiments/hourly_data/jaxa_finetune_h12_retrain.log"
LOCAL_EXP="${LOCAL_DIR}/experiments/${EXP_NAME}"

echo "========================================"
echo "[$(date '+%H:%M:%S')] 启动远程 h12 训练 (8卡 A100)"
echo "  配置: CBAM=ON, 预训练=ON, grad=0.2, boundary=0.1, temporal=0.0"
echo "  远程目录: ${REMOTE_EXP}"
echo "  本地回传: ${LOCAL_EXP}"
echo "========================================"

# 1. 启动远程训练 (前台阻塞, 等训练结束)
ssh ${REMOTE_HOST} "cd ${REMOTE_DIR}/training && MASTER_PORT=29910 ${PY} -u train_jaxa_hourly.py \
  --hour 12 \
  --mask-mode realcloud \
  --epochs 60 \
  --batch-size 2 \
  --gpus 8 \
  --use-cbam \
  --alpha-grad 0.2 \
  --alpha-boundary 0.1 \
  --alpha-temporal 0.0 \
  > ${REMOTE_LOG} 2>&1"

TRAIN_RC=$?
echo "[$(date '+%H:%M:%S')] 远程训练结束 (退出码: ${TRAIN_RC})"

# 2. 验证远程 best_model.pth 存在
if ! ssh ${REMOTE_HOST} "test -f ${REMOTE_EXP}/best_model.pth"; then
  echo "❌ 远程未生成 best_model.pth, 传输中止! 请检查日志: ${REMOTE_LOG}"
  # 仍然传回日志方便排查
  rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_DIR}/experiments/hourly_data/ || true
  exit 1
fi

# 3. rsync 整个实验目录回本地 (覆盖本地被误存的 fno_only)
echo "[$(date '+%H:%M:%S')] 回传实验目录..."
mkdir -p ${LOCAL_EXP}
rsync -avz --progress ${REMOTE_HOST}:${REMOTE_EXP}/ ${LOCAL_EXP}/

# 4. 回传训练日志
echo "[$(date '+%H:%M:%S')] 回传训练日志..."
rsync -az ${REMOTE_HOST}:${REMOTE_LOG} ${LOCAL_DIR}/experiments/hourly_data/

# 5. 验证本地文件完整性
echo "[$(date '+%H:%M:%S')] 验证本地文件..."
for f in best_model.pth config.json training_history.json; do
  if [ -f "${LOCAL_EXP}/${f}" ]; then
    echo "  ✓ ${f} ($(ls -lh ${LOCAL_EXP}/${f} | awk '{print $5}'))"
  else
    echo "  ✗ ${f} 缺失!"
  fi
done

# 6. 确认 config 是正常模型 (use_cbam=true, no_pretrain 缺省/false)
echo "[$(date '+%H:%M:%S')] 确认模型配置..."
${PY} -c "import json; c=json.load(open('${LOCAL_EXP}/config.json')); print('  use_cbam:', c.get('use_cbam'), '| no_pretrain:', c.get('no_pretrain', False))" 2>/dev/null || true

echo "========================================"
echo "[$(date '+%H:%M:%S')] ✓ h12 重训练 + 回传完成!"
echo "  注意: 远程文件保留 (未清理), 如需清理请手动执行"
echo "========================================"
