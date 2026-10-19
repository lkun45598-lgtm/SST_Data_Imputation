#!/bin/bash
# 把"核心可跑包"从当前机器传到 10.16.46.215,目标全部放 /home/lz 下。
# 前提:已把本机公钥加到目标机 ~/.ssh/authorized_keys(目标机禁密码,只能密钥)。
# 用法:  bash transfer_to_215.sh
# 断点续传:中断后重跑即可(--partial)。
set -u
DST=lz@10.16.46.215
SRC_REPO=/data1/user/lz/SST_Data_Imputation/Data_Imputation
DST_REPO=/home/lz/SST_Data_Imputation/Data_Imputation
SRC_OSTIA=/data/sst_data/sst_missing_value_imputation/processed_data
DST_OSTIA=/home/lz/sst_data/sst_missing_value_imputation/processed_data
SSH_CMD="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -o ConnectTimeout=15"

echo "==== 0/4 目标机建目录 ===="
ssh -o BatchMode=yes $DST "mkdir -p $DST_REPO/experiments/hourly_data $DST_REPO/experiments/ostia_pretrain $DST_OSTIA" || {
  echo '!! ssh 失败:确认公钥已加到目标机 ~/.ssh/authorized_keys'; exit 1; }

echo "==== 1/4 代码 (~49M;排除根级/ablation experiments 大模型/缓存/.git;保留 baselines/experiments) ===="
rsync -ah --info=progress2 --partial -e "$SSH_CMD" \
    --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='/experiments/' --exclude='/ablation/experiments/' --exclude='SST_Paper/' --exclude='.codex/' \
    --exclude='visualization/paper_figs/cache/' \
    "$SRC_REPO/" "$DST:$DST_REPO/"

echo "==== 2/4 JAXA 训练数据 hourly_data (~25G) ===="
rsync -ah --info=progress2 --partial -e "$SSH_CMD" \
    "$SRC_REPO/experiments/hourly_data/" "$DST:$DST_REPO/experiments/hourly_data/"

echo "==== 3/4 FNO OSTIA 基座模型 best_model.pth (~7G) ===="
rsync -ah --info=progress2 --partial -e "$SSH_CMD" \
    "$SRC_REPO/experiments/ostia_pretrain/best_model.pth" "$DST:$DST_REPO/experiments/ostia_pretrain/"

echo "==== 4/4 OSTIA 原始数据 (~5.9G;只做基线/FNO 预训练才需要) ===="
rsync -ah --info=progress2 --partial -e "$SSH_CMD" \
    "$SRC_OSTIA/" "$DST:$DST_OSTIA/"

echo ""
echo "==== 完成核对 ===="
ssh -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=6 $DST "echo repo:; du -sh $DST_REPO 2>/dev/null; echo hourly_data:; du -sh $DST_REPO/experiments/hourly_data 2>/dev/null; echo ostia_base:; ls -lah $DST_REPO/experiments/ostia_pretrain/best_model.pth 2>/dev/null; echo ostia_data:; du -sh $DST_OSTIA 2>/dev/null; echo GPUs:; nvidia-smi --query-gpu=index,memory.total --format=csv,noheader 2>/dev/null"
echo ""
echo ">> 目标机上跑 OSTIA 相关(基线预训练/FNO 预训练)前,先 export OSTIA_DIR=$DST_OSTIA"
