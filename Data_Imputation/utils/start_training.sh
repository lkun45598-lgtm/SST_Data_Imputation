#!/bin/bash
#
# FNO-CBAM 训练启动脚本
# 使用8卡GPU进行分布式训练
#

set -e

echo "========================================"
echo "FNO-CBAM 海温数据重建训练"
echo "========================================"
echo ""

# 激活conda环境
echo "1. 激活pytorch环境..."
source /root/miniconda3/bin/activate pytorch

# 检查GPU
echo "2. 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# 检查数据文件
echo "3. 检查数据文件..."
if [ ! -f "/home/lz/processed_data/processed_sst_train.h5" ]; then
    echo "❌ 错误：训练数据不存在！"
    exit 1
fi
if [ ! -f "/home/lz/processed_data/processed_sst_valid.h5" ]; then
    echo "❌ 错误：验证数据不存在！"
    exit 1
fi
echo "✅ 数据文件检查通过"
echo ""

# 设置环境变量
echo "4. 设置环境变量..."
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4

# 进入工作目录
cd /root/data_for_agent_FNO_CBAM/FNO_CBAM

# 创建实验目录
mkdir -p /home/lz/data_for_agent_FNO_CBAM/FNO_CBAM/experiments/manual_sync_8gpu_modes80_64_bs32_ep40

# 启动训练
echo "5. 开始训练..."
echo "======================================"
echo ""

python train_sst_fno_cbam_manual_sync.py 2>&1 | tee /root/data_for_agent_FNO_CBAM/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================"
echo "训练完成或已中断"
echo "======================================"
