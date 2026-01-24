#!/bin/bash
# 30天时间序列FNO_CBAM训练启动脚本
# 使用conda pytorch环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "30天时间序列FNO_CBAM训练"
echo "=========================================="
echo ""

# 激活conda环境
echo "激活pytorch环境..."
source ~/miniconda3/bin/activate pytorch

# 检查环境
echo ""
echo "环境检查:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  GPU数量: {torch.cuda.device_count()}')"
echo ""

# 进入工作目录
cd /home/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM

# 检查数据
echo "检查训练数据..."
if [ ! -f "/home/lz/processed_data/processed_sst_train_bilinear.h5" ]; then
    echo "✗ 训练数据不存在！"
    exit 1
fi
echo "✓ 训练数据存在"
echo ""

# 创建日志目录
log_dir="training_logs_30days"
mkdir -p $log_dir
timestamp=$(date +%Y%m%d_%H%M%S)

# 启动训练
echo "开始训练..."
echo "日志文件: ${log_dir}/train_${timestamp}.log"
echo ""

python train_temporal_8gpu.py 2>&1 | tee ${log_dir}/train_${timestamp}.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
