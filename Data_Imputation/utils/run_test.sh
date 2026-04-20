#!/bin/bash
# 30天时间序列FNO_CBAM小规模测试（2-3 epochs）
# 用于快速验证训练流程

set -e

echo "=========================================="
echo "30天时间序列 - 小规模训练测试"
echo "=========================================="
echo ""

# 激活conda环境
echo "激活pytorch环境..."
source ~/miniconda3/bin/activate pytorch

cd /home/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM

# 创建测试配置
echo "创建测试配置（3 epochs）..."

# 备份原始训练脚本
if [ ! -f "train_temporal_8gpu_full.py.bak" ]; then
    cp train_temporal_8gpu.py train_temporal_8gpu_full.py.bak
fi

# 修改epochs为3（用于快速测试）
sed -i 's/num_epochs = 40/num_epochs = 3/' train_temporal_8gpu.py

echo "✓ 测试配置完成（3 epochs）"
echo ""

# 创建日志目录
test_log_dir="test_logs_30days"
mkdir -p $test_log_dir
timestamp=$(date +%Y%m%d_%H%M%S)

echo "开始测试训练..."
echo "预计时间: 30-60分钟"
echo "日志: ${test_log_dir}/test_${timestamp}.log"
echo ""

python train_temporal_8gpu.py 2>&1 | tee ${test_log_dir}/test_${timestamp}.log

# 恢复原始配置
echo ""
echo "恢复原始配置..."
sed -i 's/num_epochs = 3/num_epochs = 40/' train_temporal_8gpu.py

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "如果测试通过，可以运行完整训练:"
echo "  bash run_train.sh"
