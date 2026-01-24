#!/bin/bash
# 优化后的OSTIA训练启动脚本

# 激活conda环境
source /home/lz/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# 设置环境变量解决MKL冲突
export MKL_SERVICE_FORCE_INTEL=1
export LD_PRELOAD=/home/lz/miniconda3/envs/pytorch/lib/libmkl_core.so:/home/lz/miniconda3/envs/pytorch/lib/libmkl_sequential.so

# 设置CUDA环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 训练配置信息
echo "=========================================="
echo "OSTIA优化训练启动"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "配置: lr=1e-3, epochs=60, AdamW+StepLR"
echo "GPU: 8×RTX 4090"
echo "=========================================="
echo ""

# 启动训练
python train_temporal_8gpu.py

