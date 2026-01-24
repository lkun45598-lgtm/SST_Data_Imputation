#!/bin/bash
# GPU使用情况实时监控

echo "GPU实时监控 - 按Ctrl+C退出"
echo ""

watch -n 2 "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{printf \"GPU %s: %s | 温度:%s°C | GPU利用率:%s%% | 显存利用率:%s%% | 显存:%s/%sMB\\n\", \$1, \$2, \$3, \$4, \$5, \$6, \$7}'"
