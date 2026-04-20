#!/bin/bash
# 批量预处理 h=2 到 h=23 的JAXA逐时数据
# 同时运行 PARALLEL 个小时的处理，每个用 WORKERS 个核心

PYTHON=/home/lz/miniconda3/envs/pytorch/bin/python
SCRIPT=/data1/user/lz/SST_Data_Imputation/Data_Imputation/preprocessing/hourly_preprocess.py
LOG_DIR=/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/hourly_data
PARALLEL=6    # 同时运行6个小时
WORKERS=32    # 每个进程32核

mkdir -p "$LOG_DIR"

# 待处理的小时列表
HOURS=($(seq 2 23))
TOTAL=${#HOURS[@]}
echo "=========================================="
echo "批量预处理: h=2 到 h=23 (共${TOTAL}个小时)"
echo "并行度: ${PARALLEL}, 每进程核心: ${WORKERS}"
echo "=========================================="

idx=0
while [ $idx -lt $TOTAL ]; do
    # 启动一批
    PIDS=()
    for ((j=0; j<PARALLEL && idx<TOTAL; j++, idx++)); do
        h=${HOURS[$idx]}
        hh=$(printf "%02d" $h)
        LOG_FILE="${LOG_DIR}/h${hh}_preprocess.log"
        echo "[$(date '+%H:%M:%S')] 启动 h=${hh} (series=all, workers=${WORKERS}) -> ${LOG_FILE}"
        $PYTHON $SCRIPT --hour $h --series all --workers $WORKERS > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    # 等待这一批完成
    echo "[$(date '+%H:%M:%S')] 等待 ${#PIDS[@]} 个进程完成..."
    for pid in "${PIDS[@]}"; do
        wait $pid
        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo "  警告: PID $pid 退出码 $STATUS"
        fi
    done
    echo "[$(date '+%H:%M:%S')] 本批完成"
    echo "------------------------------------------"
done

echo ""
echo "=========================================="
echo "全部预处理完成! $(date)"
echo "=========================================="

# 汇总结果
echo ""
echo "生成的数据目录:"
for h in $(seq 2 23); do
    hh=$(printf "%02d" $h)
    DIR="${LOG_DIR}/h${hh}"
    if [ -d "$DIR" ]; then
        COUNT=$(ls "$DIR"/*.h5 2>/dev/null | wc -l)
        echo "  h${hh}: ${COUNT} 个h5文件"
    else
        echo "  h${hh}: 未生成"
    fi
done
