#!/bin/bash
# A100 训练监控 + 自动拉回脚本 (在本机运行, ssh 操作 A100)
# 作用:
#   1. 频繁删除 A100 上的 checkpoint_epoch_*.pth / final_model.pth (腾空间, 本机只要 best)
#   2. 某 hour 训完(master log 标记完成) -> 拉回 best_model.pth -> 校验大小 -> 删 A100 该 hour 目录
# A100 /home 只剩 ~33G, 单 checkpoint 7G, 必须及时清理

A100=lz@10.16.46.215
RREPO=/home/lz/SST_Data_Imputation/Data_Imputation
LREPO=/data1/user/lz/SST_Data_Imputation/Data_Imputation
REXP=$RREPO/experiments
LEXP=$LREPO/experiments
MASTERLOG=$REXP/hourly_data/_batch_clean_master.log
INTERVAL=170
MONLOG=$LREPO/experiments/hourly_data/_monitor_pull.log

echo "=== A100 监控启动 $(date) ===" | tee -a $MONLOG

while true; do
    # --- 1. 删除 A100 上所有 clean 目录的中间 checkpoint + final (只删 >2 分钟前的, 避免删正在写的) ---
    ssh $A100 "find $REXP/jaxa_finetune_*_realmask_clean -maxdepth 1 -name 'checkpoint_epoch_*.pth' -mmin +2 -delete 2>/dev/null; \
               find $REXP/jaxa_finetune_*_realmask_clean -maxdepth 1 -name 'final_model.pth' -mmin +2 -delete 2>/dev/null" 2>/dev/null

    # --- 2. 找 master log 标记为完成的 hour ---
    DONE=$(ssh $A100 "grep -oP 'h\K[0-9]{2}(?=: 完成)' $MASTERLOG 2>/dev/null" | sort -u)

    for hh in $DONE; do
        LDIR=$LEXP/jaxa_finetune_h${hh}_realmask_clean
        LBEST=$LDIR/best_model.pth
        RDIR=$REXP/jaxa_finetune_h${hh}_realmask_clean
        RBEST=$RDIR/best_model.pth

        # 本机已有该 hour 的 best 则跳过
        [ -f "$LBEST" ] && continue
        # A100 上该 hour 目录已被删(说明之前处理过) 则跳过
        ssh $A100 "[ -f $RBEST ]" 2>/dev/null || continue

        echo "[$(date '+%m-%d %H:%M')] h${hh}: 训练完成, 拉回 best_model.pth ..." | tee -a $MONLOG
        mkdir -p $LDIR
        rsync -a --partial $A100:$RBEST $LBEST 2>>$MONLOG

        # 校验大小一致
        RSIZE=$(ssh $A100 "stat -c%s $RBEST 2>/dev/null")
        LSIZE=$(stat -c%s $LBEST 2>/dev/null)
        if [ -n "$LSIZE" ] && [ "$RSIZE" = "$LSIZE" ]; then
            echo "[$(date '+%m-%d %H:%M')] h${hh}: 拉回成功 (${LSIZE} bytes), 删除 A100 目录" | tee -a $MONLOG
            ssh $A100 "rm -rf $RDIR" 2>/dev/null
        else
            echo "[$(date '+%m-%d %H:%M')] h${hh}: 大小不符 (R=$RSIZE L=$LSIZE), 保留下轮重试" | tee -a $MONLOG
            rm -f $LBEST   # 删掉不完整的, 下轮重拉
        fi
    done

    # --- 3. 全部 23 个 hour 都在本机就绪则退出 ---
    CNT=0
    for h in $(seq 0 23); do
        [ $h -eq 12 ] && continue
        hh=$(printf "%02d" $h)
        [ -f "$LEXP/jaxa_finetune_h${hh}_realmask_clean/best_model.pth" ] && CNT=$((CNT+1))
    done
    if [ $CNT -ge 23 ]; then
        echo "[$(date '+%m-%d %H:%M')] 全部 23 个 hour 已拉回本机, 监控退出" | tee -a $MONLOG
        break
    fi

    sleep $INTERVAL
done
