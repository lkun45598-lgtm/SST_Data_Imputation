#!/bin/bash
# 验证所有消融实验文件是否完整传回

echo "=========================================="
echo "消融实验文件完整性检查"
echo "=========================================="
echo ""

ABLATION_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation/ablation/experiments"
OSTIA_DIR="/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/jaxa_finetune_h12_nopretrain_realmask_clean"

echo "【CBAM 消融】"
for exp in fno_only cbam_basic cbam_boundary cbam_temporal; do
  echo "----------------------------"
  echo "实验: ${exp}"
  
  if [ -f "${ABLATION_DIR}/${exp}/best_model.pth" ]; then
    size=$(ls -lh ${ABLATION_DIR}/${exp}/best_model.pth | awk '{print $5}')
    echo "  ✓ best_model.pth (${size})"
  else
    echo "  ✗ best_model.pth 缺失"
  fi
  
  if [ -f "${ABLATION_DIR}/${exp}/config.json" ]; then
    echo "  ✓ config.json"
  else
    echo "  ✗ config.json 缺失"
  fi
  
  if [ -f "${ABLATION_DIR}/${exp}/training_history.json" ]; then
    echo "  ✓ training_history.json"
  else
    echo "  ✗ training_history.json 缺失"
  fi
  
  log_files=$(ls ${ABLATION_DIR}/${exp}/*train.log 2>/dev/null | wc -l)
  if [ $log_files -gt 0 ]; then
    echo "  ✓ 训练日志 (${log_files} 个)"
  else
    echo "  ✗ 训练日志缺失"
  fi
done

echo ""
echo "【OSTIA 消融】"
echo "----------------------------"
echo "实验: h12_nopretrain"

if [ -f "${OSTIA_DIR}/best_model.pth" ]; then
  size=$(ls -lh ${OSTIA_DIR}/best_model.pth | awk '{print $5}')
  echo "  ✓ best_model.pth (${size})"
else
  echo "  ✗ best_model.pth 缺失"
fi

if [ -f "${OSTIA_DIR}/config.json" ]; then
  echo "  ✓ config.json"
else
  echo "  ✗ config.json 缺失"
fi

if [ -f "${OSTIA_DIR}/training_history.json" ]; then
  echo "  ✓ training_history.json"
else
  echo "  ✗ training_history.json 缺失"
fi

echo ""
echo "=========================================="
