#!/usr/bin/env bash
set -euo pipefail
cd /home/lz/SST_Data_Imputation
BASE=Data_Imputation/experiments/gradient_prior_v2_natural_gap
EXP=h12_proxy_smooth_v1_20260904_1740
CKPT=$BASE/$EXP/best_model.pth
DATA=Data_Imputation/experiments/hourly_data_proxy_smooth_v1_20260904_full/h12
STATS=$BASE/stats/h12_train_monthly_v3.json
OUT=$BASE/inference_proxy_smooth_v1_20260906
SESSION=sst_proxy_smooth_infer_20260906

test -f "$CKPT"
test -d "$DATA"
test ! -e "$OUT"
mkdir -p "$OUT"

/home/lz/miniconda3/envs/pytorch/bin/python -m Data_Imputation.retraining.gradient_prior_v2_natural_gap.evaluate \
  --checkpoint "$CKPT" --data-dir "$DATA" --gradient-stats "$STATS" \
  --output "$OUT/series7_evaluation.json" --device cuda:0 --batch-size 1 --num-workers 2

for frame in 50 100 150 179 223; do
  /home/lz/miniconda3/envs/pytorch/bin/python -m Data_Imputation.retraining.gradient_prior_v2_natural_gap.visualize_realcloud_style \
    --checkpoint "$CKPT" --data-path "$DATA/jaxa_knn_filled_08.h5" --stats-path "$STATS" \
    --output "$OUT/series8_frame${frame}_realcloud_gradient_style.png" --frame "$frame" --device cuda:0
done

date -u > "$OUT/INFERENCE_COMPLETE.txt"
