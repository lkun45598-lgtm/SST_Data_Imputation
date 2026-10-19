# 因果逐时序列时空 KNN v3

该目录是独立的 JAXA 逐时预处理实现，不修改旧的 `hourly_preprocess.py`、旧 H5 或模型 checkpoint。

处理顺序为：

1. 目标时刻之前最多 72 小时内，按真实 `timestamps` 查找同一网格位置的原始观测，并使用 `1 / delta_hours` 加权填充。
2. 使用旧流程的 `sigma=1.5` Gaussian 操作，只写回时序填充像素；原始观测和仍缺失像素不被修改。
3. 对剩余海洋缺口执行带真实时间衰减的空间 KNN。历史候选只来自 `original_obs_mask==1`；当前帧新生成的值只在当前帧内渐进使用。

所有源值都通过 `original_obs_mask` 和有限 SST 检查。源文件间少量变化的 `land_mask` 使用保守并集，陆地像素始终保持 NaN。若海洋像素最终无法解析，程序报错并拒绝写入 H5。

## Smoke run

```bash
python -m Data_Imputation.preprocessing.temporal_hourly_v3.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root Data_Imputation/experiments/hourly_data_v3_temporal72 \
  --series 8 --hour 12 --frame 223 --neighbor-frames 5 --workers 196
```

该命令处理目标帧前后邻帧，并在 `diagnostics/frame223/summary.json` 记录实际历史时间戳、缺失小时和掩码覆盖率。

## Full regeneration

完整序列必须显式添加 `--full`，并使用一个新的空输出目录：

```bash
python -m Data_Imputation.preprocessing.temporal_hourly_v3.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root Data_Imputation/experiments/hourly_data_v3_temporal72_full_s08_h12 \
  --series 8 --hour 12 --full --workers 196
```

模型重新训练不属于本预处理流程；应在 smoke 结果审查通过后另行执行。
