# 计划：评估指标补充 + Proxy Construction消融

## 任务1：补充评估指标（优先，~1小时）

### 目标
在`baselines/eval_all_methods.py`中增加5项新指标，重新运行7方法评估。

### 新增指标

**逐帧计算（在line 239-252的循环中添加）：**
1. **Signed Bias / Mean Error**: `bias = float(e.mean())` — 判断整体偏暖/偏冷
2. **Centered RMSE**: `crmse = float(np.sqrt(((e - e.mean()) ** 2).mean()))` — 去均值偏差后的RMSE
3. **Gradient MAE**: `gradmae = float(np.abs(gdiff_all).mean()) if vv.sum() > 0 else np.nan` — 梯度幅值场的MAE（`gdiff_all = (pg - gtg)[vv]`已有，直接复用）

**汇总统计（在line 260-274的打印循环中添加）：**
4. **Median Absolute Error**: 对每个method+level组合，计算`np.median([r["mae"] for r in sub])`
5. **IQR (Interquartile Range)**: `np.percentile(..., 75) - np.percentile(..., 25)` for MAE, RMSE

### 实现步骤

1. **修改`eval_all_methods.py:247-252`的`rec`字典**，添加新字段：
   ```python
   rec = dict(
       method=name_m, level=name, target_ratio=ratio, idx=int(idx),
       mae=float(np.abs(e).mean()), 
       rmse=float(np.sqrt((e ** 2).mean())),
       bias=float(e.mean()),                              # NEW
       crmse=float(np.sqrt(((e - e.mean()) ** 2).mean())),  # NEW
       grad_ratio=grad_ratio, 
       hf_log_ratio=hf_log_ratio, 
       gradrmse=gradrmse,
       gradmae=float(np.abs(gdiff_all).mean()) if vv.sum() > 0 else np.nan,  # NEW
       gradcorr=gradcorr, 
       psd_curve=psd_curve, 
       n=int(ev.sum()),
   )
   ```

2. **修改打印部分（line 260-274）**，在表头和内容中加入median/IQR列：
   ```python
   print(f"{'method':<10}{'level':<8}{'MAE':>8}{'MAE_med':>8}{'MAE_IQR':>8}{'RMSE':>8}{'Bias':>8}{'cRMSE':>8}{'gradMAE':>10}{'gradRMSE':>10}{'gradCorr':>10}{'n':>8}")
   for name_m in methods:
       for lvl, _ in levels:
           sub = [r for r in records if r["method"] == name_m and r["level"] == lvl]
           if not sub:
               continue
           mae_vals = [r["mae"] for r in sub]
           mae = np.mean(mae_vals)
           mae_med = np.median(mae_vals)
           mae_iqr = np.percentile(mae_vals, 75) - np.percentile(mae_vals, 25)
           rmse = np.mean([r["rmse"] for r in sub])
           bias = np.mean([r["bias"] for r in sub])
           crmse = np.mean([r["crmse"] for r in sub])
           gradmae = np.nanmean([r["gradmae"] for r in sub])
           grmse = np.nanmean([r["gradrmse"] for r in sub])
           gcorr = np.nanmean([r["gradcorr"] for r in sub])
           print(f"{name_m:<10}{lvl:<8}{mae:>8.4f}{mae_med:>8.4f}{mae_iqr:>8.4f}{rmse:>8.4f}{bias:>8.4f}{crmse:>8.4f}{gradmae:>10.4f}{grmse:>10.4f}{gcorr:>10.3f}{len(sub):>8}")
   ```

3. **重新运行评估**：
   ```bash
   cd /data1/user/lz/SST_Data_Imputation/Data_Imputation/baselines
   python eval_all_methods.py --methods fno dincae unet convlstm dineof oi knn --out-name all_methods_stats_v7_newmetrics.npz
   ```
   预计耗时：~45-60分钟（7方法 × 3 levels × 20 samples = 420 frames推理）

4. **验证新指标合理性**：
   - Bias应接近0（模型无系统性偏差）
   - Centered RMSE应略小于RMSE（去掉均值偏差后）
   - Gradient MAE应与gradRMSE趋势一致（MAE < RMSE）
   - Median MAE vs Mean MAE差异显示分布是否被极值拉偏

---

## 任务2：Proxy Construction消融（较大，~2-3天）

### 目标
对比三个预处理变体对h12模型性能的影响：
- **Baseline（完整流程）**: temporal_fill → gaussian → 3D KNN
- **Variant A（去Gaussian）**: temporal_fill → 3D KNN
- **Variant B（去3D KNN）**: temporal_fill → gaussian

每个变体重新生成训练数据，重新训练h12模型，在相同评估协议下对比。

### 实现步骤

#### 2.1 修改预处理脚本支持跳过选项

**文件**: `preprocessing/hourly_preprocess.py`

**关键修改点**:
1. 在`process_series()`主函数（line 334起）中增加`skip_gaussian`和`skip_knn3d`参数：
   ```python
   def process_series(series_id, hour_offset, num_workers, skip_gaussian=False, skip_knn3d=False):
       """对一个年度序列的指定小时完整三阶段预处理"""
       ...
       # Stage 2
       if not skip_gaussian:
           sst_stage2, fill_masks_stage2 = stage2_gaussian_filter(...)
       else:
           print("\n  Stage 2: SKIPPED (gaussian filter disabled)")
           sst_stage2 = sst_stage1
           fill_masks_stage2 = fill_masks_stage1
       
       # Stage 3
       if not skip_knn3d:
           filled_data = stage3_knn_fill(...)
       else:
           print("\n  Stage 3: SKIPPED (3D KNN fill disabled)")
           filled_data = sst_stage2
   ```

2. 在`main()`的argparse中添加选项：
   ```python
   parser.add_argument('--skip-gaussian', action='store_true', help='跳过高斯滤波')
   parser.add_argument('--skip-knn3d', action='store_true', help='跳过3D KNN填充')
   ```

3. 修改输出路径逻辑，让不同变体输出到不同目录：
   ```python
   suffix = ""
   if skip_gaussian:
       suffix += "_nogauss"
   if skip_knn3d:
       suffix += "_noknn"
   output_dir = OUTPUT_BASE / f"h{hour_offset:02d}{suffix}"
   output_path = output_dir / f"jaxa_knn_filled_{series_id:02d}.h5"
   ```

**注意事项**:
- 如果跳过3D KNN，最终数据`filled_data`会保留一些NaN（3D KNN本该填的深层缺失区），模型训练时需能容忍NaN输入或在DataLoader中mask掉NaN像素（当前训练代码已支持`missing_mask`，无需额外改动）。
- 如果跳过Gaussian，时间填充像素保持原始噪声，可能增加训练难度。

#### 2.2 生成三个变体的训练数据

**目标hour**: h=12（与主线fig6/fig7一致）  
**目标series**: 0-7（训练集）+ 8（验证集）  
**预计耗时**: 
- 完整版：9个series × 25分钟 ≈ 3.75小时（已有，复用`experiments/hourly_data/h12/`）
- 去Gaussian：9 series × 23分钟 ≈ 3.5小时（省掉stage2 1-2分钟）
- 去3D KNN：9 series × 5分钟 ≈ 45分钟（只跑stage1+stage2，省掉stage3的20分钟）

**命令**:
```bash
cd /data1/user/lz/SST_Data_Imputation/Data_Imputation/preprocessing

# Variant A: 去Gaussian
python hourly_preprocess.py --hour 12 --series all --workers 64 --skip-gaussian
# 输出到 experiments/hourly_data/h12_nogauss/jaxa_knn_filled_*.h5

# Variant B: 去3D KNN
python hourly_preprocess.py --hour 12 --series all --workers 64 --skip-knn3d
# 输出到 experiments/hourly_data/h12_noknn/jaxa_knn_filled_*.h5
```

#### 2.3 重新训练三个模型

**训练脚本**: `training/train_jaxa_hourly.py`（推测，需要确认实际脚本名称和参数）

**关键参数**（从`docs/training.md`推断）:
- `--hour 12`
- `--data-dir experiments/hourly_data/h12{suffix}/`
- `--output-dir experiments/jaxa_finetune_h12{suffix}_realmask/`
- `--use-realmask` （使用RealCloudMaskGenerator，ratio 0.2-0.3）
- `--epochs 50` （或根据现有配置）
- `--gpus 0,1,2,3` （4卡DDP）

**预计耗时**: 每个模型约4小时（50 epoch × ~5分钟/epoch），三个模型串行需12小时（或并行到不同机器/时段）。

**命令**:
```bash
cd /data1/user/lz/SST_Data_Imputation/Data_Imputation/training

# Baseline (完整版，已有experiments/jaxa_finetune_h12_realmask/best_model.pth，跳过)

# Variant A: 去Gaussian
python train_jaxa_hourly.py --hour 12 \
    --data-dir ../experiments/hourly_data/h12_nogauss/ \
    --output-dir ../experiments/jaxa_finetune_h12_nogauss_realmask/ \
    --use-realmask --epochs 50 --gpus 0,1,2,3

# Variant B: 去3D KNN
python train_jaxa_hourly.py --hour 12 \
    --data-dir ../experiments/hourly_data/h12_noknn/ \
    --output-dir ../experiments/jaxa_finetune_h12_noknn_realmask/ \
    --use-realmask --epochs 50 --gpus 0,1,2,3
```

#### 2.4 评估对比

**评估脚本**: 复用`visualization/paper_figs/gen_fig7_stats.py`的逻辑，或编写新脚本`eval_proxy_ablation.py`。

**关键点**:
- 三个模型在**相同的evaluation data**（series 8，完整版h12数据）上推理
- 使用**相同的extra masks**（RealCloudMaskGenerator，ratio 0.2-0.3，固定seed）
- 记录每个变体的MAE/RMSE/Bias/gradMAE等新指标

**输出**:
- `cache/proxy_ablation_stats.npz` — 包含三个模型各自的records
- 对比表格：
  ```
  Variant          MAE     RMSE    Bias    gradMAE  gradRMSE  hf_log_ratio
  -----------------------------------------------------------------------
  Baseline         0.320   0.424   -0.01   0.045    0.058     2.29
  去Gaussian       0.XXX   0.XXX   X.XX    0.XXX    0.XXX     X.XX
  去3D KNN         0.XXX   0.XXX   X.XX    0.XXX    0.XXX     X.XX
  ```

#### 2.5 可视化对比

**目标图**: 
1. **Bar chart**: 三个变体的MAE/RMSE/gradMAE对比（类似fig8风格）
2. **Qualitative examples**: 保存3-5个case的重建结果（原始观测 | 完整版 | 去Gaussian | 去3D KNN），展示视觉差异

**实现**:
- `visualization/paper_figs/fig_proxy_ablation.py` — 生成bar chart
- `visualization/paper_figs/fig_proxy_cases.py` — 生成case对比图（2×5 layout，每行一个case）

---

## 工作量总结

| 任务 | 预计耗时 | GPU占用 | 依赖 |
|------|---------|---------|------|
| 1. 补充评估指标 | ~1小时 | 1卡 × 1小时 | 立即可做 |
| 2.1 修改预处理脚本 | ~30分钟 | 无 | 立即可做 |
| 2.2 生成Variant A数据 | ~3.5小时 | CPU密集（64核） | 依赖2.1 |
| 2.2 生成Variant B数据 | ~45分钟 | CPU密集（64核） | 依赖2.1 |
| 2.3 训练Variant A模型 | ~4小时 | 4卡 × 4小时 | 依赖2.2 |
| 2.3 训练Variant B模型 | ~4小时 | 4卡 × 4小时 | 依赖2.2 |
| 2.4 评估对比 | ~30分钟 | 1卡 × 30分钟 | 依赖2.3 |
| 2.5 可视化 | ~1小时 | 无 | 依赖2.4 |
| **总计** | **~15.5小时** | 串行关键路径：1h(任务1) + 3.5h(数据A) + 4h(训练A) + 4h(训练B) + 1h(评估+可视化) = 13.5h | |

**优化建议**:
- 任务1和2.1可并行
- 数据生成A/B可并行（分到不同机器或错峰）
- 训练A/B如果GPU资源充足可并行（需8卡或分时段）

---

## 风险和备选方案

1. **训练脚本路径/参数未确认**：
   - 需要检查`training/`目录下实际使用的脚本（可能是`train_jaxa_hourly.py`或其他名称）
   - 需要确认`--use-realmask`等参数是否存在

2. **去3D KNN后数据包含NaN**：
   - 如果模型训练时不能容忍NaN输入，需在DataLoader中显式mask或填充（用temporal_fill结果）
   - 备选方案：去3D KNN后用简单方法（如KNN 2D或线性插值）填充剩余NaN，保证数据完整性

3. **预处理耗时过长**：
   - 如果3.5小时生成数据过久，可先只生成series 8（验证集）测试流程（~25分钟），确认无误后再跑全部series

4. **评估发现某变体效果极差**：
   - 如果去Gaussian或去3D KNN导致MAE从0.32跳到0.5+，说明该步骤确实关键
   - 可额外测试"只保留Gaussian不保留3D KNN"等更细粒度组合（4种组合：完整/去G/去K/都去），但工作量翻倍

---

## 执行顺序建议

**Phase 1（立即开始）**:
1. 修改`eval_all_methods.py`增加新指标
2. 重新运行7方法评估（GPU 3，后台）
3. 同时修改`hourly_preprocess.py`支持跳过选项

**Phase 2（数据生成，CPU密集）**:
4. 启动Variant A数据生成（--skip-gaussian）
5. 启动Variant B数据生成（--skip-knn3d）— 可与4并行

**Phase 3（模型训练，GPU密集）**:
6. 训练Variant A模型（4卡，4小时）
7. 训练Variant B模型（4卡，4小时）— 如无8卡资源则串行

**Phase 4（评估+可视化）**:
8. 在series 8上评估三个模型
9. 生成对比图表

---

## 需要用户确认的点

1. **训练脚本路径和参数格式**：`training/train_jaxa_hourly.py`是否存在？参数名是否正确？
2. **是否需要先跑一个series测试**：在全量生成9个series前，是否先跑series 8验证流程？
3. **GPU资源分配**：任务1的评估(~1h)和任务2的训练(4h×2)是否可以错峰？还是需要等任务1完成再开始任务2？
4. **去3D KNN后的NaN处理**：是否需要在DataLoader中特殊处理，还是模型训练代码已支持部分NaN输入？
