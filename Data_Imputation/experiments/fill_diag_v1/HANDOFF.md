# 交接文档 (2026-10-16)

新对话先读本文件，再读 IMPLEMENTATION_PLAN.md (已定决策) 和 FINDINGS.md (实验依据)。
所有路径相对 `/data1/user/lz/SST_Data_Imputation/Data_Imputation/`。
环境: 训练必须用 `/home/lz/miniconda3/envs/pytorch/bin/python` (torch 2.5.1+cu121)；GPU 共享，启动前先 `nvidia-smi`。

## 一、已完成
1. **底图方案定案**: tw (过去 48h，1/lag 加权，逐帧偏移对齐) + 去云边 2px 的 bc + tanh 软限幅 1K。依据 FINDINGS.md 第五至十一节。
   - 真实帧下一小时参照 (116 帧): erode0 的 tw+bc 比 tw 更差 (偏冷 0.4K)；erode1/2 最好 (0.78~0.80)。
   - 人工洞评估高估 bc (边缘无云污染)，应以真实帧验证为准。
2. **全库真实帧预计算**: `preprocessing/build_filled_store.py` → `experiments/filled_store_v1/filled_hourly.h5` (24GB)
   - 数据集: `sst_filled` int16 (0.01K，无效 -32768)、`hist_count` uint8 (过去48h观测次数 C)、`valid`、`bg_only`。
   - 83444 个有文件槽中 80926 个完整填充；`fill_background_only.py` 又补了 2441 个"仅背景"槽 (当前观测<100 px，无 bc)，`bg_only=1`。仍有 77 个槽无输出 (库开头无历史)。1876 个槽原始数据就没有文件。
   - **只能用于无人工挖洞的场景 (推理、真实云缺口评估)**；人工洞必须挖洞后重算。训练取样应排除 `bg_only=1`。
3. **人工洞样本函数**: `datasets/artificial_holes.py` (`build_bank_from_store`、`make_sample`)
   - 目标帧用 `inference/real_cloud_mask.py` 的 `RealCloudMaskGenerator` 挖洞，再叠一个源云事件的过去 48h 真实缺失序列到历史帧，挖洞后重算底图和 C；窗口内每帧都重填。
   - 单个完整 30 帧窗口样本约 10.6s；云库只取训练期 (≤2023-07-04)。
4. **有效块统计** (`experiments/fill_diag_v1/hole_stats.py`，450 样本，目标帧覆盖率≥0.7，42px 块须落在"洞内、有真值、离真实云边≥3px"):
   | 挖洞比例 | C=0占洞 | 块数≥100的样本 | 块数为0 |
   |---|---|---|---|
   | 0.25 | 8.8% | 45% | 39% |
   | 0.50 | 10.3% | 80% | 12% |
   | 0.60 | 11.9% | 85% | 7% |
   对抗损失只能用大洞 (≥0.5) 且有效块≥100 的样本；C=0 占比仍低于真实缺失区的 22%。

## 二、已定决策 (详见 IMPLEMENTATION_PLAN.md)
- **路线 A**: 模型仍输出绝对 SST；底图 B 与 C 作为输入通道；不做 `B+网络输出` 的残差化。论文表述需改: 底图是先验输入，不写"FNO 预测残差"。
- 窗口 **30 小时** (30 SST + 30 mask)，与 OSTIA 预训练权重 (SST/mask 编码器各 30 通道) 形状一致，无需重新预训练；C 通道另加编码器，权重零初始化。
- 填充用 48h 历史，窗口内每帧都填；训练挖洞改为**时间一致**。
- 小时编码: 每个 FNO 块 LayerNorm 后加 FiLM，零初始化，仅微调阶段。
- GAN 配对设计 (待用户对"不再单独抽低缺失率真样本"做明确确认): 真残差=真值−B，假残差=预测−B，同一个洞同一个 B；D 输入 `[R·M_dil, M]` (洞向外膨胀 3–5px)，残差归一化；谱归一化 PatchGAN n_down=3 (感受野 42px≈90km)，in_ch=2，LSGAN；λ 终值 0.01，调度 (50 epoch) 0–10 为 0、10–30 线性升、30–50 固定；推理不用 D。
- 先训**不带 GAN 的重建基线**，再加 GAN 做开/关对比。
- 大洞课程: 在覆盖率≥0.7 的帧 (训练期约 8009 帧) 上增加大洞 (挖 40–60%)，与常规小洞 (20–30%) 混合；大洞样本同时是 GAN 样本。

## 三、后续待办 (按顺序)
1. **读现有数据集输入格式** (`datasets/jaxa_dataset.py`、`training/train_jaxa_hourly.py` 的 batch 键: input_sst_seq / mask_seq / ground_truth_sst / loss_mask / original_obs_mask / land_mask)，写**单模型数据集**: 30h 窗口 (30 SST + 30 mask) + C + 目标小时 sin/cos；底图填充已在窗口内每帧完成；输入 SST 通道用填充后的值还是原始观测需要定 (现有输入是填充后值+掩码)。
2. 样本供给方式: 预生成固定样本 (5000 条约 48GB、约 13 分钟 64 进程，但洞固定) 还是在线生成 (每样本约 10s，需大量 worker)；需测 DataLoader 吞吐后决定，未定。
3. 模型改造: 新输入通道编码器 + FiLM 小时编码，写成包装类/子类，不改原网络文件。
4. 预训练迁移: 先核实 OSTIA 的实际时间分辨率；加载预训练权重；做 OSTIA 预训练 vs 随机初始化消融。
5. 重建基线训练 (λ=0)，在 50 epoch 内；评估口径见第四节。
6. GAN: 新写训练脚本和数据集 (不动旧的 `train_jaxa_hourly_gan.py`、`datasets/lowmiss_real_frames.py`)，`models/patchgan_discriminator.py` 的 `PatchDiscriminator(in_ch=2, n_down=3)` 可直接用；对抗项仅对有效块数≥100 的大洞样本计算，无有效块则跳过。监控 D 准确率，长期>95% 或≈50% 报警。
7. 消融矩阵: OSTIA 预训练 vs 随机初始化；tw vs tw+bc；有无 C 通道；GAN 开/关。
8. 自然缺口的弱约束 (低通锚定 + 接缝连续，参见 `docs/superpowers/specs/2026-08-23-natural-gap-supervision-design.md`；该设计是否已实现、结果如何**未核实**) 作为基线之后的单独消融。
9. 评估与交付物 (原始需求): 低缺失率帧人工挖洞评估 (缺失率阈值沿用≥0.7，未经用户明确确认；评估帧与 GAN 帧时间上不重叠，只在洞内且原有观测像素统计误差)；真实云洞用 `eval_realgap_noref.py` 的无参考指标；指标 MAE/RMSE、梯度相关、高频谱 (`baselines/spectral_metrics.py`)、交界处梯度不连续度；对比基线: 当前 24 模型、linear/cubic、DINEOF、OI；单模型结果按小时 h00–h23 分组汇报。交付: 缺失区域演化 GIF、趋势与细节折线图、接缝对比表、指标汇总。

## 四、需要用户确认/未解决的问题
- 配对 GAN 设计 (不再单独抽低缺失率真样本帧) 用户未明确确认。
- 评估缺失率阈值、微调总 epoch (默认 50) 未明确确认。
- 输入 SST 通道用原始观测还是填充值；窗口内非目标帧是否也用填充值 (已定"每帧都填"，但需实现时确认格式)。
- 底图 C=0 区边界梯度突变未消除 (部分可能是物理云边)；设想由 C 通道交给模型，**未验证**。
- 原流水线 (旧 hourly_data 的 time+gauss+3D KNN) 输出存在 NaN 和个别帧误差>10K，用户已决定弃用旧数据，未追查原因。
- 辅助变量: 此前调研只有 wind_speed 不受云影响，用户决定暂不使用。
- 训练里的 C=0 占洞比例 (9–12%) 低于真实 (22%)，推理是否因此受影响未验证。
- OSTIA GT 比 JAXA 平滑得多 (梯度小 3 倍、拉普拉斯小 36 倍)，预训练分布与微调不同。
- 当前整体基线对比、GAN、单模型**都没有训练过**，所有训练相关数字目前不存在。
