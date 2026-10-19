# 实现清单 (路线 A: 输出绝对 SST，底图作输入，GAN 判残差)

2026-10-15 讨论定稿。前提与依据见同目录 FINDINGS.md。

## 一、已确定
1. 模型仍输出**绝对 SST**。底图 B 与 C 作为**输入通道**，不做 `B + 网络输出` 的残差化。
   论文表述需改为: 底图是先验输入，不写成"FNO 预测残差"。
2. 底图: tw (过去 48 h，1/lag 加权，逐帧偏移对齐) + 去云边 2 px 的 bc + tanh 软限幅 1 K。
   - 填充用 48 h 历史，模型输入窗口 30 h；窗口内**每一帧都填**，C 通道也按挖洞后的掩码算。
   - 真实帧预处理: `preprocessing/build_filled_store.py` → `experiments/filled_store_v1/filled_hourly.h5`
     (sst_filled int16, hist_count uint8, valid)。
3. 训练/评估的人工洞: 挖洞作用于整个 48 h 历史 (时间一致)，**挖洞之后重新算底图和 C**，
   不能直接用上面预计算的真实帧结果 (否则真值经历史帧泄漏)。
4. GAN (配对设计): 真/假残差来自同一个洞、同一个 B。
   - 真残差 = 真值 − B；假残差 = 预测 − B；残差按训练集标准差归一化。
   - D 输入 `[R·M_dil, M]`，M_dil 为洞向外膨胀 3–5 像元 (保留接缝参照)；第一版不送 B。
   - D: 谱归一化 PatchGAN，n_down=3 (感受野 42 px ≈ 90 km)，in_ch=2；LSGAN。
   - D 只评价感受野落在"有真值且离真实云边 ≥2–3 像元"的位置；人工洞本身不腐蚀。
   - 调度 (50 epoch): 0–10 λ=0；10–30 线性升到 0.01；30–50 固定。推理不用 D。
   - 监控 D 准确率，长期 >95% 或 ≈50% 报警。
5. 评价: GAN 开/关对比洞内 RMSE、高频幅度比、高频相关、接缝指标。GAN 可能使 RMSE 变差，不预设结论。
6. 消融: OSTIA 预训练 vs 随机初始化；tw vs tw+bc；有无 C 通道；GAN 开/关。

## 二、待办 (按顺序)
1. [完成] 真实帧底图预计算 (`build_filled_store.py` + `fill_background_only.py`)，见 HANDOFF.md。
2. [完成] 人工洞样本函数 `datasets/artificial_holes.py` 与有效块统计 (单窗口样本约 10.6 s)；样本供给方式(预生成/在线)未定。
3. 单模型数据集: 30 h 窗口 (30 SST + 30 mask + C 通道 + 小时 sin/cos)，输入通道数随之变化。
4. 时间编码: 每个 FNO 块 LayerNorm 后加 FiLM，零初始化，只在微调阶段加。
5. 预训练迁移: OSTIA 预训练保持绝对 SST；新增输入通道权重零初始化；先核实 OSTIA 时间分辨率。
6. GAN: 新写训练脚本与数据集 (不动旧脚本)，`PatchDiscriminator(in_ch=2, n_down=3)`。
7. 重建基线 (λ=0) → 小规模消融 → 再开 GAN。

## 三、未定 / 风险
- 微调总 epoch 数: 默认沿用 train_jaxa_hourly.py 的 50。
- OSTIA 日步长 vs JAXA 小时步长的迁移不匹配，需用随机初始化消融验证。
- OSTIA GT 比 JAXA 平滑得多 (梯度小 3 倍、拉普拉斯小 36 倍)，预训练分布与微调不同。
- C=0 区域底图本身约 0.9 K 误差且边界有梯度突变，靠 C 通道让模型自行决定信任度，未验证。
- 真值紧贴真实云边偏冷约 0.4–0.5 K，GAN 真样本需避开云边。
- 低缺失率帧偏晴天，人工洞训练内部自洽，但真实云区推理是分布外。
- GPU 共享，启动训练前先 nvidia-smi；必须用 `/home/lz/miniconda3/envs/pytorch/bin/python`。
完整交接见 HANDOFF.md
