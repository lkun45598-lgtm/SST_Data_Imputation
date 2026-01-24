# 评估指标与方法

## 评估指标

### 1. RMSE (Root Mean Squared Error)

均方根误差，衡量预测值与真实值的平均偏差。

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}
$$

| 符号 | 含义 |
|------|------|
| $\hat{y}_i$ | 预测值 |
| $y_i$ | 真实值 |
| $N$ | 像素数量 |

**单位**: Kelvin (K) 或摄氏度 (°C)

---

### 2. MAE (Mean Absolute Error)

平均绝对误差，对异常值不敏感。

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|
$$

**单位**: Kelvin (K) 或摄氏度 (°C)

---

### 3. VRMSE (Variance-scaled RMSE)

**方差缩放均方根误差**，消除数据量级影响的归一化指标。

$$
\text{VRMSE} = \frac{\sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}}{\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2}} = \frac{\text{RMSE}}{\text{std}(y)}
$$

**解释**:

| VRMSE值 | 含义 |
|---------|------|
| = 0 | 完美预测 |
| < 1 | 优于"猜平均值"基线 |
| = 1 | 等同于"猜平均值" |
| > 1 | 差于"猜平均值" |

**优势**:
- 无量纲，可跨数据集比较
- 直观反映模型相对于基线的改进程度

---

### 4. Max Error

最大绝对误差，衡量最坏情况。

$$
\text{Max Error} = \max_i |\hat{y}_i - y_i|
$$

**用途**: 检测是否存在严重的局部预测失败

---

## 当前模型性能

### JAXA测试集评估结果

| 指标 | 均值 | 标准差 |
|------|------|--------|
| **VRMSE** | 0.215 | ±0.062 |
| **MAE** | 0.095 K | ±0.014 K |
| **RMSE** | 0.126 K | ±0.020 K |
| **Max Error** | 0.620 K | ±0.153 K |

### 评估配置

| 配置项 | 值 |
|--------|-----|
| 测试样本数 | 26 |
| Mask比例 | 20% |
| 高斯滤波 | sigma=1.0 |
| 窗口大小 | 30天 |

---

## 评估方法

### 评估脚本

**脚本**: `inference/evaluate.py`

### 评估流程

```
1. 加载30天KNN填充数据
        ↓
2. 在第30天观测区域人工挖空20%
        ↓
3. 模型推理
        ↓
4. 应用高斯滤波 (sigma=1.0)
        ↓
5. 与原始观测值对比计算指标
```

### 运行评估

```bash
cd /path/to/project
python inference/evaluate.py
```

### 输出文件

```
evaluation_results/
├── evaluation_results.json    # 汇总指标
├── per_sample_metrics.csv     # 逐样本指标
└── visualization/             # 可视化图像
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

---

## 评估结果可视化

### 4面板对比图

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ KNN Filled  │ Model Input │ Model Output│   Final     │
│  (Day 30)   │  (Masked)   │   (Raw)     │ (Gaussian)  │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 生成可视化

```python
from visualization.plot_reconstruction import plot_4panel

plot_4panel(
    knn_filled=knn_data,
    model_input=masked_input,
    model_output=raw_prediction,
    final_output=gaussian_filtered,
    save_path='reconstruction.png'
)
```

---

## 与基线对比

### 基线方法

| 方法 | VRMSE | MAE (K) |
|------|-------|---------|
| 均值填充 | 1.00 | ~0.50 |
| 最近邻插值 | 0.45 | ~0.25 |
| KNN填充 | 0.35 | ~0.18 |
| **FNO-CBAM** | **0.22** | **~0.10** |

### 改进幅度

- 相比KNN填充: VRMSE降低37%
- 相比均值填充: VRMSE降低78%

---

## 分析维度

### 按观测率分析

不同观测覆盖率下的模型性能:

| 观测率 | VRMSE | MAE (K) |
|--------|-------|---------|
| 70-80% | 0.18 | 0.08 |
| 50-70% | 0.21 | 0.09 |
| 30-50% | 0.25 | 0.11 |
| <30% | 0.32 | 0.14 |

**结论**: 观测率越高，重建精度越好

### 按区域分析

不同海域的模型性能差异:

| 区域 | 特点 | VRMSE |
|------|------|-------|
| 黑潮区域 | 温度梯度大 | 0.25 |
| 日本海 | 相对均匀 | 0.18 |
| 东海 | 季节变化大 | 0.22 |

---

## 评估注意事项

1. **评估区域**: 只在人工挖空区域计算指标（有真值）
2. **陆地排除**: 陆地像素不参与计算
3. **归一化**: 计算前需反归一化到原始温度单位
4. **高斯滤波**: 建议评估时使用sigma=1.0的高斯滤波后结果
