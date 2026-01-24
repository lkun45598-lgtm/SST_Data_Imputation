# 数据预处理流程

## 概述

JAXA卫星SST数据预处理Pipeline，将原始hourly NC文件转换为模型可用的30天时间序列H5数据。

## 预处理流程图

```
JAXA原始数据 (hourly NC)
        │
        ▼
┌─────────────────────────────┐
│  Step 1: 时间加权填充        │
│  temporal_weighted_fill.py  │
│  hourly → daily            │
│  逆时间距离加权填充缺失值    │
└─────────────────────────────┘
        │
        ▼
    jaxa_weighted_aligned/*.h5
        │
        ▼
┌─────────────────────────────┐
│  Step 2: 低通滤波           │
│  lowpass_filter.py          │
│  去除高频噪声               │
└─────────────────────────────┘
        │
        ▼
    jaxa_filtered/*.h5
        │
        ▼
┌─────────────────────────────┐
│  Step 3: KNN空间填充        │
│  knn_fill.py                │
│  渐进式KNN填充剩余缺失      │
└─────────────────────────────┘
        │
        ▼
    jaxa_knn_filled/*.h5  ← 模型输入数据
```

---

## Step 1: 时间加权填充

**脚本**: `preprocessing/temporal_weighted_fill.py`

### 算法原理

对于目标时刻 `t` 的每个缺失像素，使用回溯窗口内历史观测的加权平均值：

```
权重(t_history) = 1 / (t - t_history)
填充值 = Σ(w_i × v_i) / Σ(w_i)
```

### 核心特性

- **逆时间距离加权**: 越近的观测权重越大
- **自适应回溯窗口**: 优先24小时，不足时扩展至48小时
- **hourly → daily**: 同时完成时间聚合
- **并行处理**: 支持多核CPU并行

### 输入输出

| 项目 | 说明 |
|------|------|
| 输入 | `/data/sst_data/.../jaxa_extract_L3/YYYYMM/DD/YYYYMMDDHHMMSS.nc` |
| 输出 | `jaxa_weighted_aligned/series_XX/*.h5` |
| 时间范围 | 2015-2025 (10年) |

### 使用方法

```bash
cd /path/to/project
python preprocessing/temporal_weighted_fill.py --mode full --workers 64
```

---

## Step 2: 低通滤波

**脚本**: `preprocessing/lowpass_filter.py`

### 算法原理

应用空间低通滤波去除高频噪声，保留SST的大尺度空间结构。

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| cutoff_frequency | 0.1 | 截止频率 |
| filter_type | butterworth | 滤波器类型 |

### 输入输出

| 项目 | 说明 |
|------|------|
| 输入 | `jaxa_weighted_aligned/*.h5` |
| 输出 | `jaxa_filtered/*.h5` |

### 使用方法

```bash
python preprocessing/lowpass_filter.py
```

---

## Step 3: KNN空间填充

**脚本**: `preprocessing/knn_fill.py`

### 算法原理

渐进式KNN填充：从缺失区域边缘开始，逐层向内填充。

```
迭代过程:
1. 找到缺失区域的边界像素
2. 对每个边界像素，找K个最近的有效邻居
3. 使用距离加权平均填充
4. 更新边界，重复直到全部填充
```

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k_neighbors | 5 | KNN邻居数 |
| max_iterations | 100 | 最大迭代次数 |
| distance_power | 2 | 距离权重指数 |

### 输入输出

| 项目 | 说明 |
|------|------|
| 输入 | `jaxa_filtered/*.h5` |
| 输出 | `jaxa_knn_filled/*.h5` |

### 使用方法

```bash
python preprocessing/knn_fill.py
```

---

## H5文件格式

### 输出数据结构

```python
with h5py.File('jaxa_knn_filled_YYYYMMDD.h5', 'r') as f:
    sst_knn_filled = f['sst_knn_filled'][:]  # [H, W] float32, 单位: Kelvin
    missing_mask = f['missing_mask'][:]       # [H, W] uint8, 1=原始缺失
    lat = f['lat'][:]                         # [H] float32
    lon = f['lon'][:]                         # [W] float32
```

### 数据规格

| 属性 | 值 |
|------|-----|
| 空间分辨率 | 0.05° × 0.05° |
| 空间范围 | 20°N-50°N, 115°E-150°E |
| 网格尺寸 | 600 × 700 (lat × lon) |
| 数据类型 | float32 (SST), uint8 (mask) |
| 单位 | Kelvin |

---

## 数据统计

### 各阶段缺失率变化

| 阶段 | 平均缺失率 | 说明 |
|------|-----------|------|
| 原始数据 | 40-80% | 云层遮挡严重 |
| 时间加权填充后 | 20-40% | 利用时间信息 |
| 低通滤波后 | 20-40% | 不改变缺失率 |
| KNN填充后 | 0% | 完全填充 |

### 数据集划分

| 数据集 | Series ID | 年份 | 样本数 |
|--------|-----------|------|--------|
| 训练集 | 0-7 | 8年 | ~2900 |
| 验证集 | 8 | 1年 | ~365 |
| 测试集 | 9 | 1年 | ~365 |

---

## 运行完整预处理

```bash
# 1. 时间加权填充 (最耗时)
python preprocessing/temporal_weighted_fill.py --mode full --workers 64

# 2. 低通滤波
python preprocessing/lowpass_filter.py

# 3. KNN填充
python preprocessing/knn_fill.py

# 检查结果
ls -lh jaxa_knn_filled/
```

**预计总时间**: 约24-48小时 (取决于CPU核心数)
