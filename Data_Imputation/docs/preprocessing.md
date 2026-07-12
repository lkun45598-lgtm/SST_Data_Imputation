# 数据预处理流程

## 概述

JAXA 卫星 SST 数据预处理 Pipeline，将原始逐时 NC 文件转换为模型可用、无缺失的日序列 H5 数据。研究区域为**南海北部**，最终产物 `jaxa_knn_filled_XX.h5` 中每个像素的温度场完全填满（海洋区无 NaN），并配套多张掩码用于区分像素来源（真实观测 / 时间回填 / KNN 重建）。

## 预处理流程图

```
JAXA 原始数据 (逐时 NC, 单位 Kelvin)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: 时间加权填充       │
│  temporal_weighted_fill.py  │
│  按天回溯同一小时的历史观测  │
│  1/Δdays 逆时间距离加权填充  │
└─────────────────────────────┘
        │  输出 sst + missing_mask + fill_mask
        ▼
┌─────────────────────────────┐
│  Stage 2: 高斯低通滤波       │
│  lowpass_filter.py          │
│  scipy gaussian_filter σ=1.5│
│  仅平滑“时间填充像素”        │
│  原始观测保持不变            │
└─────────────────────────────┘
        │  残余缺失仍为 NaN
        ▼
┌─────────────────────────────┐
│  Stage 3: 3D 因果渐进式 KNN  │
│  knn_fill_3d.py             │
│  时空 IDW 插值填满残余缺失   │
└─────────────────────────────┘
        │
        ▼
    jaxa_knn_filled/*.h5  ← 模型输入数据（无缺失）
```

> **逐时家族 h00–h23**：`hourly_preprocess.py` 对小时 h（1–23）执行与 h=0 完全相同的三阶段流程，输出到
> `experiments/hourly_data/h{hh}/jaxa_knn_filled_XX.h5`；**h00 即 `jaxa_knn_filled/` 目录**，由
> `temporal_weighted_fill.py` → `lowpass_filter.py` → 3D KNN 生成。因此每一天、每个整点各有一套独立的三阶段产物。

---

## Stage 1: 时间加权填充

**脚本**: `preprocessing/temporal_weighted_fill.py`（h=0）/ `preprocessing/hourly_preprocess.py`（h=1–23，同一算法）

### 算法原理

对于目标帧（某一天某个整点）中的**每个云遮挡缺失像素**，只用**之前若干天里同一小时**的历史观测做逆时间距离加权平均：

```
权重(历史帧) = 1 / Δdays        # Δdays = 目标日与历史日之间相差的天数
填充值        = Σ(w_i × v_i) / Σ(w_i)
```

### 核心特性

- **逆时间距离加权**：距离目标日越近的历史观测权重越大（`w = 1/Δdays`）。
- **同小时逐日回溯**：只比较相同整点的帧，避免昼夜温差干扰。
- **回溯窗口 48 小时**：向前最多回溯约 2 天（`LOOKBACK_WINDOW = 48` 小时）。
- **首帧不填充**：序列起始帧没有历史数据可用，保持原样。
- **并行处理**：支持多核 CPU 并行（`--workers`）。

### 输出

Stage 1 产出三类逐帧数据（供后续阶段使用）：

| 数据 | 含义 |
|------|------|
| `sst` | 时间加权填充后的温度场（Kelvin），仍可能残留缺失 |
| `missing_mask` | 1 = Stage 1 之后仍然缺失（残余缺失，将由 KNN 填充） |
| `fill_mask` | 1 = 本阶段被时间回填的像素 |

### 输入输出路径

| 项目 | 说明 |
|------|------|
| 输入 | `/data/sst_data/.../jaxa_extract_L3/YYYYMM/DD/YYYYMMDDHHMMSS.nc` |
| 输出（h=0） | `jaxa_weighted_aligned/jaxa_weighted_series_XX.h5` |

### 使用方法

```bash
# h=0（原始 30 天/年度序列生成）
python preprocessing/temporal_weighted_fill.py --mode full --workers 64

# h=1..23（逐时家族，单独指定 series 或 all）
python preprocessing/hourly_preprocess.py --hour 1 --series all --workers 64
```

---

## Stage 2: 高斯低通滤波

**脚本**: `preprocessing/lowpass_filter.py`

### 算法原理

使用 **scipy `ndimage.gaussian_filter`（高斯低通，σ=1.5）** 平滑温度场，去除时间回填引入的高频伪影。关键在于**写回策略**：

- **只对“时间填充像素”（`fill_mask == 1`）写回滤波值**（见 `_writeback_region` 与 `apply_gaussian_filter`）。
- **原始观测像素永远保持不变**（真值绝不被滤波覆盖）。
- **残余缺失像素仍为 NaN**，留给 Stage 3 处理。

高斯卷积的输入会先把非有效区临时填成有效区均值，使填充区能借助真实邻居平滑；但结果只写回填充像素，观测真值原封不动。

> 说明：脚本另实现了 `median` / `uniform` / `bilateral` 等方法备选，默认且实际使用的是 `gaussian`。

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| method | gaussian | 滤波器类型（scipy `gaussian_filter`） |
| sigma | 1.5 | 高斯核标准差 |
| 写回范围 | fill_mask==1 | 仅平滑时间填充像素，保留原始观测 |

### 输入输出

| 项目 | 说明 |
|------|------|
| 输入 | `jaxa_weighted_aligned/*.h5` |
| 输出（h=0） | `jaxa_filtered/*.h5` |

### 使用方法

```bash
python preprocessing/lowpass_filter.py --mode full --method gaussian --sigma 1.5
```

---

## Stage 3: 3D 因果渐进式 KNN 空间填充

**脚本**: `preprocessing/knn_fill_3d.py`（`progressive_knn_fill_3d`）

### 算法原理

对 Stage 2 后仍缺失的像素，在**时空域**做 3D IDW（反距离加权）插值填满：

- **因果约束**：只使用**过去帧**（`t' < t`），不使用未来帧。
- **双层 KDTree**：Tier 1 为历史帧树（时间窗口内已填好的过去帧，静态）；Tier 2 为当前帧树（随填充渐进更新）。合并两层结果后取 k 个最近邻。
- **渐进填充顺序**：按 2D 缺失密度升序（边缘先填、中心后填）。
- **偏差校正**：将历史帧温度平移到当前帧的观测均值水平，抵消逐日整体升降温。
- **兜底**：极少数残余 NaN 用全局海洋均值填充；陆地始终保持 NaN。

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k | 20 | KNN 近邻数 |
| radius | 20 | 2D 缺失密度计算半径 |
| power | 2 | IDW 距离权重指数 |
| time_weight | 0.9 | 时间维缩放系数 |
| space_weight | 1.0 | 空间维缩放系数 |
| window_size | 7 | 时间回溯窗口（天） |
| batch_size | 500 | 批量查询大小 |

### 输入输出

| 项目 | 说明 |
|------|------|
| 输入 | `jaxa_filtered/*.h5`（h=0）；逐时流程中为内存传递 |
| 输出 | `jaxa_knn_filled/*.h5`（h=0）/ `experiments/hourly_data/h{hh}/*.h5` |

---

## H5 文件格式

### 输出数据结构

最终产物 `jaxa_knn_filled_XX.h5` 的字段如下：

```python
with h5py.File('jaxa_knn_filled_00.h5', 'r') as f:
    sst_data              = f['sst_data'][:]              # (T, H, W) float32, Kelvin, 海洋区无缺失
    land_mask             = f['land_mask'][:]             # (H, W)    uint8, 1=陆地
    original_obs_mask     = f['original_obs_mask'][:]     # (T, H, W) uint8, 1=真实观测像素
    temporal_fill_mask    = f['temporal_fill_mask'][:]    # (T, H, W) uint8, 1=Stage 1 时间回填
    original_missing_mask = f['original_missing_mask'][:] # (T, H, W) uint8, 1=残余缺失(由 KNN 填充)
    latitude              = f['latitude'][:]              # (H,) float32
    longitude             = f['longitude'][:]             # (W,) float32
    timestamps            = f['timestamps'][:]            # (T,) S32, ISO 时间戳
```

掩码由以下逻辑生成：

```
original_obs_mask     = (temporal_fill_mask == 0) & (原始缺失 == 0)   # 真实测量
temporal_fill_mask    = Stage 1 的 fill_mask                          # 时间回填
original_missing_mask = Stage 1 之后的 missing_mask                    # 残余缺失 → KNN 重建
```

### 像素来源（provenance）

`sst_data` 中三类像素的来源不同，务必区分：

| 掩码 | 含义 | 是否真实测量 |
|------|------|--------------|
| `original_obs_mask == 1` | 卫星真实观测 | **是**，唯一的真值 |
| `temporal_fill_mask == 1` | Stage 1 时间回填 | 否，重建/合成值 |
| `original_missing_mask == 1` | Stage 3 KNN 重建 | 否，重建/合成值 |

**只有 `original_obs_mask` 的像素是真实测量值**；时间回填与 KNN 像素均为重建/合成，评估误差时应仅在真实观测像素上计算。

### 数据规格

| 属性 | 值 |
|------|-----|
| 研究区域 | 南海北部 |
| 空间范围 | 纬度 15°N–24°N，经度 111°E–118°E |
| 网格尺寸 | 451 × 351 (lat × lon) |
| 空间分辨率 | 约 0.02° × 0.02° |
| 时间维 | 每个序列 ~267–365 帧（逐日，单一整点） |
| 数据类型 | float32 (SST), uint8 (mask) |
| 单位 | Kelvin |

---

## 数据统计

### 各阶段缺失率变化

| 阶段 | 缺失情况 | 说明 |
|------|-----------|------|
| 原始数据 | 云遮挡严重 | 逐时观测缺失率高 |
| Stage 1 时间加权填充后 | 明显下降 | 利用同小时历史观测 |
| Stage 2 高斯低通滤波后 | 不变 | 仅平滑填充像素，不改变缺失分布 |
| Stage 3 3D KNN 填充后 | 0%（海洋区） | 完全填满，陆地保持 NaN |

### 年度序列

共 **9 个年度序列（00–08）**，每个是某一整点的逐日时间序列。各序列起始日期与帧数（h=0）：

| Series | 起始日期 | 帧数 |
|--------|----------|------|
| 00 | 2017-07-06 | 360 |
| 01 | 2016-07-06 | 317 |
| 02 | 2021-07-05 | 365 |
| 03 | 2018-07-06 | 345 |
| 04 | 2019-07-07 | 356 |
| 05 | 2020-07-05 | 363 |
| 06 | 2022-07-05 | 365 |
| 07 | 2023-07-05 | 365 |
| 08 | 2024-07-04 | 267 |

### 数据集划分

| 数据集 | Series ID | 说明 |
|--------|-----------|------|
| 训练集 | 0–7 | 8 个年度序列 |
| 验证集 | 8 | 1 个年度序列 |

> 数据集只有训练/验证两部分：**没有 series 9，也没有单独的测试集划分**。

---

## 运行完整预处理

```bash
# 1. 时间加权填充 (Stage 1, 最耗时)
python preprocessing/temporal_weighted_fill.py --mode full --workers 64

# 2. 高斯低通滤波 (Stage 2, 仅平滑时间填充像素)
python preprocessing/lowpass_filter.py --mode full --method gaussian --sigma 1.5

# 3. 3D 因果 KNN 填充 (Stage 3)
#    knn_fill_3d 由上游脚本调用 progressive_knn_fill_3d 完成

# 逐时家族 h01–h23（三阶段一体化）
python preprocessing/hourly_preprocess.py --hour 1 --series all --workers 64

# 检查结果
ls -lh jaxa_knn_filled/
```
