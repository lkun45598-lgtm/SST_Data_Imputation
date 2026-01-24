#!/usr/bin/env python3
"""
JAXA Fine-tuning Dataset with Artificial Masking
用于JAXA微调的数据集，支持30天输入序列和人工挖空

特点:
1. 双输入: sst_seq [30, H, W] + mask_seq [30, H, W]
2. 人工方形挖空: 只在原始观测区域 (original_obs_mask) 上生成
3. Loss区域: artificial_mask ∩ original_obs_mask
4. 异步预取和缓存

作者: Claude Code
日期: 2026-01-19
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from threading import Thread
from queue import Queue
import time


class SquareMaskGenerator:
    """方形挖空生成器"""

    def __init__(self, mask_ratio=0.2, min_size=10, max_size=50, seed=None):
        """
        Args:
            mask_ratio: 目标挖空比例（相对于有效区域）
            min_size: 方形最小边长
            max_size: 方形最大边长
            seed: 随机种子
        """
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_mask, target_ratio=None):
        """
        在valid_mask区域内生成方形挖空

        Args:
            valid_mask: 有效区域掩码 (H, W), 1=可挖空, 0=不可挖空
            target_ratio: 目标挖空比例，None则使用默认值

        Returns:
            artificial_mask: 挖空掩码 (H, W), 1=挖空, 0=保留
        """
        if target_ratio is None:
            target_ratio = self.mask_ratio

        H, W = valid_mask.shape
        artificial_mask = np.zeros((H, W), dtype=np.float32)

        valid_count = valid_mask.sum()
        if valid_count == 0:
            return artificial_mask

        target_masked = int(valid_count * target_ratio)
        current_masked = 0

        # 获取有效区域的边界
        valid_y, valid_x = np.where(valid_mask == 1)
        if len(valid_y) == 0:
            return artificial_mask

        y_min, y_max = valid_y.min(), valid_y.max()
        x_min, x_max = valid_x.min(), valid_x.max()

        max_attempts = 1000
        attempts = 0

        while current_masked < target_masked and attempts < max_attempts:
            # 随机方形大小
            size = self.rng.integers(self.min_size, self.max_size + 1)

            # 随机位置（在有效区域范围内）
            if y_max - size < y_min or x_max - size < x_min:
                attempts += 1
                continue

            y_start = self.rng.integers(y_min, max(y_min + 1, y_max - size + 1))
            x_start = self.rng.integers(x_min, max(x_min + 1, x_max - size + 1))

            y_end = min(y_start + size, H)
            x_end = min(x_start + size, W)

            # 只挖空valid_mask区域内的像素
            region = valid_mask[y_start:y_end, x_start:x_end].copy()
            new_masked = region.sum() - (artificial_mask[y_start:y_end, x_start:x_end] * region).sum()

            if new_masked > 0:
                artificial_mask[y_start:y_end, x_start:x_end] = np.where(
                    region == 1, 1.0, artificial_mask[y_start:y_end, x_start:x_end]
                )
                current_masked = (artificial_mask * valid_mask).sum()

            attempts += 1

        return artificial_mask


class JAXAFinetuneDataset(Dataset):
    """
    JAXA微调数据集

    输入:
        - KNN填充后的JAXA数据 (jaxa_knn_filled/*.h5)

    输出:
        {
            'input_sst_seq': [30, H, W] - 30天SST序列（挖空位置填充均值）
            'mask_seq': [30, H, W] - 30天mask序列（第30天为artificial_mask）
            'ground_truth_sst': [H, W] - 第30天真值（KNN填充后的完整数据）
            'loss_mask': [H, W] - Loss计算区域 = artificial_mask ∩ original_obs_mask
            'original_obs_mask': [H, W] - 原始观测区域
            'time_index': int
        }
    """

    def __init__(self, data_dir, series_ids=None, window_size=30,
                 mask_ratio=0.2, min_mask_size=10, max_mask_size=50,
                 normalize=True, mean=None, std=None, cache_size=100, seed=42):
        """
        Args:
            data_dir: KNN填充后的数据目录
            series_ids: 使用的序列ID列表，None则使用全部
            window_size: 时间窗口大小（默认30天）
            mask_ratio: 人工挖空比例（默认20%）
            min_mask_size: 最小方形边长
            max_mask_size: 最大方形边长
            normalize: 是否归一化
            mean/std: 归一化参数，None则自动计算
            cache_size: 缓存帧数
            seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.normalize = normalize
        self.cache_size = cache_size

        # 确定使用的序列
        if series_ids is None:
            series_ids = [0, 1, 2]
        self.series_ids = series_ids

        # 加载所有序列的元数据
        self.series_info = []
        self.cumulative_frames = [0]

        total_frames = 0
        all_sst_for_stats = []

        for sid in series_ids:
            h5_path = self.data_dir / f'jaxa_knn_filled_{sid:02d}.h5'
            if not h5_path.exists():
                print(f"Warning: {h5_path} not found, skipping...")
                continue

            with h5py.File(h5_path, 'r') as f:
                num_frames = f['sst_data'].shape[0]
                shape = f['sst_data'].shape[1:]

                self.series_info.append({
                    'series_id': sid,
                    'h5_path': str(h5_path),
                    'num_frames': num_frames,
                    'shape': shape
                })

                total_frames += num_frames
                self.cumulative_frames.append(total_frames)

                # 收集统计数据
                if mean is None or std is None:
                    sst_sample = f['sst_data'][::10]  # 每10帧采样
                    all_sst_for_stats.append(sst_sample)

        self.total_frames = total_frames
        self.shape = shape if self.series_info else (451, 351)

        # 计算归一化参数
        if mean is None or std is None:
            if all_sst_for_stats:
                all_sst = np.concatenate(all_sst_for_stats, axis=0)
                valid_sst = all_sst[~np.isnan(all_sst)]
                self.mean = float(np.mean(valid_sst))
                self.std = float(np.std(valid_sst))
            else:
                # 默认JAXA SST统计值（开尔文）
                self.mean = 300.0
                self.std = 5.0
        else:
            self.mean = mean
            self.std = std

        # 方形挖空生成器
        self.mask_generator = SquareMaskGenerator(
            mask_ratio=mask_ratio,
            min_size=min_mask_size,
            max_size=max_mask_size,
            seed=seed
        )

        # 缓存
        self._cache = {}
        self._cache_order = []

        print(f"\nJAXA Finetune Dataset initialized:")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Series: {[s['series_id'] for s in self.series_info]}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Window size: {window_size}")
        print(f"  Mask ratio: {mask_ratio*100:.0f}%")
        print(f"  Mask size: {min_mask_size}-{max_mask_size}")
        print(f"  Normalize: {normalize} (mean={self.mean:.2f}, std={self.std:.2f})")
        print(f"  Cache size: {cache_size}")

    def __len__(self):
        return self.total_frames

    def _get_series_and_local_idx(self, global_idx):
        """根据全局索引获取序列ID和局部索引"""
        for i, info in enumerate(self.series_info):
            if global_idx < self.cumulative_frames[i + 1]:
                local_idx = global_idx - self.cumulative_frames[i]
                return i, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def _load_frame(self, series_idx, local_idx):
        """加载单帧数据（带缓存）"""
        cache_key = (series_idx, local_idx)

        if cache_key in self._cache:
            return self._cache[cache_key]

        info = self.series_info[series_idx]

        with h5py.File(info['h5_path'], 'r') as f:
            sst = f['sst_data'][local_idx].astype(np.float32)
            original_obs_mask = f['original_obs_mask'][local_idx].astype(np.float32)
            original_missing_mask = f['original_missing_mask'][local_idx].astype(np.float32)

            # 读取陆地掩码（如果存在）
            if 'land_mask' in f:
                land_mask = f['land_mask'][:].astype(np.float32)
            else:
                # 兼容旧数据：陆地=NaN的位置
                land_mask = np.isnan(sst).astype(np.float32)

        data = {
            'sst': sst,
            'original_obs_mask': original_obs_mask,
            'original_missing_mask': original_missing_mask,
            'land_mask': land_mask
        }

        # 更新缓存
        if len(self._cache) >= self.cache_size:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        self._cache[cache_key] = data
        self._cache_order.append(cache_key)

        return data

    def __getitem__(self, idx):
        series_idx, local_idx = self._get_series_and_local_idx(idx)
        info = self.series_info[series_idx]

        # 计算30天序列的起始索引
        if local_idx < self.window_size - 1:
            start_local_idx = 0
            num_valid = local_idx + 1
        else:
            start_local_idx = local_idx - (self.window_size - 1)
            num_valid = self.window_size

        # 加载30天数据
        sst_seq = []
        mask_seq = []

        for i in range(start_local_idx, start_local_idx + num_valid):
            frame_data = self._load_frame(series_idx, i)
            sst_seq.append(frame_data['sst'].copy())
            # 使用原始missing_mask作为mask_seq（0=有效观测，1=缺失）
            mask_seq.append(frame_data['original_missing_mask'].copy())

        # Padding（不足30天用第一天填充）
        while len(sst_seq) < self.window_size:
            sst_seq.insert(0, sst_seq[0].copy())
            mask_seq.insert(0, mask_seq[0].copy())

        sst_seq = np.stack(sst_seq)  # [30, H, W]
        mask_seq = np.stack(mask_seq)  # [30, H, W]

        # 第30天的数据
        target_data = self._load_frame(series_idx, local_idx)
        ground_truth_sst = target_data['sst'].copy()  # KNN填充后的完整数据
        original_obs_mask = target_data['original_obs_mask'].copy()
        land_mask = target_data['land_mask'].copy()

        # 生成人工挖空（只在original_obs_mask区域，排除陆地）
        valid_for_mask = original_obs_mask * (1 - land_mask)  # 海洋观测区域
        artificial_mask = self.mask_generator.generate(valid_for_mask)

        # Loss区域 = artificial_mask ∩ original_obs_mask ∩ ocean
        loss_mask = (artificial_mask * original_obs_mask * (1 - land_mask)).astype(np.float32)

        # 更新第30天的mask_seq为artificial_mask
        mask_seq[-1] = artificial_mask

        # 在挖空位置填充均值
        fill_value = self.mean  # 始终用mean填充，归一化后变成0
        sst_seq_input = sst_seq.copy()
        sst_seq_input[-1] = np.where(artificial_mask == 1, fill_value, sst_seq[-1])

        # 归一化
        if self.normalize:
            sst_seq_input = (sst_seq_input - self.mean) / self.std
            ground_truth_sst = (ground_truth_sst - self.mean) / self.std

        # 处理NaN
        sst_seq_input = np.nan_to_num(sst_seq_input, nan=0.0)
        ground_truth_sst = np.nan_to_num(ground_truth_sst, nan=0.0)

        return {
            'input_sst_seq': sst_seq_input.astype(np.float32),  # [30, H, W]
            'mask_seq': mask_seq.astype(np.float32),  # [30, H, W]
            'ground_truth_sst': ground_truth_sst.astype(np.float32),  # [H, W]
            'loss_mask': loss_mask.astype(np.float32),  # [H, W]
            'original_obs_mask': original_obs_mask.astype(np.float32),  # [H, W]
            'land_mask': land_mask.astype(np.float32),  # [H, W]
            'time_index': idx
        }


class AsyncPrefetchDataLoader:
    """异步预取数据加载器"""

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4,
                 prefetch_factor=2, pin_memory=True, drop_last=False):
        """
        Args:
            dataset: PyTorch Dataset
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: DataLoader worker数
            prefetch_factor: 预取因子
            pin_memory: 是否固定内存
            drop_last: 是否丢弃最后不完整批次
        """
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=drop_last
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def test_dataset():
    """测试数据集"""
    print("="*70)
    print("Testing JAXAFinetuneDataset")
    print("="*70)

    # 创建数据集
    dataset = JAXAFinetuneDataset(
        data_dir='/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled',
        series_ids=[0],  # 只测试第一个序列
        window_size=30,
        mask_ratio=0.2,
        normalize=True,
        cache_size=50
    )

    print(f"\nDataset length: {len(dataset)}")

    # 测试单样本
    print("\nTesting single sample (idx=50)...")
    sample = dataset[50]

    print(f"\nSample shapes:")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")

    # 检查挖空比例
    loss_mask = sample['loss_mask']
    original_obs_mask = sample['original_obs_mask']

    actual_mask_ratio = loss_mask.sum() / (original_obs_mask.sum() + 1e-8) * 100
    print(f"\nMask statistics:")
    print(f"  Original obs pixels: {original_obs_mask.sum():.0f}")
    print(f"  Loss mask pixels: {loss_mask.sum():.0f}")
    print(f"  Actual mask ratio: {actual_mask_ratio:.1f}%")

    # 测试DataLoader
    print("\nTesting DataLoader...")
    loader = AsyncPrefetchDataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2
    )

    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("Dataset test passed!")
    print("="*70)


if __name__ == '__main__':
    test_dataset()
