"""
JAXA微调数据集
关键特性：
1. 加载JAXA观测SST（7天窗口）
2. 在观测点上添加额外的块状mask（模拟云层）
3. 区分：原始云层mask、额外mask、可见区域
"""

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from typing import List, Tuple, Dict
import random


def generate_block_mask(observed_mask: np.ndarray,
                       target_ratio_range: Tuple[float, float] = (0.2, 0.4),
                       block_size_range: Tuple[int, int] = (15, 40),
                       mix_random_prob: float = 0.3) -> np.ndarray:
    """
    生成块状遮挡mask（模拟真实云层）

    Args:
        observed_mask: (H, W) bool数组，True=观测到的海洋像素
        target_ratio_range: 目标遮挡比例范围（相对于观测像素）
        block_size_range: 块大小范围（像素）
        mix_random_prob: 使用随机像素而非块状的概率

    Returns:
        extra_mask: (H, W) bool数组，True=额外遮挡
    """
    H, W = observed_mask.shape
    extra_mask = np.zeros((H, W), dtype=bool)

    # 获取可遮挡区域的坐标
    obs_coords = np.argwhere(observed_mask)
    num_observed = len(obs_coords)

    if num_observed == 0:
        return extra_mask

    # 目标遮挡像素数
    target_ratio = np.random.uniform(*target_ratio_range)
    target_num_pixels = int(num_observed * target_ratio)

    # 决定使用块状还是随机
    if np.random.rand() < mix_random_prob:
        # 30%概率：随机像素遮挡（保持多样性）
        mask_indices = np.random.choice(num_observed,
                                       size=target_num_pixels,
                                       replace=False)
        for idx in mask_indices:
            y, x = obs_coords[idx]
            extra_mask[y, x] = True
    else:
        # 70%概率：块状遮挡
        current_masked = 0

        while current_masked < target_num_pixels * 0.9:  # 至少达到90%的目标
            # 随机选择中心点（必须在观测区域内）
            center_idx = np.random.randint(num_observed)
            cy, cx = obs_coords[center_idx]

            # 随机块大小
            block_h = np.random.randint(*block_size_range)
            block_w = np.random.randint(*block_size_range)

            # 随机形状（矩形或圆形）
            shape = np.random.choice(['rect', 'circle'])

            if shape == 'rect':
                y1 = max(0, cy - block_h // 2)
                y2 = min(H, cy + block_h // 2)
                x1 = max(0, cx - block_w // 2)
                x2 = min(W, cx + block_w // 2)

                # 只遮挡观测区域
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if observed_mask[y, x] and not extra_mask[y, x]:
                            extra_mask[y, x] = True
                            current_masked += 1

            else:  # circle
                radius = min(block_h, block_w) // 2
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        if dy*dy + dx*dx <= radius*radius:
                            y, x = cy + dy, cx + dx
                            if 0 <= y < H and 0 <= x < W and observed_mask[y, x] and not extra_mask[y, x]:
                                extra_mask[y, x] = True
                                current_masked += 1

            # 防止死循环
            if current_masked >= target_num_pixels:
                break

    return extra_mask


class JAXAFineTuneDataset(Dataset):
    """
    JAXA微调数据集

    返回数据结构：
    {
        'input_sst_seq': [7, H, W] - 7天的输入SST（归一化，额外遮挡后）
        'mask_seq': [7, H, W] - 7天的总mask（原始云层+额外mask）
        'ground_truth_sst': [H, W] - 第7天的真值SST（归一化）
        'original_cloud_mask': [H, W] - 原始JAXA云层mask
        'extra_mask': [H, W] - 额外添加的mask
        'visible_mask': [H, W] - 输入中可见的区域
        'land_mask': [H, W] - 陆地mask
        'date_str': str - 日期字符串
    }
    """

    def __init__(
        self,
        stats_file: str,
        data_root: str,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        ostia_mean: float = 26.71,
        ostia_std: float = 2.69,
        seed: int = 42,
    ):
        """
        Args:
            stats_file: JAXA统计数据文件（analyze_jaxa_coverage.py生成的pkl）
            data_root: JAXA数据根目录
            split: 'train', 'val', or 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            ostia_mean: OSTIA均值（用于归一化）
            ostia_std: OSTIA标准差
            seed: 随机种子
        """
        self.data_root = Path(data_root)
        self.split = split
        self.ostia_mean = ostia_mean
        self.ostia_std = ostia_std

        # 加载统计数据
        with open(stats_file, 'rb') as f:
            data = pickle.load(f)

        self.land_mask = data['land_mask']
        valid_samples = data['valid_windows']  # 有完整7天窗口的样本
        self.ocean_pixels = data['ocean_pixels']

        # 划分数据集
        random.seed(seed)
        np.random.seed(seed)

        indices = list(range(len(valid_samples)))
        random.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train+n_val]
        else:  # test
            self.indices = indices[n_train+n_val:]

        self.samples = [valid_samples[i] for i in self.indices]

        print(f"JAXA {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Mean cloud coverage: {np.mean([s['cloud_coverage'] for s in self.samples])*100:.1f}%")
        print(f"  Mean observed pixels: {np.mean([s['observed_pixels'] for s in self.samples]):,.0f}")

    def _load_single_day(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单天数据

        Returns:
            sst_celsius: (H, W) SST摄氏度
            cloud_mask: (H, W) 云层mask（True=云层遮挡）
        """
        ds = xr.open_dataset(str(file_path))
        sst_kelvin = ds.sea_surface_temperature.values[0]
        ds.close()

        # 开尔文转摄氏度
        sst_celsius = sst_kelvin - 273.15

        # 云层mask（NaN的地方）
        cloud_mask = np.isnan(sst_celsius)

        # 填充NaN为0（后续会被mask掉）
        sst_celsius = np.nan_to_num(sst_celsius, nan=0.0)

        return sst_celsius, cloud_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        target_date = sample_info['date']
        target_file = sample_info['file_path']

        # 构建7天窗口的文件路径
        window_files = []
        for i in range(7):
            days_back = 6 - i
            date_i = target_date - timedelta(days=days_back)

            year_month = date_i.strftime('%Y%m')
            day = date_i.strftime('%d')
            filename = date_i.strftime('%Y%m%d') + '000000.nc'

            file_path = self.data_root / year_month / day / filename
            window_files.append(str(file_path))

        # 加载7天数据
        sst_seq = []
        cloud_mask_seq = []

        for file_path in window_files:
            sst, cloud = self._load_single_day(file_path)
            sst_seq.append(sst)
            cloud_mask_seq.append(cloud)

        # 第7天（目标天）
        gt_sst = sst_seq[-1]  # (H, W)
        original_cloud_mask = cloud_mask_seq[-1]  # (H, W)

        # 生成额外mask（只在观测到的海洋区域）
        observed_ocean = (~original_cloud_mask) & (~self.land_mask)
        extra_mask = generate_block_mask(observed_ocean)

        # 总mask = 原始云层 + 额外mask
        total_mask_day7 = original_cloud_mask | extra_mask

        # 可见区域 = 未被任何mask遮挡的区域
        visible_mask = (~total_mask_day7) & (~self.land_mask)

        # 归一化
        gt_sst_norm = (gt_sst - self.ostia_mean) / self.ostia_std

        # 准备输入序列
        input_sst_seq = []
        total_mask_seq = []

        for day_idx in range(7):
            sst_day = sst_seq[day_idx]
            cloud_day = cloud_mask_seq[day_idx]

            # 第7天使用总mask，前6天只用原始云层mask
            if day_idx == 6:
                mask_day = total_mask_day7
            else:
                mask_day = cloud_day

            # 归一化SST
            sst_norm = (sst_day - self.ostia_mean) / self.ostia_std

            # 应用mask（被遮挡的地方设为0）
            sst_masked = sst_norm.copy()
            sst_masked[mask_day | self.land_mask] = 0.0

            input_sst_seq.append(sst_masked)
            total_mask_seq.append(mask_day)

        return {
            'input_sst_seq': torch.FloatTensor(np.array(input_sst_seq)),  # [7, H, W]
            'mask_seq': torch.FloatTensor(np.array(total_mask_seq)),      # [7, H, W]
            'ground_truth_sst': torch.FloatTensor(gt_sst_norm),           # [H, W]
            'original_cloud_mask': torch.FloatTensor(original_cloud_mask.astype(float)),  # [H, W]
            'extra_mask': torch.FloatTensor(extra_mask.astype(float)),    # [H, W]
            'visible_mask': torch.FloatTensor(visible_mask.astype(float)), # [H, W]
            'land_mask': torch.FloatTensor(self.land_mask.astype(float)), # [H, W]
            'date_str': target_date.strftime('%Y-%m-%d'),
        }


if __name__ == '__main__':
    # 测试数据集
    print("测试JAXA微调数据集...")

    dataset = JAXAFineTuneDataset(
        stats_file='/root/data_for_agent_FNO_CBAM/FNO_CBAM/jaxa_analysis/jaxa_coverage_stats.pkl',
        data_root='/home/lz/jaxa_data/jaxa_extract_L3/',
        split='train',
    )

    print(f"\n加载第一个样本...")
    sample = dataset[0]

    print(f"  Date: {sample['date_str']}")
    print(f"  input_sst_seq: {sample['input_sst_seq'].shape}")
    print(f"  ground_truth_sst: {sample['ground_truth_sst'].shape}")
    print(f"  original_cloud_mask: {sample['original_cloud_mask'].sum():.0f} pixels")
    print(f"  extra_mask: {sample['extra_mask'].sum():.0f} pixels")
    print(f"  visible_mask: {sample['visible_mask'].sum():.0f} pixels")
    print(f"  land_mask: {sample['land_mask'].sum():.0f} pixels")

    # 验证mask关系
    ocean_pixels = (~sample['land_mask'].numpy().astype(bool)).sum()
    observed_pixels = (~sample['original_cloud_mask'].numpy().astype(bool) &
                       ~sample['land_mask'].numpy().astype(bool)).sum()
    extra_pixels = sample['extra_mask'].sum().item()
    visible_pixels = sample['visible_mask'].sum().item()

    print(f"\n  Mask统计:")
    print(f"    海洋像素: {ocean_pixels}")
    print(f"    原始观测: {observed_pixels} ({observed_pixels/ocean_pixels*100:.1f}%)")
    print(f"    额外遮挡: {extra_pixels} ({extra_pixels/observed_pixels*100:.1f}% of observed)")
    print(f"    最终可见: {visible_pixels} ({visible_pixels/observed_pixels*100:.1f}% of observed)")

    print("\n✅ 数据集测试通过!")
