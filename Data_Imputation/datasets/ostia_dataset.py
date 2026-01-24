"""
SST Dataset with 30-day temporal sequence
返回30天的SST序列 + mask序列，用于时间建模

注意：此数据集会对input_sst进行预处理，将缺失区域（=0）用最近邻插值填充
"""
import h5py
import numpy as np
from torch.utils.data import Dataset
from scipy import ndimage


class SSTDatasetTemporal(Dataset):
    """
    返回30天时间序列的SST数据集

    Returns:
        {
            'input_sst_seq': [30, H, W] - 30天的输入SST（带插值填充）
            'mask_seq': [30, H, W] - 30天的missing_mask
            'ground_truth_sst': [H, W] - 第30天的真值
            'missing_mask': [H, W] - 第30天的mask
            'land_mask': [H, W] - 陆地mask
            'time_index': int - 当前样本的时间索引
        }
    """

    def __init__(self, hdf5_path, normalize=True, mean=None, std=None, window_size=30):
        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.window_size = window_size

        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = f['ground_truth_sst'].shape[0]
            self.shape = f['ground_truth_sst'].shape[1:]

            # 计算统计量
            if mean is None or std is None:
                all_data = f['ground_truth_sst'][:]
                land_mask = f['land_mask'][:]
                ocean_data = all_data[:, land_mask == 0]
                self.mean = float(np.nanmean(ocean_data))
                self.std = float(np.nanstd(ocean_data))
            else:
                self.mean = mean
                self.std = std

        print(f"Loaded Temporal SST dataset from {hdf5_path}")
        print(f"  Samples: {self.num_samples}")
        print(f"  Window size: {window_size} days")
        print(f"  Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        print(f"  Normalized: {normalize}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            # 计算30天序列的起始索引
            if idx < self.window_size - 1:
                # 前29个样本：padding
                start_idx = 0
                num_valid = idx + 1
            else:
                start_idx = idx - (self.window_size - 1)
                num_valid = self.window_size

            # 读取30天的input_sst和missing_mask
            sst_seq = []
            mask_seq = []

            for i in range(start_idx, start_idx + num_valid):
                sst = f['input_sst'][i].astype(np.float32).copy()
                mask = f['missing_mask'][i].astype(np.float32)

                # 用最近邻插值填充缺失区域（=0或NaN）
                invalid = (sst == 0) | np.isnan(sst)
                if invalid.any():
                    valid_mask = (sst != 0) & (~np.isnan(sst))
                    if valid_mask.sum() > 0:
                        indices = ndimage.distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
                        sst = sst[tuple(indices)]
                    else:
                        # 如果完全没有有效值，用全局均值填充
                        sst = np.full_like(sst, self.mean)

                sst_seq.append(sst)
                mask_seq.append(mask)

            # 如果不足30天，用第一天padding
            while len(sst_seq) < self.window_size:
                sst_seq.insert(0, sst_seq[0])
                mask_seq.insert(0, mask_seq[0])

            sst_seq = np.stack(sst_seq)  # [30, H, W]
            mask_seq = np.stack(mask_seq)  # [30, H, W]

            # 第30天的真值和mask
            gt_sst = f['ground_truth_sst'][idx].astype(np.float32)
            missing_mask = f['missing_mask'][idx].astype(np.float32)
            land_mask = f['land_mask'][:].astype(np.float32)

        # 归一化
        if self.normalize:
            sst_seq = (sst_seq - self.mean) / self.std
            gt_sst = (gt_sst - self.mean) / self.std

        # NaN替换为0（陆地区域）
        sst_seq = np.nan_to_num(sst_seq, nan=0.0)
        gt_sst = np.nan_to_num(gt_sst, nan=0.0)

        return {
            'input_sst_seq': sst_seq,  # [30, H, W]
            'mask_seq': mask_seq,  # [30, H, W]
            'ground_truth_sst': gt_sst,  # [H, W]
            'missing_mask': missing_mask,  # [H, W]
            'land_mask': land_mask,  # [H, W]
            'time_index': idx
        }


if __name__ == '__main__':
    # 测试
    import torch

    train_path = '/data_new/sst_data/sst_missing_value_imputation/processed_data/processed_sst_train.h5'
    dataset = SSTDatasetTemporal(hdf5_path=train_path, normalize=True)

    print("\n" + "="*60)
    print("测试样本:")
    sample = dataset[100]

    print(f"  input_sst_seq shape: {sample['input_sst_seq'].shape}")
    print(f"  mask_seq shape: {sample['mask_seq'].shape}")
    print(f"  ground_truth_sst shape: {sample['ground_truth_sst'].shape}")
    print(f"  missing_mask shape: {sample['missing_mask'].shape}")

    print(f"\n  input_sst_seq range: [{sample['input_sst_seq'].min():.3f}, {sample['input_sst_seq'].max():.3f}]")

    # 检查30天mask互补性
    mask_seq = sample['mask_seq']  # [30, H, W]
    observation_count = (1 - mask_seq).sum(axis=0)  # [H, W]
    land_mask = sample['land_mask']
    ocean_mask = 1 - land_mask

    ocean_obs_count = observation_count[ocean_mask == 1]
    print(f"\n  海洋区域30天观测次数统计:")
    print(f"    0次观测: {(ocean_obs_count == 0).sum()} 像素")
    print(f"    1-10次: {((ocean_obs_count >= 1) & (ocean_obs_count <= 10)).sum()} 像素")
    print(f"    11-20次: {((ocean_obs_count >= 11) & (ocean_obs_count <= 20)).sum()} 像素")
    print(f"    21-30次: {((ocean_obs_count >= 21) & (ocean_obs_count <= 30)).sum()} 像素")
    print(f"    30次全观测: {(ocean_obs_count == 30).sum()} 像素")

    print("\n" + "="*60)
    print("✓ Dataset测试通过")
