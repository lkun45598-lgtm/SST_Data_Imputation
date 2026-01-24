"""
SST Dataset with 30-day temporal sequence - 使用预填充数据
无需实时KNN计算，数据加载极快
"""
import h5py
import numpy as np
from torch.utils.data import Dataset


class SSTDatasetTemporalFilled(Dataset):
    """
    返回30天时间序列的SST数据集 - 使用预填充数据版本

    要求数据文件包含 'input_sst_filled' 字段（已做KNN填充）

    Returns:
        {
            'input_sst_seq': [30, H, W] - 30天的输入SST（已填充）
            'mask_seq': [30, H, W] - 30天的missing_mask
            'ground_truth_sst': [H, W] - 第30天的真值
            'missing_mask': [H, W] - 第30天的mask
            'land_mask': [H, W] - 陆地mask
            'time_index': int - 当前样本的时间索引
        }
    """

    def __init__(self, hdf5_path, normalize=True, mean=None, std=None, window_size=30, preload=True):
        """
        Args:
            hdf5_path: h5文件路径（需包含input_sst_filled字段）
            normalize: 是否归一化
            mean, std: 归一化参数
            window_size: 时间窗口大小
            preload: 是否预加载数据到内存（推荐True，约12GB内存）
        """
        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.window_size = window_size
        self.preload = preload

        with h5py.File(hdf5_path, 'r') as f:
            # 检查是否有预填充数据
            if 'input_sst_filled' in f.keys():
                self.sst_key = 'input_sst_filled'
            else:
                print("警告: 未找到input_sst_filled，使用input_sst（可能有缺失值）")
                self.sst_key = 'input_sst'

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

            # 预加载数据到内存
            if preload:
                print(f"预加载数据到内存...")
                self.input_sst = f[self.sst_key][:].astype(np.float32)
                self.ground_truth = f['ground_truth_sst'][:].astype(np.float32)
                self.missing_mask = f['missing_mask'][:].astype(np.float32)
                self.land_mask = f['land_mask'][:].astype(np.float32)
                mem_gb = (self.input_sst.nbytes + self.ground_truth.nbytes + self.missing_mask.nbytes) / 1e9
                print(f"  内存占用: ~{mem_gb:.2f} GB")

        print(f"Loaded SST dataset from {hdf5_path}")
        print(f"  Samples: {self.num_samples}, Shape: {self.shape}")
        print(f"  SST key: {self.sst_key}")
        print(f"  Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        print(f"  Preload: {preload}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 计算30天序列的起始索引
        if idx < self.window_size - 1:
            start_idx = 0
            num_valid = idx + 1
        else:
            start_idx = idx - (self.window_size - 1)
            num_valid = self.window_size

        if self.preload:
            # 直接从内存数组切片（极快）
            sst_seq = self.input_sst[start_idx:start_idx + num_valid].copy()
            mask_seq = self.missing_mask[start_idx:start_idx + num_valid].copy()
            gt_sst = self.ground_truth[idx].copy()
            missing_mask = self.missing_mask[idx].copy()
            land_mask = self.land_mask
        else:
            # 从h5文件读取
            with h5py.File(self.hdf5_path, 'r') as f:
                sst_seq = f[self.sst_key][start_idx:start_idx + num_valid].astype(np.float32)
                mask_seq = f['missing_mask'][start_idx:start_idx + num_valid].astype(np.float32)
                gt_sst = f['ground_truth_sst'][idx].astype(np.float32)
                missing_mask = f['missing_mask'][idx].astype(np.float32)
                land_mask = f['land_mask'][:].astype(np.float32)

        # 如果不足30天，用第一天padding
        if num_valid < self.window_size:
            pad_count = self.window_size - num_valid
            sst_pad = np.repeat(sst_seq[0:1], pad_count, axis=0)
            mask_pad = np.repeat(mask_seq[0:1], pad_count, axis=0)
            sst_seq = np.concatenate([sst_pad, sst_seq], axis=0)
            mask_seq = np.concatenate([mask_pad, mask_seq], axis=0)

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
    import time

    # 测试预填充数据
    train_path = '/data/sst_data/sst_missing_value_imputation/processed_data_filled/processed_sst_train.h5'

    print("="*60)
    print("测试 SSTDatasetTemporalFilled")
    print("="*60)

    start = time.time()
    dataset = SSTDatasetTemporalFilled(hdf5_path=train_path, normalize=True, preload=True)
    init_time = time.time() - start
    print(f"\n初始化耗时: {init_time:.1f}秒")

    # 测试__getitem__速度
    print("\n测试__getitem__速度...")
    start = time.time()
    for i in range(1000):
        _ = dataset[i % len(dataset)]
    getitem_time = time.time() - start
    print(f"1000次__getitem__耗时: {getitem_time:.3f}秒 ({getitem_time:.1f}ms/次)")

    # 检查数据
    sample = dataset[100]
    print(f"\n样本检查:")
    print(f"  input_sst_seq shape: {sample['input_sst_seq'].shape}")
    print(f"  input_sst_seq range: [{sample['input_sst_seq'].min():.3f}, {sample['input_sst_seq'].max():.3f}]")

    print("\n" + "="*60)
    print("✓ 测试通过")
