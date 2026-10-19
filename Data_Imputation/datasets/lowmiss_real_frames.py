#!/usr/bin/env python3
"""
低缺失率 JAXA 帧数据集: 作为 GAN 判别器的真样本

只用海洋像素计算观测覆盖率 (陆地占约 17%，不能算进缺失)。
返回整帧 SST 和"真实观测海洋"掩码；填充值所在位置在掩码里为 0，
判别器通过块有效性把这些位置排除。
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class LowMissRealFrames(Dataset):
    def __init__(self, data_dir, series_ids, min_ocean_cov=0.7, mean=299.9221, std=2.6919,
                 month_balance=False):
        self.mean = mean
        self.std = std
        self.items = []  # (h5_path, frame_idx, month, cov)

        for sid in series_ids:
            h5_path = Path(data_dir) / f'jaxa_knn_filled_{sid:02d}.h5'
            if not h5_path.exists():
                print(f"Warning: {h5_path} not found, skipping...")
                continue
            with h5py.File(h5_path, 'r') as f:
                ocean = f['land_mask'][:] == 0
                obs = f['original_obs_mask'][:]
                cov = obs[:, ocean].mean(axis=1)
                ts = [t.decode() if isinstance(t, bytes) else str(t) for t in f['timestamps'][:]]
            for i in np.where(cov >= min_ocean_cov)[0]:
                self.items.append((str(h5_path), int(i), int(ts[i][5:7]), float(cov[i])))

        months = np.array([it[2] for it in self.items])
        counts = {m: (months == m).sum() for m in set(months.tolist())}
        # 默认按帧均匀抽样 (有多少用多少); month_balance=True 时按月份均衡，
        # 用来缓解低缺失率帧集中在晴朗月份的偏差
        if month_balance:
            self.weights = [1.0 / counts[it[2]] for it in self.items]
        else:
            self.weights = [1.0] * len(self.items)

        print(f"LowMissRealFrames: {len(self.items)} frames (ocean cov >= {min_ocean_cov}), "
              f"series {list(series_ids)}, per-month {dict(sorted(counts.items()))}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        h5_path, i, _, _ = self.items[idx]
        with h5py.File(h5_path, 'r') as f:
            sst = f['sst_data'][i].astype(np.float32)
            obs = f['original_obs_mask'][i].astype(np.float32)
            land = f['land_mask'][:].astype(np.float32)
        valid = obs * (1.0 - land)
        sst = np.nan_to_num((sst - self.mean) / self.std, nan=0.0)
        return {
            'sst': torch.from_numpy(sst).unsqueeze(0),
            'valid': torch.from_numpy(valid).unsqueeze(0),
        }
