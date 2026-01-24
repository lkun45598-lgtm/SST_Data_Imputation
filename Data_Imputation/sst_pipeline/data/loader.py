#!/usr/bin/env python3
"""
SST Pipeline 数据加载模块
负责加载预处理后的H5数据
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
import torch


class JaxaDataLoader:
    """JAXA数据加载器"""

    def __init__(self, knn_filled_dir: Path, window_size: int = 30):
        """
        初始化数据加载器

        Args:
            knn_filled_dir: KNN填充后的H5文件目录
            window_size: 时间窗口大小（天数）
        """
        self.knn_filled_dir = Path(knn_filled_dir)
        self.window_size = window_size
        self._file_index = None
        self._build_file_index()

    def _build_file_index(self):
        """构建日期到文件的索引"""
        self._file_index = {}
        h5_files = sorted(self.knn_filled_dir.glob('*.h5'))
        for f in h5_files:
            # 文件名格式: jaxa_knn_filled_YYYYMMDD.h5
            date_str = f.stem.split('_')[-1]
            try:
                date = datetime.strptime(date_str, '%Y%m%d')
                self._file_index[date.strftime('%Y-%m-%d')] = f
            except ValueError:
                continue
        print(f"[DataLoader] 已索引 {len(self._file_index)} 个H5文件")

    def get_available_dates(self) -> List[str]:
        """获取所有可用日期"""
        return sorted(self._file_index.keys())

    def load_single_day(self, date: str) -> Dict:
        """
        加载单天数据

        Args:
            date: 日期字符串 'YYYY-MM-DD'

        Returns:
            包含SST和mask的字典
        """
        if date not in self._file_index:
            raise ValueError(f"日期 {date} 不在可用数据范围内")

        h5_path = self._file_index[date]
        with h5py.File(h5_path, 'r') as f:
            sst = f['sst_knn_filled'][:]
            mask = f['missing_mask'][:] if 'missing_mask' in f else np.zeros_like(sst)
            lat = f['lat'][:] if 'lat' in f else None
            lon = f['lon'][:] if 'lon' in f else None

        return {
            'sst': sst,
            'mask': mask,
            'lat': lat,
            'lon': lon,
            'date': date
        }

    def load_window(self, target_date: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        加载目标日期的30天窗口数据

        Args:
            target_date: 目标日期 'YYYY-MM-DD'

        Returns:
            sst_window: [T, H, W] 30天SST数据
            mask_window: [T, H, W] 30天mask数据
            metadata: 元数据字典
        """
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')

        # 获取前29天 + 目标日期
        dates = []
        for i in range(self.window_size - 1, -1, -1):
            dt = target_dt - timedelta(days=i)
            dates.append(dt.strftime('%Y-%m-%d'))

        # 检查所有日期是否可用
        missing_dates = [d for d in dates if d not in self._file_index]
        if missing_dates:
            raise ValueError(f"缺少数据: {missing_dates}")

        # 加载数据
        sst_list = []
        mask_list = []
        lat, lon = None, None

        for date in dates:
            data = self.load_single_day(date)
            sst_list.append(data['sst'])
            mask_list.append(data['mask'])
            if lat is None:
                lat = data['lat']
                lon = data['lon']

        sst_window = np.stack(sst_list, axis=0)  # [T, H, W]
        mask_window = np.stack(mask_list, axis=0)  # [T, H, W]

        metadata = {
            'dates': dates,
            'target_date': target_date,
            'lat': lat,
            'lon': lon,
            'shape': sst_window.shape
        }

        return sst_window, mask_window, metadata

    def prepare_model_input(self, sst_window: np.ndarray, mask_window: np.ndarray,
                           mask_ratio: float = 0.2,
                           mask_day_30: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
        """
        准备模型输入

        Args:
            sst_window: [T, H, W] SST数据
            mask_window: [T, H, W] mask数据
            mask_ratio: 第30天的mask比例
            mask_day_30: 是否对第30天进行mask

        Returns:
            model_input: [1, 60, H, W] 模型输入tensor
            day30_mask: 第30天的mask
        """
        T, H, W = sst_window.shape

        # 复制以避免修改原始数据
        sst = sst_window.copy()
        mask = mask_window.copy()

        # 第30天处理
        day30_sst = sst[-1].copy()
        day30_mask = mask[-1].copy()

        if mask_day_30:
            # 在观测区域进行人工mask
            valid_mask = ~np.isnan(day30_sst) & (day30_mask == 0)
            valid_indices = np.where(valid_mask)
            n_valid = len(valid_indices[0])

            if n_valid > 0:
                n_mask = int(n_valid * mask_ratio)
                mask_indices = np.random.choice(n_valid, n_mask, replace=False)
                for idx in mask_indices:
                    i, j = valid_indices[0][idx], valid_indices[1][idx]
                    day30_mask[i, j] = 1

        # 对第30天使用均值填充mask区域
        mean_val = np.nanmean(day30_sst)
        day30_sst_filled = day30_sst.copy()
        day30_sst_filled[day30_mask == 1] = mean_val
        day30_sst_filled[np.isnan(day30_sst_filled)] = mean_val

        sst[-1] = day30_sst_filled

        # 处理NaN
        sst = np.nan_to_num(sst, nan=mean_val)

        # 构建模型输入: [30 SST channels, 30 mask channels]
        sst_tensor = torch.from_numpy(sst).float()  # [T, H, W]
        mask_tensor = torch.from_numpy(mask).float()  # [T, H, W]

        # 合并为60通道
        model_input = torch.cat([sst_tensor, mask_tensor], dim=0)  # [60, H, W]
        model_input = model_input.unsqueeze(0)  # [1, 60, H, W]

        return model_input, day30_mask


def create_data_loader(config) -> JaxaDataLoader:
    """创建数据加载器的工厂函数"""
    return JaxaDataLoader(
        knn_filled_dir=config.paths.jaxa_knn_filled_dir,
        window_size=config.preprocess.window_size
    )
