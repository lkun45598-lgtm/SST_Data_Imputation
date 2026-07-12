#!/usr/bin/env python3
"""
SST Pipeline 后处理模块
高斯滤波后处理
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional


def apply_gaussian_filter(sst_data: np.ndarray,
                          land_mask: np.ndarray,
                          sigma: float = 1.0,
                          fill_region: np.ndarray = None) -> np.ndarray:
    """
    对SST数据应用高斯滤波

    Args:
        sst_data: SST数据 [H, W]
        land_mask: 陆地mask [H, W]，1表示陆地
        sigma: 高斯滤波sigma值

    Returns:
        滤波后的SST数据 [H, W]
    """
    sst = sst_data.copy()

    # 有效区域：非NaN且非陆地
    mask_valid = ~np.isnan(sst) & (land_mask == 0)

    if mask_valid.sum() == 0:
        return sst

    # 准备滤波数据：无效区域用均值填充
    sst_for_filter = sst.copy()
    mean_val = np.nanmean(sst)
    sst_for_filter[~mask_valid] = mean_val

    # 应用高斯滤波
    filtered = gaussian_filter(sst_for_filter, sigma=sigma)

    # 只在填充区(fill_region)写回滤波值；原始观测保留原值，陆地/无效保持NaN。
    # fill_region=None 时退回旧行为(整个有效区写回)。
    if fill_region is None:
        write = mask_valid
    else:
        write = mask_valid & (fill_region == 1)
    result = np.where(write, filtered, np.where(mask_valid, sst, np.nan))

    return result


class GaussianPostProcessor:
    """高斯后处理器"""

    def __init__(self, sigma: float = 1.0, enabled: bool = True):
        """
        初始化后处理器

        Args:
            sigma: 高斯滤波sigma值
            enabled: 是否启用后处理
        """
        self.sigma = sigma
        self.enabled = enabled

    def process(self, sst_data: np.ndarray, land_mask: np.ndarray,
                fill_region: np.ndarray = None) -> np.ndarray:
        """
        执行后处理

        Args:
            sst_data: SST数据 [H, W]
            land_mask: 陆地mask [H, W]
            fill_region: 允许平滑的填充区 [H, W] (1=模型填充区)。只在该区写回滤波值，
                         原始观测保留原值。None 时退回整场平滑(旧行为)。

        Returns:
            处理后的SST数据 [H, W]
        """
        if not self.enabled:
            return sst_data

        return apply_gaussian_filter(sst_data, land_mask, self.sigma,
                                     fill_region=fill_region)

    def __call__(self, sst_data: np.ndarray, land_mask: np.ndarray,
                 fill_region: np.ndarray = None) -> np.ndarray:
        """支持直接调用"""
        return self.process(sst_data, land_mask, fill_region=fill_region)


def create_postprocessor(config) -> GaussianPostProcessor:
    """从配置创建后处理器的工厂函数"""
    return GaussianPostProcessor(
        sigma=config.postprocess.gaussian_sigma,
        enabled=config.postprocess.apply_gaussian
    )
