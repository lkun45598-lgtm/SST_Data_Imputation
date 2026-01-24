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
                          sigma: float = 1.0) -> np.ndarray:
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

    # 只在有效区域使用滤波结果
    result = np.where(mask_valid, filtered, np.nan)

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

    def process(self, sst_data: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
        """
        执行后处理

        Args:
            sst_data: SST数据 [H, W]
            land_mask: 陆地mask [H, W]

        Returns:
            处理后的SST数据 [H, W]
        """
        if not self.enabled:
            return sst_data

        return apply_gaussian_filter(sst_data, land_mask, self.sigma)

    def __call__(self, sst_data: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
        """支持直接调用"""
        return self.process(sst_data, land_mask)


def create_postprocessor(config) -> GaussianPostProcessor:
    """从配置创建后处理器的工厂函数"""
    return GaussianPostProcessor(
        sigma=config.postprocess.gaussian_sigma,
        enabled=config.postprocess.apply_gaussian
    )
