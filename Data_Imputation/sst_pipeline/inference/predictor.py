#!/usr/bin/env python3
"""
SST Pipeline 推理模块
负责执行模型推理
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class Predictor:
    """推理器"""

    def __init__(self, model_wrapper):
        """
        初始化推理器

        Args:
            model_wrapper: ModelWrapper实例
        """
        self.model = model_wrapper

    def predict(self, model_input: torch.Tensor,
                original_sst: np.ndarray,
                land_mask: np.ndarray) -> Dict:
        """
        执行推理

        Args:
            model_input: 模型输入 [1, 60, H, W]
            original_sst: 原始SST数据 [H, W]
            land_mask: 陆地mask [H, W]

        Returns:
            包含预测结果的字典
        """
        # 模型推理
        output = self.model.predict(model_input)  # [1, 1, H, W]
        predicted = output.squeeze().cpu().numpy()  # [H, W]

        # 合成最终结果：观测区域用原始值，缺失区域用预测值
        result = self._compose_output(predicted, original_sst, land_mask)

        return {
            'predicted_raw': predicted,
            'composed': result,
            'land_mask': land_mask
        }

    def _compose_output(self, predicted: np.ndarray,
                        original_sst: np.ndarray,
                        land_mask: np.ndarray) -> np.ndarray:
        """
        合成输出：观测区域保留原始值，缺失区域用预测值填充

        Args:
            predicted: 模型预测结果 [H, W]
            original_sst: 原始SST [H, W]
            land_mask: 陆地mask [H, W]

        Returns:
            合成后的SST [H, W]
        """
        result = original_sst.copy()

        # 缺失区域（非陆地且原始值为NaN）用预测值填充
        missing_mask = np.isnan(original_sst) & (land_mask == 0)
        result[missing_mask] = predicted[missing_mask]

        # 陆地区域保持NaN
        result[land_mask == 1] = np.nan

        return result

    def predict_with_composition(self, model_input: torch.Tensor,
                                  sst_window: np.ndarray,
                                  mask_window: np.ndarray) -> Dict:
        """
        带合成的推理（使用前29天KNN + 第30天预测）

        Args:
            model_input: 模型输入 [1, 60, H, W]
            sst_window: 30天SST数据 [30, H, W]
            mask_window: 30天mask数据 [30, H, W]

        Returns:
            包含预测结果的字典
        """
        # 模型推理
        output = self.model.predict(model_input)  # [1, 1, H, W]
        predicted = output.squeeze().cpu().numpy()  # [H, W]

        # 获取第30天原始数据和陆地mask
        day30_sst = sst_window[-1]
        land_mask = np.isnan(sst_window[0]).astype(np.uint8)  # 第一天的NaN就是陆地

        # 合成结果
        composed = self._compose_output(predicted, day30_sst, land_mask)

        return {
            'predicted_raw': predicted,
            'composed': composed,
            'day30_original': day30_sst,
            'land_mask': land_mask
        }
