#!/usr/bin/env python3
"""
SST Pipeline 模型封装模块
封装FNO-CBAM模型的加载和推理
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal


class ModelWrapper:
    """FNO-CBAM模型封装器"""

    def __init__(self, model_path: Path, device: str = 'cuda',
                 out_size: tuple = (451, 351),
                 modes1: int = 80, modes2: int = 64,
                 width: int = 64, depth: int = 6):
        """
        初始化模型封装器

        Args:
            model_path: 模型权重路径
            device: 运行设备
            out_size: 输出尺寸
            modes1: FNO模式数1
            modes2: FNO模式数2
            width: 网络宽度
            depth: 网络深度
        """
        self.model_path = Path(model_path)
        self.device = device

        # 创建模型
        self.model = FNO_CBAM_SST_Temporal(
            out_size=out_size,
            modes1=modes1,
            modes2=modes2,
            width=width,
            depth=depth
        )

        # 加载权重
        self._load_weights()

        # 移动到设备并设为评估模式
        self.model = self.model.to(device)
        self.model.eval()

        print(f"[Model] 模型已加载: {self.model_path.name}")
        print(f"[Model] 设备: {device}")

    def _load_weights(self):
        """加载模型权重"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location='cpu')

        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除 'module.' 前缀 (如果是DataParallel保存的)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型推理

        Args:
            x: 输入tensor [B, 60, H, W]

        Returns:
            输出tensor [B, 1, H, W]
        """
        x = x.to(self.device)
        output = self.model(x)
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """支持直接调用"""
        return self.predict(x)


def load_model(config) -> ModelWrapper:
    """从配置加载模型的工厂函数"""
    return ModelWrapper(
        model_path=config.paths.model_path,
        device=config.model.device,
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6
    )
