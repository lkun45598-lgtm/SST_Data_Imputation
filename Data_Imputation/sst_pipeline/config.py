#!/usr/bin/env python3
"""
SST Pipeline 配置文件
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PathConfig:
    """数据路径配置"""
    # 项目根目录
    project_root: Path = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM')

    # JAXA原始数据目录 (hourly NC files)
    jaxa_raw_dir: Path = Path('/data1/user/lz/FNO_CBAM/data/jaxa')

    # 预处理中间数据目录
    jaxa_weighted_dir: Path = field(default_factory=lambda: Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_weighted_aligned'))
    jaxa_filtered_dir: Path = field(default_factory=lambda: Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_filtered'))
    jaxa_knn_filled_dir: Path = field(default_factory=lambda: Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled'))

    # 模型权重路径
    model_path: Path = field(default_factory=lambda: Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_8years/best_model.pth'))

    # 输出目录
    output_dir: Path = field(default_factory=lambda: Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/pipeline_output'))

    def __post_init__(self):
        # 转换为Path对象
        for attr_name in ['project_root', 'jaxa_raw_dir', 'jaxa_weighted_dir',
                          'jaxa_filtered_dir', 'jaxa_knn_filled_dir',
                          'model_path', 'output_dir']:
            val = getattr(self, attr_name)
            if isinstance(val, str):
                setattr(self, attr_name, Path(val))


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型架构参数
    in_channels: int = 60  # 30 SST + 30 mask
    out_channels: int = 1
    modes: int = 32
    width: int = 64
    num_layers: int = 4
    temporal_channels: int = 30

    # 推理参数
    device: str = 'cuda'
    batch_size: int = 1


@dataclass
class PreprocessConfig:
    """预处理配置"""
    # 时间窗口
    window_size: int = 30  # 30天窗口

    # 低通滤波参数
    cutoff_frequency: float = 0.1

    # KNN填充参数
    knn_neighbors: int = 5
    knn_max_iterations: int = 100

    # 目标区域 (与OSTIA对齐)
    lat_range: tuple = (20.025, 49.975)  # 0.05度分辨率
    lon_range: tuple = (115.025, 149.975)


@dataclass
class PostprocessConfig:
    """后处理配置"""
    # 高斯滤波
    apply_gaussian: bool = True
    gaussian_sigma: float = 1.0

    # 输出格式
    save_nc: bool = True
    save_h5: bool = False


@dataclass
class VisualizationConfig:
    """可视化配置"""
    enabled: bool = True
    dpi: int = 300
    figsize: tuple = (20, 5)
    cmap: str = 'RdYlBu_r'
    land_color: str = '#D2B48C'


@dataclass
class PipelineConfig:
    """Pipeline总配置"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def default(cls) -> 'PipelineConfig':
        """返回默认配置"""
        return cls()

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if '.' in key:
                # 支持 'postprocess.gaussian_sigma' 形式
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            elif hasattr(self, key):
                setattr(self, key, value)
