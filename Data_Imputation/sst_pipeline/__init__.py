"""
SST Pipeline 模块
用于JAXA SST缺失值重建的统一Pipeline

使用示例:
    from sst_pipeline import Pipeline

    # 使用默认配置
    pipeline = Pipeline()

    # 处理单个日期
    result = pipeline.process(
        date="2017-08-08",
        apply_gaussian=True,
        sigma=1.0,
        visualize=True
    )

    # 查看可用日期
    dates = pipeline.get_available_dates()
"""

from .config import PipelineConfig, PathConfig, ModelConfig, PreprocessConfig, PostprocessConfig, VisualizationConfig
from .pipeline import Pipeline

__all__ = [
    'Pipeline',
    'PipelineConfig',
    'PathConfig',
    'ModelConfig',
    'PreprocessConfig',
    'PostprocessConfig',
    'VisualizationConfig'
]

__version__ = '1.0.0'
