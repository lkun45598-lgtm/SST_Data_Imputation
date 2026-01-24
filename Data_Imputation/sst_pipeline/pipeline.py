#!/usr/bin/env python3
"""
SST Pipeline 主模块
整合数据加载、模型推理、后处理和可视化
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union
import json

from .config import PipelineConfig
from .data.loader import JaxaDataLoader, create_data_loader
from .model.wrapper import ModelWrapper, load_model
from .inference.predictor import Predictor
from .postprocess.gaussian import GaussianPostProcessor, create_postprocessor
from .visualization.plotter import Plotter, create_plotter


class Pipeline:
    """SST重建Pipeline"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        初始化Pipeline

        Args:
            config: Pipeline配置，为None时使用默认配置
        """
        self.config = config or PipelineConfig.default()
        self._init_components()

    def _init_components(self):
        """初始化各组件"""
        print("=" * 60)
        print("SST Pipeline 初始化")
        print("=" * 60)

        # 数据加载器
        self.data_loader = create_data_loader(self.config)

        # 模型
        self.model = load_model(self.config)

        # 推理器
        self.predictor = Predictor(self.model)

        # 后处理器
        self.postprocessor = create_postprocessor(self.config)

        # 可视化器
        self.plotter = create_plotter(self.config)

        print("=" * 60)
        print("Pipeline 初始化完成")
        print("=" * 60)

    def process(self, date: str,
                mask_ratio: float = 0.2,
                apply_gaussian: bool = True,
                sigma: float = 1.0,
                visualize: bool = True,
                save_nc: bool = True) -> Dict:
        """
        处理单个日期的SST重建

        Args:
            date: 目标日期 'YYYY-MM-DD'
            mask_ratio: 人工mask比例
            apply_gaussian: 是否应用高斯滤波
            sigma: 高斯滤波sigma值
            visualize: 是否生成可视化
            save_nc: 是否保存NC文件

        Returns:
            包含处理结果的字典
        """
        print(f"\n处理日期: {date}")
        print("-" * 40)

        # 1. 加载30天窗口数据
        print("[1/5] 加载数据...")
        sst_window, mask_window, metadata = self.data_loader.load_window(date)
        print(f"      数据形状: {sst_window.shape}")

        # 2. 准备模型输入
        print("[2/5] 准备模型输入...")
        model_input, day30_mask = self.data_loader.prepare_model_input(
            sst_window, mask_window, mask_ratio=mask_ratio
        )
        print(f"      输入形状: {model_input.shape}")

        # 3. 模型推理
        print("[3/5] 模型推理...")
        prediction_result = self.predictor.predict_with_composition(
            model_input, sst_window, mask_window
        )

        # 4. 后处理
        print("[4/5] 后处理...")
        if apply_gaussian:
            self.postprocessor.sigma = sigma
            self.postprocessor.enabled = True
        else:
            self.postprocessor.enabled = False

        composed = prediction_result['composed']
        land_mask = prediction_result['land_mask']
        gaussian_filtered = self.postprocessor(composed, land_mask)

        # 组装结果
        result = {
            'knn_filled': sst_window[-1],  # 第30天KNN填充结果
            'day30_input': model_input[0, 29].cpu().numpy(),  # 第30天模型输入
            'predicted_raw': prediction_result['predicted_raw'],
            'composed': composed,
            'gaussian_filtered': gaussian_filtered if apply_gaussian else None,
            'land_mask': land_mask,
            'day30_mask': day30_mask
        }

        # 5. 保存和可视化
        print("[5/5] 保存结果...")
        output_dir = self.config.paths.output_dir / date.replace('-', '')
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_nc:
            nc_path = output_dir / f'sst_reconstruction_{date}.nc'
            self._save_nc(result, metadata, nc_path, apply_gaussian, sigma)

        if visualize:
            viz_path = output_dir / f'sst_reconstruction_{date}.png'
            self.plotter.plot_reconstruction(result, metadata, viz_path)

        # 计算指标
        metrics = self._compute_metrics(result, sst_window[-1])
        result['metrics'] = metrics

        # 保存指标
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n结果已保存到: {output_dir}")
        print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, VRMSE: {metrics['vrmse']:.4f}")

        return result

    def _save_nc(self, result: Dict, metadata: Dict, output_path: Path,
                 apply_gaussian: bool, sigma: float):
        """保存结果为NC文件"""
        import netCDF4 as nc

        with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
            lat = metadata.get('lat')
            lon = metadata.get('lon')
            H, W = result['composed'].shape

            # 创建维度
            f.createDimension('lat', H)
            f.createDimension('lon', W)

            # 经纬度变量
            if lat is not None:
                lat_var = f.createVariable('lat', 'f4', ('lat',))
                lat_var[:] = lat
            if lon is not None:
                lon_var = f.createVariable('lon', 'f4', ('lon',))
                lon_var[:] = lon

            # SST变量
            sst_var = f.createVariable('sst_reconstructed', 'f4', ('lat', 'lon'))
            final_output = result['gaussian_filtered'] if result['gaussian_filtered'] is not None else result['composed']
            sst_var[:] = final_output

            # 属性
            f.date = metadata.get('target_date', '')
            f.gaussian_filter = str(apply_gaussian)
            f.gaussian_sigma = sigma

        print(f"      NC文件已保存: {output_path}")

    def _compute_metrics(self, result: Dict, ground_truth: np.ndarray) -> Dict:
        """计算评估指标"""
        final_output = result['gaussian_filtered'] if result['gaussian_filtered'] is not None else result['composed']
        land_mask = result['land_mask']

        # 有效区域：非陆地且都有值
        valid = (land_mask == 0) & ~np.isnan(final_output) & ~np.isnan(ground_truth)

        if valid.sum() == 0:
            return {'rmse': float('nan'), 'mae': float('nan'), 'vrmse': float('nan')}

        pred = final_output[valid]
        gt = ground_truth[valid]

        # RMSE
        rmse = np.sqrt(np.mean((pred - gt) ** 2))

        # MAE
        mae = np.mean(np.abs(pred - gt))

        # VRMSE (相对于标准差)
        gt_std = np.std(gt)
        vrmse = rmse / gt_std if gt_std > 0 else float('nan')

        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'vrmse': float(vrmse),
            'gt_std': float(gt_std),
            'num_valid_pixels': int(valid.sum())
        }

    def get_available_dates(self):
        """获取所有可用的处理日期"""
        return self.data_loader.get_available_dates()
