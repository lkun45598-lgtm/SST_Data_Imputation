"""
FNO-CBAM SST重建可视化 - 4连图
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from pathlib import Path
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
import sys
import imageio

# 添加项目根目录到路径
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from datasets.ostia_dataset import SSTDatasetTemporal
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

warnings.filterwarnings('ignore')


def setup_matplotlib():
    """配置matplotlib高质量绘图"""
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1.5, labelsize=14)
    plt.rc('lines', linewidth=1.5)

    params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
    }
    plt.rcParams.update(params)


def format_lon(x, pos):
    """格式化经度标签"""
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    """格式化纬度标签"""
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6,
        cbam_reduction_ratio=16
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  验证MAE: {checkpoint['val_mae']:.3f}°C\n")

    return model


def get_sample_date(hdf5_path, sample_idx):
    """获取样本对应的日期"""
    base_date = datetime(2015, 1, 1)
    sample_date = base_date + timedelta(days=sample_idx)
    return sample_date.strftime('%Y-%m-%d')


def create_four_panel_plot(input_sst, mask, ground_truth, prediction,
                          land_mask, lon_coords, lat_coords,
                          sample_date, save_path,
                          mean=26.71, std=2.69):
    """
    创建4连图（去掉Mask图）

    Args:
        input_sst: (H, W) 第30天有观测+mask的SST (归一化)
        mask: (H, W) 第30天的mask (1=缺失, 0=观测)
        ground_truth: (H, W) 第30天完整真值 (归一化)
        prediction: (H, W) 模型预测 (归一化)
        land_mask: (H, W) 陆地mask (1=陆地, 0=海洋)
        lon_coords: (W,) 经度坐标
        lat_coords: (H,) 纬度坐标
        sample_date: str 样本日期
        save_path: str 保存路径
        mean, std: 归一化参数
    """
    # 反归一化到开尔文，然后转换为摄氏度
    input_kelvin = input_sst * std + mean
    gt_kelvin = ground_truth * std + mean
    pred_kelvin = prediction * std + mean

    # 转换为摄氏度
    input_celsius = input_kelvin - 273.15
    gt_celsius = gt_kelvin - 273.15
    pred_celsius = pred_kelvin - 273.15

    # 计算绝对误差
    abs_error = np.abs(pred_celsius - gt_celsius)

    # 海洋mask
    ocean_mask = 1 - land_mask
    missing_ocean = mask * ocean_mask

    # 统计（只在缺失海洋区域的误差）
    mae = np.mean(abs_error[missing_ocean > 0])
    rmse = np.sqrt(np.mean((abs_error[missing_ocean > 0])**2))
    max_error = np.max(abs_error[missing_ocean > 0])
    missing_ratio = np.sum(missing_ocean) / np.sum(ocean_mask) * 100

    # 创建网格坐标
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # 创建图形 - 4连图布局 [Input, GT, Pred+cbar, Error+cbar]
    fig = plt.figure(figsize=(28, 7))

    # 定义布局：4个主图，只有第3和第4图有colorbar
    # [图1, 图2, 图3, cbar3, 图4, cbar4]
    gs = gridspec.GridSpec(1, 6, figure=fig,
                          width_ratios=[1, 1, 1, 0.08, 1, 0.08],
                          wspace=0.15, hspace=0.1,
                          left=0.04, right=0.98, top=0.85, bottom=0.15)

    # 创建4个子图轴
    ax_input = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_error = fig.add_subplot(gs[0, 4])

    # 设置colormap和范围
    vmin_sst = np.percentile(gt_celsius[ocean_mask > 0], 2)
    vmax_sst = np.percentile(gt_celsius[ocean_mask > 0], 98)
    cmap_sst = 'RdYlBu_r'
    cmap_error = 'hot_r'

    # 准备数据
    input_display = input_celsius.copy()
    input_display[mask > 0] = np.nan  # 缺失区域设为NaN
    input_masked = np.ma.masked_where(land_mask > 0, input_display)

    gt_masked = np.ma.masked_where(land_mask > 0, gt_celsius)
    pred_masked = np.ma.masked_where(land_mask > 0, pred_celsius)

    # 陆地颜色
    from matplotlib.colors import ListedColormap
    land_color = '#D2B48C'  # 浅棕色

    # ===== 1. 输入SST（有观测+云层）=====
    ax = ax_input
    # 云层区域用浅蓝色
    ax.set_facecolor('skyblue')

    # 先画陆地（棕色）
    land_display = np.ma.masked_where(land_mask == 0, land_mask)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # 再画SST数据
    im0 = ax.pcolormesh(lon_grid, lat_grid, input_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst,
                        shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    # 减少刻度数量避免重叠
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title(f'Input SST (Day 7)\n{missing_ratio:.1f}% Missing',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)

    # ===== 2. Ground Truth（完整无遮挡）=====
    ax = ax_gt
    ax.set_facecolor('lightgray')

    # 先画陆地（棕色）
    land_display2 = np.ma.masked_where(land_mask == 0, land_mask)
    ax.pcolormesh(lon_grid, lat_grid, land_display2,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    im1 = ax.pcolormesh(lon_grid, lat_grid, gt_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst,
                        shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Ground Truth SST\n(Complete, No Missing)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 3. FNO-CBAM Reconstruction（显示colorbar）=====
    ax = ax_pred
    ax.set_facecolor('lightgray')

    # 先画陆地（棕色）
    ax.pcolormesh(lon_grid, lat_grid, land_display2,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    im2 = ax.pcolormesh(lon_grid, lat_grid, pred_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst,
                        shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('FNO-CBAM Reconstruction\n(Model Output)',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # Colorbar for FNO-CBAM (共用SST colorbar)
    cax2_container = fig.add_subplot(gs[0, 3])
    cax2_container.axis('off')
    cax2 = inset_axes(cax2_container,
                      width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1),
                      bbox_transform=cax2_container.transAxes)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('SST (°C)', fontsize=11)

    # ===== 4. 绝对误差图（只显示缺失区域的误差）=====
    ax = ax_error
    ax.set_facecolor('white')  # 观测海洋区域为白色

    # 先画陆地（棕色）
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # 再画缺失区域的误差
    error_display = abs_error.copy()
    error_display[mask == 0] = np.nan  # 观测区域设为NaN
    error_masked = np.ma.masked_where((land_mask > 0) | (mask == 0), error_display)

    im3 = ax.pcolormesh(lon_grid, lat_grid, error_masked,
                        cmap=cmap_error, vmin=0, vmax=0.5, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Absolute Error (Missing Region Only)\n|Reconstruction - Ground Truth|',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # 添加统计信息到第4张图
    stats_text = f'MAE: {mae:.3f}°C\nRMSE: {rmse:.3f}°C\nMax: {max_error:.3f}°C'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Colorbar for Error
    cax3_container = fig.add_subplot(gs[0, 5])
    cax3_container.axis('off')
    cax3 = inset_axes(cax3_container,
                      width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1),
                      bbox_transform=cax3_container.transAxes)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('|Error| (°C)', fontsize=11)

    # ===== 添加日期标签（在图形最顶部）=====
    fig.text(0.5, 0.98, f'Date: {sample_date}',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return mae, rmse, max_error


def main():
    print("="*60)
    print("FNO-CBAM SST重建可视化 - 4连图")
    print("="*60)

    # 配置
    setup_matplotlib()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = '/home/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/temporal_30days_composition_fast/best_model.pth'
    data_dir = '/data/sst_data/sst_missing_value_imputation/processed_data'
    output_dir = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visualization_4panel')
    output_dir.mkdir(exist_ok=True)

    print(f"\n使用设备: {device}")

    # 加载模型
    model = load_model(checkpoint_path, device)

    # 加载经纬度坐标
    with h5py.File(f'{data_dir}/processed_sst_valid.h5', 'r') as f:
        lon_coords = f['longitude'][:]
        lat_coords = f['latitude'][:]
        print(f"经度范围: {lon_coords.min():.2f}° ~ {lon_coords.max():.2f}°")
        print(f"纬度范围: {lat_coords.min():.2f}° ~ {lat_coords.max():.2f}°\n")

    # 加载数据集
    print("加载数据集...")
    train_dataset = SSTDatasetTemporal(
        hdf5_path=f'{data_dir}/processed_sst_train.h5',
        normalize=True
    )
    valid_dataset = SSTDatasetTemporal(
        hdf5_path=f'{data_dir}/processed_sst_valid.h5',
        normalize=True,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    print(f"验证集样本数: {len(valid_dataset)}\n")

    # 可视化样本 - 连续20天
    num_samples = 20
    sample_indices = list(range(num_samples))
    all_mae, all_rmse, all_max = [], [], []
    image_paths = []

    print(f"开始推理和可视化（连续{num_samples}天）...")
    for idx in tqdm(sample_indices, desc="处理样本"):
        if idx >= len(valid_dataset):
            continue

        # 加载样本
        sample = valid_dataset[idx]
        sst_seq = torch.from_numpy(sample['input_sst_seq']).unsqueeze(0).to(device)
        mask_seq = torch.from_numpy(sample['mask_seq']).unsqueeze(0).to(device)
        gt_sst = torch.from_numpy(sample['ground_truth_sst']).unsqueeze(0).unsqueeze(0).to(device)
        land_mask = torch.from_numpy(sample['land_mask']).to(device)

        # 推理
        with torch.no_grad():
            pred = model(sst_seq, mask_seq)

            # Output Composition: 非缺失区域用输入，缺失区域用预测
            last_input = sst_seq[:, -1:, :, :]  # (B, 1, H, W)
            last_mask = mask_seq[:, -1:, :, :]  # (B, 1, H, W)
            pred = last_input * (1 - last_mask) + pred * last_mask

        # 转numpy
        input_sst = sst_seq[0, -1].cpu().numpy()
        mask = mask_seq[0, -1].cpu().numpy()
        gt = gt_sst[0, 0].cpu().numpy()
        prediction = pred[0, 0].cpu().numpy()
        land = land_mask.cpu().numpy()

        # 获取日期
        sample_date = get_sample_date(f'{data_dir}/processed_sst_valid.h5', idx)

        # 绘图
        save_path = output_dir / f'reconstruction_4panel_{sample_date}_sample{idx:04d}.png'
        mae, rmse, max_err = create_four_panel_plot(
            input_sst, mask, gt, prediction, land,
            lon_coords, lat_coords,
            sample_date, save_path,
            mean=valid_dataset.mean,
            std=valid_dataset.std
        )

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_max.append(max_err)
        image_paths.append(str(save_path))

        print(f"  {sample_date} (样本{idx}): MAE={mae:.3f}°C, RMSE={rmse:.3f}°C, Max={max_err:.3f}°C")

    # 总体统计
    print("\n" + "="*60)
    print("总体统计 (选取样本)")
    print("="*60)
    print(f"平均MAE:  {np.mean(all_mae):.3f}°C")
    print(f"平均RMSE: {np.mean(all_rmse):.3f}°C")
    print(f"平均Max:  {np.mean(all_max):.3f}°C")
    print(f"最大Max:  {np.max(all_max):.3f}°C")
    print(f"\n可视化结果保存在: {output_dir}")

    # 生成GIF动画
    print("\n生成GIF动画...")
    gif_path = output_dir / 'reconstruction_4panel_animation.gif'
    # 使用fps参数：fps=0.5表示每秒0.5帧，即2秒显示1帧
    with imageio.get_writer(gif_path, mode='I', fps=0.5) as writer:
        for img_path in image_paths:
            image = imageio.imread(img_path)
            writer.append_data(image)
    print(f"GIF动画保存在: {gif_path}")
    print(f"总时长: {len(image_paths) * 2.0:.0f}秒 ({len(image_paths)}帧 × 2秒/帧)")


if __name__ == '__main__':
    main()
