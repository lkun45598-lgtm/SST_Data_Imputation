#!/usr/bin/env python3
"""
SST Pipeline 命令行入口
用于运行SST缺失值重建

使用示例:
    # 处理单个日期
    python -m sst_pipeline.run --date 2017-08-08

    # 处理日期范围
    python -m sst_pipeline.run --start-date 2017-08-01 --end-date 2017-08-10

    # 指定参数
    python -m sst_pipeline.run --date 2017-08-08 --sigma 1.5 --no-gaussian
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from .config import PipelineConfig
from .pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='SST Pipeline - JAXA SST缺失值重建',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 日期参数
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--date', type=str, help='处理单个日期 (YYYY-MM-DD)')
    date_group.add_argument('--start-date', type=str, help='起始日期 (与--end-date配合使用)')

    parser.add_argument('--end-date', type=str, help='结束日期')

    # 处理参数
    parser.add_argument('--mask-ratio', type=float, default=0.2, help='人工mask比例 (默认: 0.2)')
    parser.add_argument('--sigma', type=float, default=1.0, help='高斯滤波sigma (默认: 1.0)')
    parser.add_argument('--no-gaussian', action='store_true', help='禁用高斯滤波')
    parser.add_argument('--no-visualize', action='store_true', help='禁用可视化')
    parser.add_argument('--no-save-nc', action='store_true', help='禁用NC文件保存')

    # 路径参数
    parser.add_argument('--model-path', type=str, help='模型权重路径')
    parser.add_argument('--data-dir', type=str, help='KNN填充数据目录')
    parser.add_argument('--output-dir', type=str, help='输出目录')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='运行设备 (默认: cuda)')
    parser.add_argument('--list-dates', action='store_true', help='列出所有可用日期')

    return parser.parse_args()


def get_date_range(start_date: str, end_date: str):
    """生成日期范围"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    return dates


def main():
    args = parse_args()

    # 创建配置
    config = PipelineConfig.default()

    # 更新配置
    if args.model_path:
        config.paths.model_path = Path(args.model_path)
    if args.data_dir:
        config.paths.jaxa_knn_filled_dir = Path(args.data_dir)
    if args.output_dir:
        config.paths.output_dir = Path(args.output_dir)
    if args.device:
        config.model.device = args.device

    # 创建Pipeline
    pipeline = Pipeline(config)

    # 列出可用日期
    if args.list_dates:
        dates = pipeline.get_available_dates()
        print(f"\n可用日期 ({len(dates)} 个):")
        for d in dates[:10]:
            print(f"  {d}")
        if len(dates) > 10:
            print(f"  ... (共 {len(dates)} 个)")
        return

    # 确定处理日期
    if args.date:
        dates = [args.date]
    else:
        if not args.end_date:
            print("错误: --start-date 需要配合 --end-date 使用")
            return
        dates = get_date_range(args.start_date, args.end_date)

    # 处理
    print(f"\n将处理 {len(dates)} 个日期")
    print("=" * 60)

    for date in dates:
        try:
            pipeline.process(
                date=date,
                mask_ratio=args.mask_ratio,
                apply_gaussian=not args.no_gaussian,
                sigma=args.sigma,
                visualize=not args.no_visualize,
                save_nc=not args.no_save_nc
            )
        except Exception as e:
            print(f"处理 {date} 失败: {e}")
            continue

    print("\n处理完成!")


if __name__ == '__main__':
    main()
