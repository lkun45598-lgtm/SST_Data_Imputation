#!/usr/bin/env python3
"""
从填充好的 SST 数据集提取某一点的一整年时序，画折线图。
默认点: (19°N, 115°E)，2024 年全年。

保存: /data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/dot_jaxa/
"""

import argparse
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

DATA_ROOT = Path('/data/sst_data/SST_Data_Imputation')
OUT_DIR   = Path('/data1/user/lz/SST_Data_Imputation/Data_Imputation/experiments/dot_jaxa')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, default=19.0)
    parser.add_argument('--lon', type=float, default=115.0)
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--month', type=int, default=7,
                        help='月度图和周图所用的月份 (1-12)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取一次确定坐标索引
    sample_file = next(DATA_ROOT.rglob(f'{args.year}*.nc'), None)
    if sample_file is None:
        raise FileNotFoundError(f'{args.year} 年无数据')
    with nc.Dataset(str(sample_file)) as ds:
        lat_arr = ds['lat'][:]
        lon_arr = ds['lon'][:]
    i = int(np.argmin(np.abs(lat_arr - args.lat)))
    j = int(np.argmin(np.abs(lon_arr - args.lon)))
    real_lat = float(lat_arr[i])
    real_lon = float(lon_arr[j])
    print(f'目标点: ({args.lat}°N, {args.lon}°E) → 索引 ({i},{j})')
    print(f'实际坐标: ({real_lat:.3f}°N, {real_lon:.3f}°E)')

    # 遍历整年 24 小时
    day = datetime(args.year, 1, 1)
    end = datetime(args.year + 1, 1, 1)

    times, sst_values = [], []
    total_hours = (end - day).days * 24

    with tqdm(total=total_hours, desc=f'扫描 {args.year}') as pbar:
        while day < end:
            for h in range(24):
                dt = day.replace(hour=h)
                fname = dt.strftime('%Y%m%d%H%M%S') + '.nc'
                fpath = DATA_ROOT / dt.strftime('%Y%m') / dt.strftime('%d') / fname
                if fpath.exists():
                    with nc.Dataset(str(fpath)) as ds:
                        val = float(ds['sea_surface_temperature'][0, i, j])
                    times.append(dt)
                    sst_values.append(val - 273.15)  # Kelvin → Celsius
                pbar.update(1)
            day += timedelta(days=1)

    if not sst_values:
        print('没有有效数据')
        return

    times = np.array(times)
    sst_values = np.array(sst_values)
    valid = ~np.isnan(sst_values)
    print(f'共读取 {len(times)} 帧，有效 {valid.sum()} 帧')
    print(f'SST 范围: {np.nanmin(sst_values):.2f} ~ {np.nanmax(sst_values):.2f}°C')

    plt.rc('font', size=12)

    # 日均序列（整年用）
    t_sec = np.array([t.timestamp() for t in times])
    daily_avg, daily_t = [], []
    for d in range((end - datetime(args.year, 1, 1)).days):
        d0 = datetime(args.year, 1, 1) + timedelta(days=d)
        d1 = d0 + timedelta(days=1)
        m = (t_sec >= d0.timestamp()) & (t_sec < d1.timestamp())
        if m.any():
            avg = np.nanmean(sst_values[m])
            if not np.isnan(avg):
                daily_avg.append(avg)
                daily_t.append(d0 + timedelta(hours=12))

    # ── 通用绘图函数 ─────────────────────────────────────────────
    def plot_range(t0, t1, title_suffix, save_name,
                   date_loc, date_fmt, show_daily=True, figsize=(16, 5)):
        mask = (times >= t0) & (times < t1)
        sub_t   = times[mask]
        sub_sst = sst_values[mask]
        if len(sub_t) == 0:
            print(f'  无数据，跳过 {save_name}')
            return

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(sub_t, sub_sst, color='#1f77b4', linewidth=0.8, alpha=0.7,
                marker='o' if len(sub_t) < 300 else None, markersize=2,
                label='Hourly SST')

        if show_daily:
            sub_daily_t   = [t for t in daily_t if t0 <= t < t1]
            sub_daily_avg = [daily_avg[i] for i, t in enumerate(daily_t) if t0 <= t < t1]
            ax.plot(sub_daily_t, sub_daily_avg, color='#d62728', linewidth=2.0,
                    label='Daily Mean')

        ax.set_xlabel('Date / Time (UTC)', fontsize=13)
        ax.set_ylabel('SST (°C)', fontsize=13)
        ax.set_title(f'{title_suffix} at ({real_lat:.2f}°N, {real_lon:.2f}°E)',
                     fontsize=14, fontweight='bold')
        ax.xaxis.set_major_locator(date_loc)
        ax.xaxis.set_major_formatter(date_fmt)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=11)

        stats = (f'Mean: {np.nanmean(sub_sst):.2f}°C\n'
                 f'Min: {np.nanmin(sub_sst):.2f}°C\n'
                 f'Max: {np.nanmax(sub_sst):.2f}°C\n'
                 f'Range: {np.nanmax(sub_sst)-np.nanmin(sub_sst):.2f}°C\n'
                 f'N = {len(sub_sst)}')
        ax.text(0.01, 0.97, stats, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        plt.tight_layout()
        save_path = OUT_DIR / save_name
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f'已保存: {save_path}')

    point_tag = f'{real_lat:.2f}N_{real_lon:.2f}E'

    MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    # ── Seasonal Cycle（年尺度） ─────────────────────────────────
    plot_range(
        datetime(args.year, 1, 1), datetime(args.year + 1, 1, 1),
        f'SST Seasonal Cycle, {args.year}',
        f'sst_{point_tag}_{args.year}_seasonal.png',
        mdates.MonthLocator(), mdates.DateFormatter('%b'),
    )

    # ── Intra-monthly Variability（月尺度） ──────────────────────
    month = args.month
    plot_range(
        datetime(args.year, month, 1),
        (datetime(args.year, month + 1, 1) if month < 12
         else datetime(args.year + 1, 1, 1)),
        f'SST Intra-monthly Variability, {MONTH_NAMES[month-1]} {args.year}',
        f'sst_{point_tag}_{args.year}{month:02d}_intramonthly.png',
        mdates.DayLocator(interval=2), mdates.DateFormatter('%d'),
    )

    # ── Synoptic Variability（周尺度） ──────────────────────────
    week_start = datetime(args.year, month, 10)
    week_end   = week_start + timedelta(days=7)
    week_title = (f'{MONTH_NAMES[week_start.month-1]} '
                  f'{week_start.day}–{(week_end - timedelta(days=1)).day}, {args.year}')
    plot_range(
        week_start, week_end,
        f'SST Synoptic Variability, {week_title}',
        f'sst_{point_tag}_{week_start:%Y%m%d}_synoptic.png',
        mdates.DayLocator(interval=1), mdates.DateFormatter('%m-%d'),
        show_daily=True, figsize=(14, 5),
    )


if __name__ == '__main__':
    main()
