#!/usr/bin/env python3
"""
缺口交界处的接缝指标。

沿 x/y 两个轴、两个方向，取"观测侧两个像素 + 缺口侧两个像素"的四元组:
    u[i-1], u[i] 在观测区;  u[i+1], u[i+2] 在填充区
    s_out = u[i]   - u[i-1]     (观测侧斜率)
    s_in  = u[i+2] - u[i+1]     (填充侧斜率)
    step  = u[i+1] - u[i]       (跨界增量)
  slope_jump = |s_in - s_out|                 梯度不连续度
  value_jump = |step - (s_in + s_out) / 2|    跨界台阶 (光滑场应接近 0)
单位 K/像素。参考值: 同一位置上真值的同一指标 (有真值时)，
或界线附近全观测四元组的同一指标 (无真值时)。
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def _scan(u, left, right):
    cond = left[:, :-3] & left[:, 1:-2] & right[:, 2:-1] & right[:, 3:]
    um1, u0, u1, u2 = u[:, :-3], u[:, 1:-2], u[:, 2:-1], u[:, 3:]
    s_out, s_in, step = u0 - um1, u2 - u1, u1 - u0
    return np.abs(s_in - s_out)[cond], np.abs(step - 0.5 * (s_in + s_out))[cond]


def _orientations(u, left, right):
    yield u, left, right
    yield u[:, ::-1], left[:, ::-1], right[:, ::-1]
    yield u.T, left.T, right.T
    yield u.T[:, ::-1], left.T[:, ::-1], right.T[:, ::-1]


def _stats(u, left, right):
    sj, vj = [], []
    for a, l, r in _orientations(u, left, right):
        s, v = _scan(a, l, r)
        sj.append(s)
        vj.append(v)
    sj, vj = np.concatenate(sj), np.concatenate(vj)
    if len(sj) == 0:
        return dict(slope_jump=np.nan, value_jump=np.nan, n=0)
    return dict(slope_jump=float(sj.mean()), value_jump=float(vj.mean()), n=int(len(sj)))


def seam_stats(u, known, hole):
    """u: 填充结果; known: 观测像素; hole: 被填充的缺口像素"""
    return _stats(u, known, hole)


def seam_reference(u, known, hole, rim=10):
    """无真值时的参考: 缺口 rim 像素范围内、全部由观测构成的四元组"""
    near = known & (distance_transform_edt(~hole) <= rim)
    return _stats(u, near, near)


def gradient_ratio(fill, truth, hole):
    """缺口内部平均梯度幅值之比 fill/truth (<1 表示被抹平)"""
    inner = binary_erosion(hole, iterations=1)
    if not inner.any():
        return np.nan
    gm = lambda a: np.hypot(*np.gradient(a))
    return float(gm(fill)[inner].mean() / gm(truth)[inner].mean())


def _hp(a, valid, sigma):
    """高频分量 = 原场 - 只在 valid 像素上做的归一化高斯平滑 (NaN/无效像素不当作 0)"""
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.where(valid, a, 0.0), sigma, mode='constant')
    den = gaussian_filter(valid.astype(np.float64), sigma, mode='constant')
    return np.where(valid, a - num / np.maximum(den, 1e-6), np.nan)


def highpass_stats(fill, truth, hole, ocean, sigma=3.0, erode=2):
    """缺口内部高频分量: 与真值的相关系数 (细节位置对不对) 和标准差之比 (幅度够不够)。
    fill 与 truth 使用同一个有效域 (海洋 & 真值有观测)"""
    valid = ocean & np.isfinite(truth)
    inner = (binary_erosion(hole, iterations=erode) if erode else hole) & valid
    if inner.sum() < 50:
        return dict(hp_corr=np.nan, hp_ratio=np.nan)
    hf, ht = _hp(fill, valid, sigma)[inner], _hp(truth, valid, sigma)[inner]
    return dict(hp_corr=float(np.corrcoef(hf, ht)[0, 1]), hp_ratio=float(hf.std() / (ht.std() + 1e-12)))
