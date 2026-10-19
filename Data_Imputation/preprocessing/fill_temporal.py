#!/usr/bin/env python3
"""
结合时序的填充方法 (只用目标时刻之前的帧，与模型输入一致，因果、无未来信息)。

接口 fill(field, known, ocean, ctx) -> (H,W) float32，语义同 fill_methods。
ctx 由 build_context 生成:
  past   : {lag小时: 原始观测帧(NaN=无观测)}，lag=1..max_lag
  sameh  : {lag=24k小时: 原始观测帧}，前 k 天的同一小时
  prev   : 递归填充得到的上一时刻完整场 F_{t-1}

共同步骤:
  1) 背景场 B: 由过去帧合成。每个过去帧先做偏差对齐 —— 用它与目标帧(挖洞后)共同
     观测像素上的差值中位数平移，抵消日变化/时间漂移。
  2) B 中从未被观测到的像素，用调和延拓补全。
  3) 边界校正 (_bc): 在已观测像素上取残差 r = obs - B，把 r 调和延拓进缺口，
     u = B + 延拓(r)。保证与周围观测在边界上连续，并让偏差向缺口内部平滑衰减。
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from preprocessing.fill_methods import _harmonic_solve, harmonic_fill, _finish


def _offsets(field, known, frames, min_common=200):
    offs = {}
    for lag, fr in frames.items():
        common = known & np.isfinite(fr)
        if common.sum() >= min_common:
            offs[lag] = float(np.median(field[common] - fr[common]))
    return offs


def _weighted_background(frames, offs, weight):
    shape = next(iter(frames.values())).shape
    num, den = np.zeros(shape), np.zeros(shape)
    for lag, fr in frames.items():
        if lag not in offs:
            continue
        v = np.isfinite(fr)
        w = weight(lag)
        num[v] += w * (fr[v] + offs[lag])
        den[v] += w
    B = np.full(shape, np.nan)
    ok = den > 0
    B[ok] = num[ok] / den[ok]
    return B


def _latest_background(frames, offs):
    shape = next(iter(frames.values())).shape
    B = np.full(shape, np.nan)
    for lag in sorted(frames):
        if lag not in offs:
            continue
        fr = frames[lag]
        ok = np.isnan(B) & np.isfinite(fr)
        B[ok] = fr[ok] + offs[lag]
    return B


def _complete(B, ocean):
    """B 里没有任何过去观测的海洋像素: 用调和延拓补全"""
    kn = np.isfinite(B) & ocean
    if kn.sum() == 0:
        return np.zeros(B.shape)
    full = harmonic_fill(np.where(kn, B, 0.0), kn, ocean)
    return np.where(np.isfinite(full), full, 0.0)


def _smooth(B_full, ocean, sigma):
    """海洋内的归一化高斯平滑 (不把陆地算进去)，压掉逐像素取自不同时刻帧带来的噪点"""
    oc = ocean.astype(np.float64)
    num = gaussian_filter(np.where(ocean, B_full, 0.0), sigma, mode='constant')
    den = gaussian_filter(oc, sigma, mode='constant')
    return np.where(ocean & (den > 1e-6), num / np.maximum(den, 1e-6), B_full)


def _boundary_correct(B_full, field, known, ocean):
    resid = np.where(known, field - B_full, 0.0)
    corr = _harmonic_solve(resid, known, ocean)
    return _finish(field, known, ocean, B_full + np.where(known, 0.0, corr))


def _make(background, bc, frames_key, max_lag=None, smooth=0.0):
    def fn(field, known, ocean, ctx):
        frames = ctx[frames_key]
        if max_lag is not None:
            frames = {l: f for l, f in frames.items() if l <= max_lag}
        offs = _offsets(field, known, frames)
        B = background(frames, offs)
        B_full = _complete(B, ocean)
        if smooth > 0:
            B_full = _smooth(B_full, ocean, smooth)
        if not bc:
            return _finish(field, known, ocean, B_full)
        return _boundary_correct(B_full, field, known, ocean)
    return fn


def rec_bc_fill(field, known, ocean, ctx, alpha=0.1):
    """递归: 背景 = (1-alpha)*上一时刻填充场 + alpha*当前帧空间调和填充，再做边界校正"""
    S = np.nan_to_num(harmonic_fill(field, known, ocean))
    B_full = (1.0 - alpha) * ctx['prev'] + alpha * S
    return _boundary_correct(B_full, field, known, ocean)


TEMPORAL_METHODS = {
    'tw': _make(lambda f, o: _weighted_background(f, o, lambda l: 1.0 / l), False, 'past'),
    'tw_bc': _make(lambda f, o: _weighted_background(f, o, lambda l: 1.0 / l), True, 'past'),
    'tw_bc_s1': _make(lambda f, o: _weighted_background(f, o, lambda l: 1.0 / l), True, 'past', smooth=1.0),
    'tw_bc_s2': _make(lambda f, o: _weighted_background(f, o, lambda l: 1.0 / l), True, 'past', smooth=2.0),
    'tw_bc_s4': _make(lambda f, o: _weighted_background(f, o, lambda l: 1.0 / l), True, 'past', smooth=4.0),
    'lvo_bc': _make(_latest_background, True, 'past'),
    'sameh_bc': _make(lambda f, o: _weighted_background(f, o, lambda l: 24.0 / l), True, 'sameh'),
    'rec_bc': rec_bc_fill,
}


def build_context(store, slot, max_lag=48, sameh_days=7, rec_steps=48, alpha=0.1):
    """只读取 slot 之前的帧。缺帧按全 NaN 处理。"""
    ocean = store.ocean
    nan_frame = np.full(store.shape, np.nan, dtype=np.float32)

    def get(s):
        return store.frame(s) if 0 <= s < store.T and store.present[s] else nan_frame

    past = {l: get(slot - l) for l in range(1, max_lag + 1)}
    sameh = {24 * k: get(slot - 24 * k) for k in range(1, sameh_days + 1)}

    F = None
    for s in range(slot - rec_steps, slot):
        obs = get(s)
        kn = np.isfinite(obs) & ocean
        if kn.sum() < 100:  # 该小时几乎无观测: 沿用上一时刻
            continue
        if F is None:
            F = np.nan_to_num(harmonic_fill(obs, kn, ocean))
            continue
        S = np.nan_to_num(harmonic_fill(obs, kn, ocean))
        B = (1.0 - alpha) * F + alpha * S
        F = np.nan_to_num(_boundary_correct(B, obs, kn, ocean))
    if F is None:
        F = np.zeros(store.shape)
    return {'past': past, 'sameh': sameh, 'prev': F}
