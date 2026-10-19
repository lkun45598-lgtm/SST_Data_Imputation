#!/usr/bin/env python3
"""
单帧空间填充方法，用来生成缺口里连续平滑的趋势底图。

统一接口 fill(field, known, ocean) -> (H,W) float32
  field : 观测值数组 (known 为 False 处的值被忽略)
  known : 观测像素 (bool)，必须是海洋
  ocean : 海洋像素 (bool)
返回: 观测像素保持原值，海洋内缺失像素被填充，陆地为 NaN。
所有方法只依赖当前帧，不使用时间信息; 因此被挖洞后重新填充即可避免真值泄漏。
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt


def _finish(field, known, ocean, filled):
    out = np.where(known, field, filled).astype(np.float32)
    out[~ocean] = np.nan
    return out


def _coords(field, known, ocean):
    pk = np.argwhere(known)
    pu = np.argwhere(ocean & ~known)
    return pk, field[known], pu


def idw_fill(field, known, ocean, k=20, power=2.0):
    """反距离加权 (旧流程里的空间 KNN 填充的单帧版本)"""
    pk, vals, pu = _coords(field, known, ocean)
    filled = np.array(field, dtype=np.float64)
    if len(pu):
        d, i = cKDTree(pk).query(pu, k=k, workers=-1)
        w = 1.0 / np.maximum(d, 1e-6) ** power
        filled[tuple(pu.T)] = (w * vals[i]).sum(1) / w.sum(1)
    return _finish(field, known, ocean, filled)


def _griddata_fill(field, known, ocean, cubic):
    pk, vals, pu = _coords(field, known, ocean)
    filled = np.array(field, dtype=np.float64)
    if len(pu):
        interp = (CloughTocher2DInterpolator if cubic else LinearNDInterpolator)(pk, vals)
        z = interp(pu)
        bad = ~np.isfinite(z)  # 凸包外
        if bad.any():
            _, i = cKDTree(pk).query(pu[bad], k=1, workers=-1)
            z[bad] = vals[i]
        filled[tuple(pu.T)] = z
    return _finish(field, known, ocean, filled)


def linear_fill(field, known, ocean):
    return _griddata_fill(field, known, ocean, cubic=False)


def cubic_fill(field, known, ocean):
    return _griddata_fill(field, known, ocean, cubic=True)


def _harmonic_solve(field, known, ocean):
    """在缺失海洋区域解 Laplace 方程: 已知像素为 Dirichlet 边界，陆地为零通量 (不参与)"""
    unk = ocean & ~known
    n = int(unk.sum())
    filled = np.array(field, dtype=np.float64)
    if n == 0:
        return filled
    idx = -np.ones(field.shape, dtype=np.int64)
    idx[unk] = np.arange(n)
    pad = lambda a, v: np.pad(a, 1, constant_values=v)
    oc_p, unk_p, kn_p = pad(ocean, False), pad(unk, False), pad(known, False)
    f_p, idx_p = pad(np.nan_to_num(field), 0.0), pad(idx, -1)
    H, W = field.shape
    deg = np.zeros(n)
    b = np.zeros(n)
    rows, cols = [], []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sl = (slice(1 + dy, 1 + dy + H), slice(1 + dx, 1 + dx + W))  # 邻居
        pi = idx[unk]
        deg += oc_p[sl][unk]
        nb_unk = unk_p[sl][unk]
        rows.append(pi[nb_unk])
        cols.append(idx_p[sl][unk][nb_unk])
        nb_kn = kn_p[sl][unk]
        np.add.at(b, pi[nb_kn], f_p[sl][unk][nb_kn])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sparse.coo_matrix((-np.ones(len(rows)), (rows, cols)), shape=(n, n)) + sparse.diags(deg)
    eps = 1e-6  # 与已知像素不连通的孤立区域: 退化为均值，避免奇异
    mean_known = float(field[known].mean()) if known.any() else 0.0
    A = (A + eps * sparse.eye(n)).tocsc()
    b = b + eps * mean_known
    filled[unk] = spsolve(A, b)
    return filled


def harmonic_fill(field, known, ocean):
    return _finish(field, known, ocean, _harmonic_solve(field, known, ocean))


def biharmonic_fill(field, known, ocean):
    """薄板样条 (双调和) 填充，边界上同时匹配值和斜率。陆地用最近海洋观测临时填值"""
    from skimage.restoration import inpaint_biharmonic
    unk = ocean & ~known
    img = np.array(field, dtype=np.float64)
    _, (iy, ix) = distance_transform_edt(~known, return_indices=True)
    img[~ocean] = img[iy[~ocean], ix[~ocean]]
    img[unk] = 0.0
    filled = inpaint_biharmonic(img, unk) if unk.any() else img
    return _finish(field, known, ocean, filled)


def _trend(field, known, scales=(10, 20, 40, 80), min_weight=0.1):
    """多尺度归一化卷积: 每个像素取满足权重门限的最小尺度，得到平滑的大尺度趋势"""
    v = np.where(known, field, 0.0).astype(np.float64)
    m = known.astype(np.float64)
    trend = np.full(field.shape, np.nan)
    for s in scales:
        num = gaussian_filter(v, s, mode='constant', truncate=3.0)
        den = gaussian_filter(m, s, mode='constant', truncate=3.0)
        ok = np.isnan(trend) & (den >= min_weight)
        trend[ok] = num[ok] / den[ok]
    bad = np.isnan(trend)
    if bad.any():  # 极端情况: 离所有观测都很远
        _, (iy, ix) = distance_transform_edt(~known, return_indices=True)
        trend[bad] = field[iy[bad], ix[bad]]
    return trend


def trend_only_fill(field, known, ocean):
    """仅大尺度趋势 (不做边界校正)，用来展示接缝问题"""
    return _finish(field, known, ocean, _trend(field, known))


def trend_boundary_fill(field, known, ocean):
    """趋势 + 边界校正: 观测处的趋势残差作 Dirichlet 边界，调和延拓进缺口，
    使填充在边界上与观测值连续，并向缺口内部平滑衰减"""
    trend = _trend(field, known)
    resid = np.where(known, field - trend, 0.0)
    corr = _harmonic_solve(resid, known, ocean)
    return _finish(field, known, ocean, trend + np.where(known, 0.0, corr))


METHODS = {
    'idw': idw_fill,
    'linear': linear_fill,
    'cubic': cubic_fill,
    'harmonic': harmonic_fill,
    'biharmonic': biharmonic_fill,
    'trend_only': trend_only_fill,
    'trend_bc': trend_boundary_fill,
}
