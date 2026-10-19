#!/usr/bin/env python3
"""
时间一致人工洞样本 (训练/评估共用)。

做法: 目标帧 t (高覆盖) 上用 RealCloudMaskGenerator 的真实云形状挖洞 (洞必落在真值像素上);
再取一个"源云事件"的过去 LOOKBACK 小时真实缺失序列叠到 t 之前的历史帧, 使历史也缺失,
这样 C 的分布接近真实。所有挖洞先于填充: 底图 B 与 C 在挖洞后重算, 不会泄漏真值。

约定: 洞内像素 = 目标帧人工遮挡; 评估/监督只在洞内进行。
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datasets.hourly_store import HourlyStore
from inference.real_cloud_mask import RealCloudMaskGenerator
from preprocessing.fill_temporal import _offsets, _weighted_background, _complete
from preprocessing.fill_methods import _harmonic_solve
from scipy.ndimage import binary_erosion

LOOKBACK = 48
ERODE, CLIP = 2, 1.0


def build_bank_from_store(st, slots, n=400, seed=0, lo=0.05, hi=0.8):
    """从逐小时库取真实云足迹 (1=云, 仅海洋)。只用 slots 里的帧 (调用方保证只含训练期)。"""
    rng = np.random.default_rng(seed); out = []
    for s in rng.permutation(slots):
        if not st.present[s]: continue
        c = (~np.isfinite(st.frame(int(s)))) & st.ocean
        fr = c.sum() / st.ocean.sum()
        if lo <= fr <= hi: out.append(c)
        if len(out) >= n: break
    return np.array(out, dtype=bool)


def fill_from(st, cur, past, known):
    """tw + 去云边 bc + tanh 限幅 (与 build_filled_store 一致), 返回完整场; past: {lag: frame}"""
    oc = st.ocean
    field = np.where(known, cur, np.nan)
    bg = _weighted_background(past, _offsets(field, known, past), lambda l: 1.0 / l)
    if np.isfinite(bg[oc]).sum() < 2000: return None
    B = _complete(bg, oc)
    src = binary_erosion(known, iterations=ERODE) & known
    if src.sum() < 100: src = known
    d = CLIP * np.tanh(np.where(src, field - B, 0.0) / CLIP)
    out = np.where(known, cur, B + _harmonic_solve(d, src, oc) * (~src))
    out[~oc] = np.nan
    return out.astype(np.float32), B.astype(np.float32)


def make_sample(st, t, bank, rng, ratio, src_slots, window=30, lookback=LOOKBACK, full_window=True, min_hole=2000):
    """返回 dict 或 None。window 内每一帧 (t-window+1..t) 的填充都在挖洞后重算。
    full_window=False 时只算目标帧 (用于统计)。"""
    oc = st.ocean
    cur = st.frame(t); fin = np.isfinite(cur) & oc
    gen = RealCloudMaskGenerator(bank, seed=int(rng.integers(1 << 31)), ratio_range=(ratio, ratio))
    hole = gen.generate(fin, target_ratio=ratio) > 0
    if hole.sum() < min_hole: return None
    # 源云事件: 一个真实时间段的缺失序列, 叠到 t 之前的历史上
    s0 = int(rng.choice(src_slots))
    ev = [(~np.isfinite(st.frame(s0 - k))) & oc if st.present[s0 - k] else np.zeros(st.shape, bool) for k in range(0, lookback + window)]
    frames = {}
    def get(k):                                    # 取 t-k 帧 (k>=0), 叠加历史云事件; k=0 是目标帧
        if k not in frames:
            f = st.frame(t - k).copy() if st.present[t - k] else np.full(st.shape, np.nan, np.float32)
            f[ev[k] if k > 0 else hole] = np.nan
            frames[k] = f
        return frames[k]
    ks = range(0, window) if full_window else [0]
    filled, known_k, B0 = {}, {}, None; C0 = None
    for k in ks:
        known = np.isfinite(get(k)) & oc
        past = {l: get(k + l) for l in range(1, lookback + 1)}
        r = fill_from(st, get(k), past, known)
        if r is None: return None
        filled[k] = r[0]; known_k[k] = known
        if k == 0:
            B0 = r[1]; C0 = sum((np.isfinite(p) & oc).astype(np.uint8) for p in past.values())
    return dict(t=t, hole=hole, truth=cur, filled=filled, known=known_k, B=B0, C=C0, obs=fin & ~hole)
