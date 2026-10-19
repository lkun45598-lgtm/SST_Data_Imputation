#!/usr/bin/env python3
"""
训练时在线换洞: 固定洞会被模型记住 (eval_memorization.py: 同一批训练帧上，固定洞 RMSE 随训练一直降，新洞几乎不动)。

整样本重算约 37 CPU 秒，在线用不起。但观察 make_sample: 窗口里历史帧 (k>=1) 的填充只依赖
  (t-k 帧 + 源云事件 ev 的叠加 + 更早的帧)，与目标帧的洞无关；只有目标帧 (k=0) 的填充、底图 B 依赖洞。
C 只数过去 48h 的观测，同样与洞无关。所以: 复用文件里存好的历史 29 帧，只对目标帧换洞重算 (一次 fill_from，
约 1-2 CPU 秒)。源云事件时间 s0 用生成时的随机数流重放得到 (种子 = seed*1000003 + t，与 build_single_samples 一致)。

正确性靠两条自检 (见 experiments/single_model_v1/verify_online_holes.py):
  1. 用原样本的洞重算，目标帧填充/B/C 必须与文件里的逐位相同 (验证 s0 重放和"历史与洞无关"的假设);
  2. 换新洞后，历史 29 帧原封不动。
局限: 历史 29 帧和源云事件在各 epoch 里仍是固定的，换的只是目标帧的洞。
"""
import numpy as np

from datasets.hourly_store import HourlyStore
from datasets.artificial_holes import LOOKBACK, build_bank_from_store, fill_from
from inference.real_cloud_mask import RealCloudMaskGenerator

BIG_COVERAGE = 0.7
SMALL, BIG = (0.20, 0.30), (0.40, 0.60)
FILL = -32768


def quantize(a):
    out = np.full(a.shape, FILL, np.int16)
    m = np.isfinite(a)
    out[m] = np.round((a[m] - 273.15) * 100).astype(np.int16)
    return out


def source_slots(st, lo, hi):
    """与 build_single_samples._init 里的 src 完全一致 (lo/hi 必须用生成样本时的范围)"""
    tr = st.slot_range(lo, hi)
    return [int(s) for s in tr if s >= 100 and st.present[s - 80:s + 1].all() and 0.2 <= st.coverage[s] <= 0.6]


def ratio_range(st, t):
    return BIG if st.coverage[t] >= BIG_COVERAGE else SMALL


def replay_source_slot(st, t, src, seed=0):
    """重放 build_single_samples._work + make_sample 的随机数消耗顺序，返回当时选中的源云事件时间槽 s0"""
    rng = np.random.default_rng(seed * 1000003 + t)
    lo, hi = ratio_range(st, t)
    rng.uniform(lo, hi)              # ratio
    rng.integers(1 << 31)            # RealCloudMaskGenerator 的种子
    return int(rng.choice(src))


def resample_target(st, t, s0, bank=None, rng=None, ratio=None, hole=None, lookback=LOOKBACK):
    """目标帧 t 换洞重算。hole 给定时 (bool, 自检用) 直接使用，否则按 ratio 用云库新生成。
    返回 dict(hole, filled0(float32 K), known0, B(float32 K), C(uint8), truth(float32 K)) 或 None。"""
    oc = st.ocean
    cur = st.frame(t); fin = np.isfinite(cur) & oc
    if hole is None:
        gen = RealCloudMaskGenerator(bank, seed=int(rng.integers(1 << 31)), ratio_range=(ratio, ratio))
        hole = gen.generate(fin, target_ratio=ratio) > 0
    cur_h = cur.copy(); cur_h[hole] = np.nan
    past = {}
    for l in range(1, lookback + 1):
        f = st.frame(t - l).copy() if st.present[t - l] else np.full(st.shape, np.nan, np.float32)
        if st.present[s0 - l]:
            f[(~np.isfinite(st.frame(s0 - l))) & oc] = np.nan          # 叠加源云事件的缺失
        else:
            pass                                                        # 源帧缺失: 与 make_sample 一致，不叠加 (全 False)
        past[l] = f
    known = np.isfinite(cur_h) & oc
    r = fill_from(st, cur_h, past, known)
    if r is None:
        return None
    C = sum((np.isfinite(p) & oc).astype(np.uint8) for p in past.values())
    return dict(hole=hole, filled0=r[0], known0=known, B=r[1], C=C, truth=cur)


def build_bank(st, slots, n=300, seed=0):
    return build_bank_from_store(st, slots, n=n, seed=seed)
