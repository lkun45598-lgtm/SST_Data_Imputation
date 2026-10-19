#!/usr/bin/env python3
"""
读取 preprocessing/build_single_samples.py 生成的预生成样本 h5，供单模型训练/评估。

batch 键 (float32 tensor，H=451, W=351):
  input_sst_seq   [30,H,W]  填充后 SST (归一化)，时间顺序，最后一帧 = 目标帧 (洞内为底图填充值)；陆地 = 0
  mask_seq        [30,H,W]  1 = 该帧该像素挖洞后仍无真实观测 (含目标帧洞，约定与旧模型一致: 1=缺失)
  c_seq           [1,H,W]   C / 48，过去 48h 观测次数 (目标帧)
  hour_feat       [2]       目标小时 sin/cos (UTC)
  ground_truth_sst[H,W]     目标帧真值 (归一化)，无真值处 = 0
  loss_mask       [H,W]     洞 ∩ 有真值 ∩ 海洋 (监督/评估区域)
  original_obs_mask[H,W]    目标帧挖洞后仍保留的真实观测
  land_mask       [H,W]     1 = 陆地
  base_sst        [H,W]     目标帧底图 (归一化)，GAN 残差 = 真值 - base
  hole            [H,W]     人工洞
  big             标量      大洞样本 (可用于 GAN)
  t               标量      目标时间槽
归一化沿用 OSTIA 常量，与预训练一致。
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

import os
from datetime import datetime, timedelta

MEAN, STD = 299.9221, 2.6919
FILL = -32768
LOOKBACK = 48


class SingleModelDataset(Dataset):
    def __init__(self, path, big_only=False, small_only=False, meta=None, t_range=None):
        """meta: 小的 npz (land, hours)，缺省则从逐小时库导出并缓存到 path 同目录 meta.npz，
        这样样本文件可以脱离 26GB 的原库使用。t_range=(lo,hi): 只取目标槽 lo<=t<hi 的样本。"""
        self.path = path
        with h5py.File(path, 'r') as f:
            ok = f['ok'][:] > 0
            big = f['big'][:] > 0
            self.W = f['sst'].shape[-1]
            self.offset = float(f.attrs['temp_offset_K'])
            tt = f['t'][:]
        sel = ok & (big if big_only else True) & (~big if small_only else True)
        if t_range is not None:
            sel &= (tt >= t_range[0]) & (tt < t_range[1])
        bad = os.path.join(os.path.dirname(os.path.abspath(path)), 'bad_idx.npy')     # 标 ok 但内容全零的样本 (写入中断所致)，见 HANDOFF
        if os.path.exists(bad):
            sel[np.load(bad)] = False
        self.idx = np.where(sel)[0]
        self.t_all = tt
        meta = meta or os.path.join(os.path.dirname(os.path.abspath(path)), 'meta.npz')
        if not os.path.exists(meta):
            from datasets.hourly_store import HourlyStore
            st = HourlyStore()
            hours = np.array([(t - datetime(1970, 1, 1)).total_seconds() // 3600 for t in st.times], np.int64)
            np.savez(meta, land=st.land, hours=hours)
        m = np.load(meta)
        self.land = m['land']
        self.times = np.array([datetime(1970, 1, 1) + timedelta(hours=int(h)) for h in m['hours']])
        self._f = None

    def __len__(self):
        return len(self.idx)

    def __getstate__(self):
        d = self.__dict__.copy(); d['_f'] = None
        return d

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.path, 'r')
        return self._f

    def _norm(self, q):
        valid = q != FILL
        k = q.astype(np.float32) / 100.0 + self.offset
        return np.where(valid, (k - MEAN) / STD, 0.0).astype(np.float32), valid

    def __getitem__(self, i):
        f = self._file(); j = int(self.idx[i])
        sst, _ = self._norm(f['sst'][j])
        truth, tv = self._norm(f['truth'][j])
        base, _ = self._norm(f['B'][j])
        W = self.W
        miss = np.unpackbits(f['miss'][j], axis=-1)[..., :W].astype(np.float32)
        hole = np.unpackbits(f['hole'][j], axis=-1)[..., :W].astype(bool)
        c = f['C'][j].astype(np.float32) / LOOKBACK
        t = int(f['t'][j]); hr = self.times[t].hour
        ang = 2 * np.pi * hr / 24.0
        ocean = ~self.land
        T = torch.from_numpy
        return dict(
            input_sst_seq=T(sst), mask_seq=T(miss), c_seq=T(c[None]),
            hour_feat=torch.tensor([np.sin(ang), np.cos(ang)], dtype=torch.float32),
            ground_truth_sst=T(truth),
            loss_mask=T((hole & tv & ocean).astype(np.float32)),
            original_obs_mask=T((tv & ~hole).astype(np.float32)),
            land_mask=T(self.land.astype(np.float32)),
            base_sst=T(base), hole=T(hole.astype(np.float32)),
            big=torch.tensor(float(f['big'][j])), t=torch.tensor(t),
        )
