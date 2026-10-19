#!/usr/bin/env python3
"""
SingleModelDataset 的在线换洞版本 (仅用于训练集)。输出的 batch 键与 SingleModelDataset 完全一致。

每次取样: 按目标帧覆盖率抽新的洞比例 -> 用云库生成新洞 -> 目标帧重算填充/底图 B/C (datasets/online_holes.py)，
历史 29 帧沿用文件里存的结果。每个 worker 用 (seed, worker_id, 调用计数, 样本号) 作种子，所以同一帧每个 epoch 的洞都不同；
以 p_online 的概率换洞，其余走文件里的固定洞 (p_online=0 时与父类行为完全相同)。
新洞比例过小 (< min_hole 像素) 或填充失败时回退到文件里的固定洞 (与生成样本时的跳过规则一致)。
注意: 云库只取训练期 (bank_slot_lt 之前) 的帧，源云事件池沿用生成样本时的范围，保证种子重放一致。
"""
from datetime import datetime
import numpy as np
import torch

from datasets.single_model_dataset import SingleModelDataset, LOOKBACK
from datasets.hourly_store import HourlyStore
from datasets import online_holes as O


class OnlineHoleDataset(SingleModelDataset):
    def __init__(self, path, store, seed=0, p_online=1.0, bank_slot_lt=None, gen_seed=0, min_hole=2000, **kw):
        super().__init__(path, **kw)
        self.seed, self.p, self.min_hole, self._n = seed, p_online, min_hole, 0
        st = HourlyStore(store)
        assert len(st.times) == len(self.times) and st.times[0] == self.times[0], '逐小时库与样本文件的时间槽不一致'
        lo, hi = datetime(2015, 7, 7), datetime(2023, 7, 5)          # 必须与生成样本时一致
        self.src = O.source_slots(st, lo, hi)
        slots = [int(s) for s in st.slot_range(lo, hi) if bank_slot_lt is None or s < bank_slot_lt]
        self.bank = O.build_bank(st, slots)
        self.st, self.gen_seed, self._s0 = st, gen_seed, {}

    def _assemble(self, j, t, sst_q, miss, c, truth_q, base_q, hole):
        """与父类 __getitem__ 相同的组装逻辑 (自检时用同一路径复现父类输出)"""
        sst, _ = self._norm(sst_q); truth, tv = self._norm(truth_q); base, _ = self._norm(base_q)
        ang = 2 * np.pi * self.times[t].hour / 24.0
        T = torch.from_numpy; ocean = ~self.land
        return dict(
            input_sst_seq=T(sst), mask_seq=T(miss.astype(np.float32)), c_seq=T((c.astype(np.float32) / LOOKBACK)[None]),
            hour_feat=torch.tensor([np.sin(ang), np.cos(ang)], dtype=torch.float32),
            ground_truth_sst=T(truth),
            loss_mask=T((hole & tv & ocean).astype(np.float32)),
            original_obs_mask=T((tv & ~hole).astype(np.float32)),
            land_mask=T(self.land.astype(np.float32)),
            base_sst=T(base), hole=T(hole.astype(np.float32)),
            big=torch.tensor(float(self._file()['big'][j])), t=torch.tensor(t),
        )

    def stored_args(self, j):
        f = self._file(); W = self.W
        return (f['sst'][j], np.unpackbits(f['miss'][j], axis=-1)[..., :W], f['C'][j], f['truth'][j], f['B'][j],
                np.unpackbits(f['hole'][j], axis=-1)[..., :W].astype(bool))

    def __getitem__(self, i):
        j = int(self.idx[i]); t = int(self.t_all[j])
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi is not None else 0
        self._n += 1
        rng = np.random.default_rng([self.seed, wid, self._n, j])
        if self.p < 1.0 and rng.random() >= self.p:
            return super().__getitem__(i)
        if t not in self._s0:
            self._s0[t] = O.replay_source_slot(self.st, t, self.src, self.gen_seed)
        lo, hi = O.ratio_range(self.st, t)
        r = O.resample_target(self.st, t, self._s0[t], self.bank, rng, float(rng.uniform(lo, hi)))
        if r is None or r['hole'].sum() < self.min_hole:
            return super().__getitem__(i)
        sst_q, miss, c, truth_q, base_q, _ = self.stored_args(j)
        sst_q = sst_q.copy(); sst_q[-1] = O.quantize(r['filled0'])
        miss = miss.copy(); miss[-1] = ((~r['known0']) & self.st.ocean).astype(miss.dtype)
        return self._assemble(j, t, sst_q, miss, r['C'], truth_q, O.quantize(r['B']), r['hole'])
