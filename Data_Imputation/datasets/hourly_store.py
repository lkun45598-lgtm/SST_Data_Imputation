#!/usr/bin/env python3
"""
读取 build_hourly_store.py 生成的连续逐小时原始库 (experiments/hourly_store_v1/sst_hourly.h5)。

frame(slot) 返回开尔文温度，无观测处为 NaN。H5 句柄在各进程内惰性打开，
可安全用于 DataLoader 多进程。
"""

import os
import h5py
import numpy as np
from datetime import datetime, timedelta

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'experiments', 'hourly_store_v1', 'sst_hourly.h5')


class HourlyStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        with h5py.File(path, 'r') as f:
            self.land = f['land_mask'][:] > 0
            self.present = f['present'][:] > 0
            self.valid_count = f['valid_count'][:].astype(np.float64)
            hours = f['slot_hours'][:]
            self.offset = float(f.attrs['offset'])
            self.scale = float(f.attrs['scale'])
            self.fill = int(f.attrs['fill_value'])
        self.ocean = ~self.land
        self.T = len(self.present)
        self.shape = self.land.shape
        t0 = datetime(1970, 1, 1)
        self.times = np.array([t0 + timedelta(hours=int(h)) for h in hours])
        self.coverage = self.valid_count / self.ocean.sum()  # 海洋像素观测覆盖率
        self._f = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_f'] = None
        return d

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.path, 'r')
        return self._f

    def frame(self, slot):
        q = self._file()['sst'][slot]
        out = q.astype(np.float32) * np.float32(self.scale) + np.float32(self.offset)
        out[q == self.fill] = np.nan
        return out

    def slot_range(self, start, end):
        """[start, end) 内的时间槽下标"""
        return np.where((self.times >= start) & (self.times < end))[0]
