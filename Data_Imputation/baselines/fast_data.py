"""In-memory datasets for the deep baselines — avoid per-sample network-h5 IO.

The shared SSTDatasetTemporal / JAXAFinetuneDataset open the (NFS) h5 file on
EVERY __getitem__ and run a per-sample 30x distance-transform fill. That is
hidden behind the FNO's 625M-param compute but starves a 7M baseline. Here we
load each h5 into RAM once (the node has ~1 TB), precompute the nearest-neighbour
input fill for OSTIA, and serve samples by pure numpy slicing. Same dict keys /
semantics as the originals, so training code is unchanged except the dataset class.
"""
import numpy as np, h5py
from scipy import ndimage
from torch.utils.data import Dataset


def _nearest_fill(frame):
    valid = (frame != 0) & (~np.isnan(frame))
    if not valid.any():
        return None
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return frame[tuple(idx)]


class OSTIAMem(Dataset):
    """OSTIA pretraining, fully in RAM. Mirrors datasets/ostia_dataset.SSTDatasetTemporal."""
    def __init__(self, hdf5_path, mean=None, std=None, window_size=30):
        self.window_size = window_size
        with h5py.File(hdf5_path, 'r') as f:
            inp = f['input_sst'][:].astype(np.float32)
            self.mask = f['missing_mask'][:].astype(np.uint8)
            gt_raw = f['ground_truth_sst'][:].astype(np.float32)
            self.land = f['land_mask'][:].astype(np.uint8)
        # gt has ~0.8% NaN in ocean -> track validity so those pixels are EXCLUDED
        # from the loss/MAE region (else nan->0 becomes -104 in norm space and blows up loss)
        self.gt_valid = np.isfinite(gt_raw).astype(np.uint8)
        self.gt = np.nan_to_num(gt_raw, nan=0.0)
        if mean is None or std is None:
            v = self.gt[self.gt != 0]
            self.mean = float(v.mean()); self.std = float(v.std())
        else:
            self.mean, self.std = float(mean), float(std)
        N = inp.shape[0]
        for i in range(N):                       # precompute nearest-fill ONCE
            filled = _nearest_fill(inp[i])
            inp[i] = filled if filled is not None else self.mean
        self.inp = inp; self.N = N
        print(f'  [OSTIAMem] {N} frames in RAM ({hdf5_path.split("/")[-1]})', flush=True)

    def __len__(self):
        return self.N

    def _win(self, arr, idx):
        s = max(0, idx - self.window_size + 1)
        seq = list(arr[s:idx + 1])
        while len(seq) < self.window_size:
            seq.insert(0, seq[0])
        return np.stack(seq)

    def __getitem__(self, idx):
        sst = (self._win(self.inp, idx).astype(np.float32) - self.mean) / self.std
        gt = (self.gt[idx].astype(np.float32) - self.mean) / self.std
        return {
            'input_sst_seq': np.nan_to_num(sst, nan=0.0),
            'mask_seq': self._win(self.mask, idx).astype(np.float32),
            'ground_truth_sst': np.nan_to_num(gt, nan=0.0),
            'missing_mask': self.mask[idx].astype(np.float32),
            'gt_valid': self.gt_valid[idx].astype(np.float32),
            'land_mask': self.land.astype(np.float32),
        }


class JAXAMem(Dataset):
    """JAXA finetuning, fully in RAM. Mirrors inference.JAXAFinetuneDataset (realcloud)."""
    def __init__(self, data_dir, series_ids, mean, std, ratio_range, seed=42, window_size=30):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(data_dir).resolve().parents[1]))
        from inference.real_cloud_mask import build_cloud_bank, RealCloudMaskGenerator
        self.window_size = window_size; self.mean, self.std = float(mean), float(std)
        self.sst, self.obs, self.miss, self.series_len = [], [], [], []
        paths = []
        for sid in series_ids:
            p = f'{data_dir}/jaxa_knn_filled_{sid:02d}.h5'; paths.append(p)
            with h5py.File(p, 'r') as f:
                self.sst.append(f['sst_data'][:].astype(np.float32))
                self.obs.append(f['original_obs_mask'][:].astype(np.uint8))
                self.miss.append(f['original_missing_mask'][:].astype(np.uint8))
                if not hasattr(self, 'land'):
                    self.land = f['land_mask'][:].astype(np.uint8)
            self.series_len.append(self.sst[-1].shape[0])
        self.cum = np.cumsum([0] + self.series_len); self.total = int(self.cum[-1])
        bank = build_cloud_bank(paths, max_donors=400, seed=seed)
        self.gen = RealCloudMaskGenerator(bank, seed=seed, ratio_range=ratio_range)
        self.ocean = (self.land == 0)
        print(f'  [JAXAMem] series {series_ids} = {self.total} frames in RAM', flush=True)

    def __len__(self):
        return self.total

    def _locate(self, idx):
        for k in range(len(self.series_len)):
            if idx < self.cum[k + 1]:
                return k, idx - self.cum[k]
        raise IndexError

    def _win(self, arr, k, li):
        s = max(0, li - self.window_size + 1)
        seq = list(arr[k][s:li + 1])
        while len(seq) < self.window_size:
            seq.insert(0, seq[0])
        return np.stack(seq)

    def __getitem__(self, idx):
        k, li = self._locate(idx)
        sst_seq = self._win(self.sst, k, li).astype(np.float32)
        mask_seq = self._win(self.miss, k, li).astype(np.float32)
        obs = self.obs[k][li]
        art = self.gen.generate((obs * self.ocean).astype(np.float32)).astype(np.float32)
        loss_mask = (art * obs * self.ocean).astype(np.float32)
        mask_seq[-1] = art
        gt = (sst_seq[-1].copy().astype(np.float32) - self.mean) / self.std
        sst_in = sst_seq.copy()
        sst_in[-1] = np.where(art == 1, self.mean, sst_seq[-1])
        sst_in = (sst_in - self.mean) / self.std
        return {
            'input_sst_seq': np.nan_to_num(sst_in, nan=0.0),
            'mask_seq': mask_seq,
            'ground_truth_sst': np.nan_to_num(gt, nan=0.0),
            'loss_mask': loss_mask,
            'original_obs_mask': obs.astype(np.float32),
            'land_mask': self.land.astype(np.float32),
        }
