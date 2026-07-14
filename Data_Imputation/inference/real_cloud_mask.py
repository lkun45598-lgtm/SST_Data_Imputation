#!/usr/bin/env python3
"""Real cloud-shape mask generator for self-supervised SST training/eval.

Instead of small random squares, borrow REAL cloud footprints from the
dataset itself (1 - original_obs_mask of donor frames) and intersect with
the TARGET frame's real observations, so:
  * the hole SHAPE is a genuine cloud pattern (irregular, fractal, large);
  * every masked pixel is a REAL observation (has ground truth) -> can be
    supervised / evaluated. Masks never fall on cloud/land.

Usage:
    bank = build_cloud_bank(list_of_h5_paths, max_donors=500)
    gen  = RealCloudMaskGenerator(bank, seed=42, ratio_range=(0.2, 0.9))
    art  = gen.generate(eligible)   # eligible = original_obs_mask * ocean
"""
import numpy as np
import h5py


def build_cloud_bank(h5_paths, max_donors=500, min_cloud_frac=0.05,
                     max_cloud_frac=0.80, seed=0):
    """Collect real ocean cloud footprints (1=cloud) from donor frames.

    Keep only mid-coverage donors (min..max frac): near-clear ones carry no
    shape, near-total ones just hug the rectangular domain edge (looks fake).
    High-coverage masks are instead built by unioning several mid donors."""
    rng = np.random.default_rng(seed)
    donors = []
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            obs = f["original_obs_mask"][:]
            land = f["land_mask"][:]
        ocean = land == 0
        ocn = ocean.sum()
        for t in range(obs.shape[0]):
            cloud = (obs[t] == 0) & ocean          # real cloud footprint over ocean
            frac = cloud.sum() / ocn
            if min_cloud_frac <= frac <= max_cloud_frac:
                donors.append(cloud)
        if len(donors) >= max_donors * 3:
            break
    donors = np.array(donors, dtype=bool)
    if len(donors) > max_donors:
        idx = rng.choice(len(donors), size=max_donors, replace=False)
        donors = donors[idx]
    return donors


class RealCloudMaskGenerator:
    def __init__(self, cloud_bank, seed=None, augment=True, ratio_range=None):
        """cloud_bank: (N,H,W) bool real cloud footprints.
        ratio_range: optional (lo,hi) target fraction of eligible to mask."""
        self.bank = cloud_bank
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.ratio_range = ratio_range

    def _aug(self, m):
        """Flip-only augmentation (no shift -> no straight zero-fill seams)."""
        if self.rng.random() < 0.5:
            m = m[:, ::-1]
        if self.rng.random() < 0.5:
            m = m[::-1, :]
        return np.ascontiguousarray(m)

    def _draw(self):
        d = self.bank[int(self.rng.integers(len(self.bank)))]
        return self._aug(d) if self.augment else d

    def generate(self, valid_mask, target_ratio=None):
        """valid_mask: (H,W) 1 where masking is allowed (=obs & ocean).
        Returns artificial_mask (H,W) float32, 1=masked. Guaranteed subset of valid_mask.

        Low ratio -> one real cloud footprint. High ratio -> UNION of several
        mid-size real clouds (mimics a heavy-cloud day of multiple systems),
        avoiding a single rectangular block."""
        vm = np.asarray(valid_mask) > 0
        vcount = int(vm.sum())
        if vcount == 0 or len(self.bank) == 0:
            return np.zeros(vm.shape, dtype=np.float32)
        rr = self.ratio_range
        lo = rr[0] if rr is not None else 0.0
        hi = rr[1] if rr is not None else 1.0
        # per-call random target ratio -> training sees a spread of missing rates
        target = float(target_ratio) if target_ratio is not None else float(self.rng.uniform(lo, hi))
        art = self._draw() & vm
        # union more real clouds until we reach the target coverage
        for _ in range(15):
            if art.sum() / vcount >= target:
                break
            art = art | (self._draw() & vm)
        # if we overshoot (big cloud on small eligible), crop to a contiguous
        # spatial window centred on the cloud so lower targets are reachable too
        if art.sum() / vcount > target * 1.2:
            art = self._crop_to(art, vcount, target)
        return art.astype(np.float32)

    def _crop_to(self, art, vcount, target):
        H, W = art.shape
        best = art
        for _ in range(8):
            cur = best.sum() / vcount
            if cur <= target * 1.15:
                break
            frac = max(0.03, target / cur)
            bh = max(1, int(H * np.sqrt(frac)))
            bw = max(1, int(W * np.sqrt(frac)))
            ys, xs = np.where(art > 0)
            k = int(self.rng.integers(len(ys)))
            cy, cx = int(ys[k]), int(xs[k])                 # centre box on a cloud pixel
            y0 = min(max(0, cy - bh // 2), max(0, H - bh))
            x0 = min(max(0, cx - bw // 2), max(0, W - bw))
            box = np.zeros_like(art)
            box[y0:y0 + bh, x0:x0 + bw] = True
            best = art & box
        return best
