"""Provenance codes and in-memory state for one causally reconstructed frame."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


LAND = np.uint8(0)
RAW = np.uint8(1)
TEMPORAL = np.uint8(2)
KNN = np.uint8(3)
UNRESOLVED = np.uint8(4)


@dataclass(frozen=True)
class ProxyFrame:
    """A completed frame retained only in the causal 72-hour rolling cache."""

    timestamp: datetime
    values: np.ndarray
    source_code: np.ndarray
    confidence: np.ndarray
    temporal_fill_mask: np.ndarray
    knn_fill_mask: np.ndarray
    land_mask: np.ndarray

