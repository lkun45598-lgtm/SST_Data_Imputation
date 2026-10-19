from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.visualize_realcloud_style import (
    _predict_realcloud_sample,
)


class _ConstantModel(torch.nn.Module):
    def forward(self, sst_sequence, _mask_sequence):
        return torch.full_like(sst_sequence[:, -1:], 9.0)


def test_realcloud_prediction_uses_local_boundary_transition():
    natural = np.zeros((5, 5), dtype=bool)
    natural[1:4, 1:4] = True
    input_sst = np.full((30, 5, 5), 3.0, dtype=np.float32)
    input_sst[-1, natural] = 0.0
    sample = {
        "input_sst_seq": input_sst,
        "mask_seq": np.zeros((30, 5, 5), dtype=np.float32),
        "natural_mask": natural,
        "land_mask": np.zeros((5, 5), dtype=bool),
        "target_proxy": np.full((5, 5), 3.0, dtype=np.float32),
    }

    result = _predict_realcloud_sample(
        _ConstantModel(), sample, device=torch.device("cpu"), mean=0.0, std=1.0
    )

    assert result[0, 0] == 3.0
    assert result[1, 2] == 6.0
    assert result[2, 2] == 9.0
