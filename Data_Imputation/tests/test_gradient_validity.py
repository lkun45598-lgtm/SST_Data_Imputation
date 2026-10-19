import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.gradient_masks import (
    central_gradient_magnitude,
    torch_gradient_edge_masks,
    torch_boundary_edge_masks,
)


def test_gradient_edges_require_both_reference_endpoints_and_supervision():
    valid_ref = torch.tensor([[[1, 1, 0], [1, 0, 1]]], dtype=torch.float32)
    supervised = torch.tensor([[[1, 0, 0], [0, 1, 1]]], dtype=torch.float32)

    mask_y, mask_x = torch_gradient_edge_masks(valid_ref, supervised)

    assert mask_y[0, 0, 0, 0].item() == 1.0
    assert mask_y.sum().item() == 1.0
    assert mask_x[0, 0, 0, 0].item() == 1.0
    assert mask_x.sum().item() == 1.0


def test_boundary_edges_require_both_reference_endpoints():
    artificial = torch.tensor([[[0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    valid_ref = torch.tensor([[[1, 0, 0], [1, 0, 1]]], dtype=torch.float32)

    mask_y, mask_x = torch_boundary_edge_masks(artificial, valid_ref)

    assert mask_y.sum().item() == 0.0
    assert mask_x.sum().item() == 0.0


def test_central_gradient_requires_complete_four_neighborhood():
    field = np.arange(9, dtype=np.float32).reshape(3, 3)
    valid_ref = np.ones((3, 3), dtype=bool)
    valid_ref[0, 1] = False

    grad, valid = central_gradient_magnitude(field, valid_ref)

    assert not valid[1, 1]
    assert np.isnan(grad[1, 1])
    assert valid.sum() == 0


def test_central_gradient_has_no_one_sided_boundary_fallback():
    field = np.ones((2, 2), dtype=np.float32)
    grad, valid = central_gradient_magnitude(field, np.ones((2, 2), dtype=bool))

    assert not valid.any()
    assert np.isnan(grad).all()
