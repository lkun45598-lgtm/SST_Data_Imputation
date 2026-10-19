import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.realcloud_diagnostics import (
    ExactValueAccumulator,
    write_realcloud_diagnostics,
)


def test_exact_value_accumulator_uses_all_edges_not_framewise_quantiles():
    accumulator = ExactValueAccumulator()
    accumulator.add(torch.tensor([0.0, 1.0]))
    accumulator.add(torch.tensor([2.0, 3.0]))

    summary = accumulator.summary()

    assert summary["count"] == 4
    assert summary["p50"] == pytest.approx(1.5)
    assert summary["p99_9"] == pytest.approx(2.997)


def test_realcloud_diagnostics_writer_refuses_to_overwrite(tmp_path):
    output = tmp_path / "series8.json"
    document = {
        "series": [8],
        "artificial_masking": False,
        "natural_interior_adjacent_difference_K_per_pixel": {"count": 0},
        "deployment_seam_adjacent_difference_K_per_pixel": {"count": 0},
        "deployment_seam_exceedance_rate": None,
        "natural_local_outlier_rate": None,
    }

    write_realcloud_diagnostics(output, document)

    loaded = json.loads(output.read_text())
    assert loaded["protocol_version"] == "gradient-prior-v2-natural-gap"
    assert loaded["artificial_masking"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        write_realcloud_diagnostics(output, document)
