import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v1.evaluate import (
    write_evaluation_result,
)


def test_evaluation_result_is_written_once_with_protocol_identity(tmp_path):
    output = tmp_path / "evaluation.json"
    result = {
        "checkpoint": "/tmp/model.pth",
        "series": [7],
        "metrics": {"mae": 0.2},
    }

    write_evaluation_result(output, result)

    loaded = json.loads(output.read_text())
    assert loaded["protocol_version"] == "gradient-prior-v1"
    assert loaded["series"] == [7]
    assert loaded["metrics"]["mae"] == pytest.approx(0.2)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_evaluation_result(output, result)
