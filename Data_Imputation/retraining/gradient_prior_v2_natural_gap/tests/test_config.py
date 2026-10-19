from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.config import (
    RunConfig,
    TEST_SERIES,
    TRAIN_SERIES,
    VALID_SERIES,
    prepare_new_output_dir,
)


def test_protocol_uses_disjoint_annual_splits(tmp_path):
    assert TRAIN_SERIES == (0, 1, 2, 3, 4, 5, 6)
    assert VALID_SERIES == (7,)
    assert TEST_SERIES == (8,)

    config = RunConfig.for_test(tmp_path)
    config.validate()


def test_validate_rejects_split_overlap(tmp_path):
    config = replace(RunConfig.for_test(tmp_path), valid_series=(6,))

    with pytest.raises(ValueError, match="overlap"):
        config.validate()


def test_prepare_output_accepts_missing_or_empty_directory(tmp_path):
    missing = tmp_path / "missing"
    assert prepare_new_output_dir(missing) == missing
    assert missing.is_dir()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert prepare_new_output_dir(empty) == empty


def test_prepare_output_rejects_nonempty_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "checkpoint.pth").write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_new_output_dir(output)


def test_validate_requires_paths_and_positive_hyperparameters(tmp_path):
    config = replace(RunConfig.for_test(tmp_path), lambda_grad=-0.1)
    with pytest.raises(ValueError, match="weights"):
        config.validate()

    config = replace(RunConfig.for_test(tmp_path), epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        config.validate()

    config = replace(RunConfig.for_test(tmp_path), max_epoch_checkpoints=0)
    with pytest.raises(ValueError, match="checkpoint"):
        config.validate()


def test_resolved_config_is_json_serializable(tmp_path):
    config = replace(RunConfig.for_test(tmp_path), batch_size=2, world_size=4)
    resolved = config.to_dict()

    assert resolved["train_series"] == [0, 1, 2, 3, 4, 5, 6]
    assert resolved["data_dir"] == str(Path(tmp_path / "data").resolve())
    assert resolved["lambda_boundary_transition"] == pytest.approx(0.05)
    assert resolved["lambda_seam"] == pytest.approx(0.02)
    assert resolved["world_size"] == 4
    assert resolved["global_batch_size"] == 8


def test_natural_gap_curriculum_fields_are_serialized(tmp_path):
    config = replace(
        RunConfig.for_test(tmp_path),
        large_mask_ratio_low=0.4,
        large_mask_ratio_high=0.6,
        large_mask_min_observed_fraction=0.7,
    )

    config.validate()

    assert config.to_dict()["large_mask_ratio_high"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "changes",
    [
        {"large_mask_ratio_low": 0.0},
        {"large_mask_ratio_high": 1.1},
        {"large_mask_ratio_low": 0.7, "large_mask_ratio_high": 0.6},
        {"large_mask_min_observed_fraction": 1.1},
        {"lambda_boundary_transition": -0.01},
        {"lambda_seam": -0.01},
    ],
)
def test_natural_gap_curriculum_rejects_invalid_values(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(RunConfig.for_test(tmp_path), **changes).validate()
