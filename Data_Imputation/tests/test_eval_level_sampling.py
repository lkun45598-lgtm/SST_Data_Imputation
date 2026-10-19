import sys
from pathlib import Path

import numpy as np
import pytest


DATA_IMPUTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_IMPUTATION_DIR))

from baselines import eval_all_methods
from baselines.optimal_interpolation import oi_reconstruct


def test_single_level_sampling_matches_the_same_level_in_full_run():
    candidates = np.arange(300)
    full = {
        name: idxs
        for name, _, idxs in eval_all_methods.sample_level_indices(
            candidates, nsamples=219, requested_level=None, seed=0
        )
    }

    for level_name in ("low", "mid", "high"):
        single = eval_all_methods.sample_level_indices(
            candidates, nsamples=219, requested_level=level_name, seed=0
        )

        assert len(single) == 1
        assert single[0][0] == level_name
        assert np.array_equal(single[0][2], full[level_name])


def test_ssim_result_fields_include_candidate_coverage():
    fields = eval_all_methods.ssim_result_fields(0.75, 30, 40)

    assert fields == {
        "ssim": 0.75,
        "n_ssim": 30,
        "n_ssim_candidates": 40,
        "ssim_coverage": 0.75,
    }

    empty = eval_all_methods.ssim_result_fields(np.nan, 0, 0)
    assert np.isnan(empty["ssim_coverage"])


def test_ssim_only_record_contains_no_non_ssim_metrics():
    record = eval_all_methods.ssim_only_record(
        method="fno",
        level="low",
        target_ratio=0.30,
        idx=42,
        n_withheld=1234,
        data_range=5.5,
        ssim_value=0.81,
        n_ssim=800,
        n_ssim_candidates=1000,
    )

    assert record == {
        "method": "fno",
        "level": "low",
        "target_ratio": 0.30,
        "idx": 42,
        "n": 1234,
        "ssim_data_range": 5.5,
        "ssim": 0.81,
        "n_ssim": 800,
        "n_ssim_candidates": 1000,
        "ssim_coverage": 0.8,
    }
    assert "mae" not in record
    assert "rmse" not in record
    assert "gradmae" not in record


def test_ssim_prediction_preserves_unwithheld_reference_context():
    reference = np.arange(16, dtype=np.float32).reshape(4, 4)
    prediction = reference + 10.0
    evaluation_mask = np.zeros(reference.shape, dtype=bool)
    evaluation_mask[1:3, 1:3] = True

    composed = eval_all_methods.compose_observation_preserving_prediction(
        prediction, reference, evaluation_mask
    )

    assert np.array_equal(composed[evaluation_mask], prediction[evaluation_mask])
    assert np.array_equal(composed[~evaluation_mask], reference[~evaluation_mask])


def test_ssim_only_output_is_isolated_and_refuses_overwrite(tmp_path):
    output = eval_all_methods.resolve_output_path(
        "fno_low.npz",
        ssim_only=True,
        cache_dir=tmp_path,
    )

    assert output == tmp_path / "ssim_standard_v2_obspreserve" / "fno_low.npz"
    output.parent.mkdir(parents=True)
    output.touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        eval_all_methods.resolve_output_path(
            "fno_low.npz",
            ssim_only=True,
            cache_dir=tmp_path,
        )

    extensionless_output = eval_all_methods.resolve_output_path(
        "dincae_low",
        ssim_only=True,
        cache_dir=tmp_path,
    )
    assert extensionless_output == (
        tmp_path / "ssim_standard_v2_obspreserve" / "dincae_low.npz"
    )
    extensionless_output.touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        eval_all_methods.resolve_output_path(
            "dincae_low",
            ssim_only=True,
            cache_dir=tmp_path,
        )


def test_oi_sparse_source_fallback_does_not_return_withheld_truth():
    y, x = np.mgrid[:8, :8]
    truth = (290.0 + y + 0.1 * x).astype(np.float32)
    mask = np.ones(truth.shape, dtype=np.uint8)
    mask[0, :3] = 0
    valid_ocean = np.ones(truth.shape, dtype=np.uint8)

    prediction, _ = oi_reconstruct(
        truth, mask, valid_ocean, k=30,
    )

    withheld = mask == 1
    assert np.all(np.isfinite(prediction[withheld]))
    assert not np.array_equal(prediction[withheld], truth[withheld])
