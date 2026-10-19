import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v1.gradient_stats import (
    GradientHistogram,
    compute_monthly_gradient_stats,
    format_month_summary,
    load_gradient_stats,
    strict_adjacent_edge_magnitude,
    strict_central_gradient,
    write_gradient_stats,
)


def test_strict_central_gradient_matches_linear_field():
    yy, xx = np.mgrid[:5, :6]
    field = (2.0 * yy + 1.0 * xx).astype(np.float32)
    observed = np.ones(field.shape, dtype=bool)
    land = np.zeros(field.shape, dtype=bool)

    magnitude, valid = strict_central_gradient(field, observed, land)

    assert valid.sum() == (5 - 2) * (6 - 2)
    assert np.allclose(magnitude[valid], np.sqrt(5.0))
    assert np.all(np.isnan(magnitude[~valid]))


def test_strict_central_gradient_excludes_incomplete_stencil():
    yy, xx = np.mgrid[:5, :6]
    field = (2.0 * yy + xx).astype(np.float32)
    observed = np.ones(field.shape, dtype=bool)
    observed[2, 3] = False
    land = np.zeros(field.shape, dtype=bool)

    _, valid = strict_central_gradient(field, observed, land)

    assert not valid[2, 2]  # right neighbor missing
    assert not valid[2, 3]  # center missing
    assert not valid[2, 4]  # left neighbor missing
    assert not valid[1, 3]  # lower neighbor missing
    assert not valid[3, 3]  # upper neighbor missing
    assert valid[1, 1]


def test_strict_adjacent_edges_require_two_observed_ocean_endpoints():
    field = np.array([[0.0, 1.0, 3.0], [0.0, 2.0, 5.0]], dtype=np.float32)
    observed = np.ones_like(field, dtype=bool)
    observed[0, 1] = False
    land = np.zeros_like(field, dtype=bool)

    values, valid = strict_adjacent_edge_magnitude(field, observed, land)

    assert valid.sum() == 4
    assert np.sort(values).tolist() == pytest.approx([0.0, 2.0, 2.0, 3.0])


def test_histogram_quantiles_are_deterministic_bin_centres():
    histogram = GradientHistogram(bin_width=0.1, max_gradient=1.0)
    histogram.update(np.array([0.01, 0.11, 0.21, 0.31], dtype=np.float32))

    summary = histogram.summary()

    assert summary["count"] == 4
    assert summary["overflow_count"] == 0
    assert summary["p50"] == pytest.approx(0.15)
    assert summary["p90"] == pytest.approx(0.35)


def test_histogram_tracks_overflow_without_using_it_as_threshold():
    histogram = GradientHistogram(bin_width=0.1, max_gradient=1.0)
    histogram.update(np.array([0.2, 1.01, 9.0], dtype=np.float32))

    summary = histogram.summary()

    assert summary["count"] == 3
    assert summary["in_range_count"] == 1
    assert summary["overflow_count"] == 2
    assert summary["p99_9"] == pytest.approx(0.25)


def test_statistics_builder_rejects_nontraining_series(tmp_path):
    with pytest.raises(ValueError, match="Series 0-6"):
        compute_monthly_gradient_stats(tmp_path, series_ids=(0, 7))

    with pytest.raises(ValueError, match="Series 0-6"):
        compute_monthly_gradient_stats(tmp_path, series_ids=(8,))


def test_monthly_payload_separates_central_and_adjacent_edge_distributions(tmp_path):
    payload = {
        "protocol_version": "gradient-prior-v1",
        "units": "K/pixel",
        "series_ids": [0, 1, 2, 3, 4, 5, 6],
        "bin_width": 0.0001,
        "max_gradient": 8.0,
        "inputs": [],
        "months": {
            str(month): {
                "central": {"count": 1, "p99_9": 1.0},
                "edge": {"count": 1, "p99_9": 2.0},
            }
            for month in range(1, 13)
        },
    }
    path = tmp_path / "separate.json"
    write_gradient_stats(payload, path)

    loaded = load_gradient_stats(path)

    assert loaded["months"]["1"]["central"]["p99_9"] == pytest.approx(1.0)
    assert loaded["months"]["1"]["edge"]["p99_9"] == pytest.approx(2.0)


def test_month_summary_formats_nested_distribution_counts_and_thresholds():
    summary = {
        "central": {"count": 101, "p99_9": 1.23456},
        "edge": {"count": 202, "p99_9": 0.98765},
    }

    text = format_month_summary(2, summary)

    assert text == (
        "month=02 central_count=101 edge_count=202 "
        "central_p99.9=1.2346 edge_p99.9=0.9877 K/pixel"
    )


def _complete_payload():
    return {
        "protocol_version": "gradient-prior-v1",
        "units": "K/pixel",
        "series_ids": [0, 1, 2, 3, 4, 5, 6],
        "bin_width": 0.0001,
        "max_gradient": 8.0,
        "inputs": [],
        "months": {
            str(month): {
                "central": {
                    "count": 100,
                    "in_range_count": 100,
                    "overflow_count": 0,
                    "p50": 0.05,
                    "p90": 0.15,
                    "p95": 0.2,
                    "p99": 0.4,
                    "p99_5": 0.55,
                    "p99_9": 1.0 + month / 100.0,
                },
                "edge": {
                    "count": 100,
                    "in_range_count": 100,
                    "overflow_count": 0,
                    "p50": 0.05,
                    "p90": 0.15,
                    "p95": 0.2,
                    "p99": 0.4,
                    "p99_5": 0.55,
                    "p99_9": 2.0 + month / 100.0,
                },
            }
            for month in range(1, 13)
        },
    }


def test_stats_round_trip_validates_digest_and_complete_months(tmp_path):
    path = tmp_path / "stats.json"
    digest = write_gradient_stats(_complete_payload(), path)

    loaded = load_gradient_stats(path)

    assert loaded["sha256"] == digest
    assert loaded["months"]["12"]["central"]["p99_9"] == pytest.approx(1.12)
    assert loaded["months"]["12"]["edge"]["p99_9"] == pytest.approx(2.12)


def test_stats_loader_rejects_tampering_or_missing_month(tmp_path):
    path = tmp_path / "stats.json"
    write_gradient_stats(_complete_payload(), path)
    data = json.loads(path.read_text())
    data["months"]["1"]["central"]["p99_9"] = 99.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        load_gradient_stats(path)

    path = tmp_path / "incomplete.json"
    payload = _complete_payload()
    del payload["months"]["12"]
    write_gradient_stats(payload, path)
    with pytest.raises(ValueError, match="12 months"):
        load_gradient_stats(path)
