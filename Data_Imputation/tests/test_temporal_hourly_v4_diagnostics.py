from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Data_Imputation.eval_diag.eval_temporal_hourly_v4_proxy import evaluate_withheld
from Data_Imputation.visualization.paper_figs.fig_temporal_hourly_v4_4panel import (
    render_four_panel,
)


def test_evaluation_masks_metrics_to_synthetically_withheld_raw_ocean_pixels():
    y, x = np.mgrid[:9, :9]
    reference = (20.0 + y * 0.1 + x * 0.2).astype(np.float32)
    prediction = reference.copy()
    prediction[3:6, 3:6] += 1.0
    prediction[0, 0] = 999.0
    withheld = np.zeros((9, 9), dtype=bool)
    withheld[3:6, 3:6] = True
    land = np.zeros((9, 9), dtype=bool)
    land[0, 0] = True

    metrics = evaluate_withheld(reference, prediction, withheld, land)

    assert metrics["pixel_count"] == int(withheld.sum())
    assert metrics["mae"] == 1.0
    assert {
        "mae",
        "gradient_mae",
        "gradient_correlation",
        "ssim",
    } <= set(metrics)


def test_four_panel_plot_requires_raw_temporal_gaussian_and_final_fields(tmp_path):
    data = np.arange(64, dtype=np.float32).reshape(8, 8) + 273.15
    output = render_four_panel(
        {
            "raw": data,
            "temporal": data + 0.1,
            "gaussian": data + 0.2,
            "final": data + 0.3,
            "land_mask": np.zeros((8, 8), dtype=bool),
            "latitude": np.linspace(20.0, 21.0, 8),
            "longitude": np.linspace(120.0, 121.0, 8),
        },
        tmp_path / "frame.png",
    )

    assert output.exists()
    assert output.stat().st_size > 0
