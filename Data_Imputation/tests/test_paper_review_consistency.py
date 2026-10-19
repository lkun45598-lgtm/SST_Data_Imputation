"""Regression checks for reviewer consistency items 3--5."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "_overleaf_review" / "main.tex"
FIG2_UPDATER = (
    ROOT
    / "Data_Imputation"
    / "visualization"
    / "paper_figs"
    / "update_fig2_proxy_label.py"
)
FIG4_SOURCE = (
    ROOT
    / "Data_Imputation"
    / "visualization"
    / "paper_figs"
    / "fig4_reconstruction.py"
)


def test_fig2_updater_uses_unambiguous_historical_range():
    source = FIG2_UPDATER.read_text(encoding="utf-8")
    assert 'PROXY_HEADER = "Historical proxy fields (t-6 to t-1)"' in source
    assert "W = 6" not in source


def test_fig4_has_no_orphaned_comparison_rectangle():
    source = FIG4_SOURCE.read_text(encoding="utf-8")
    assert "find_best_zoom" not in source
    assert "Rectangle(" not in source
    assert "ZOOM_SIZE_DEG" not in source
    assert (
        '"Reconstruction Comparison at Three Nominal Extra-mask Levels"'
        in source
    )


def test_realized_fraction_text_closes_nominal_to_bin_counts():
    source = PAPER.read_text(encoding="utf-8")
    assert "mean (range) of \\(r\\)" in source
    assert "143/60/16" in source
    assert "2/174/43" in source
    assert "0/3/216" in source
    assert "145/237/275" in source

