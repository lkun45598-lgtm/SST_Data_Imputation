import ast
from pathlib import Path


DATA_IMPUTATION = Path(__file__).resolve().parents[1]
MAIN_EVAL = DATA_IMPUTATION / "baselines/eval_all_methods.py"
FIG7_EVAL = DATA_IMPUTATION / "visualization/paper_figs/gen_fig7_stats.py"


def literal_assignments(path):
    tree = ast.parse(path.read_text())
    assignments = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return assignments


def test_fig7_uses_latest_models_and_main_mask_protocol():
    main_config = literal_assignments(MAIN_EVAL)
    fig7_config = literal_assignments(FIG7_EVAL)
    source = FIG7_EVAL.read_text()

    assert fig7_config.get("SERIES_TO_USE") == main_config["SERIES_ID"]
    assert fig7_config.get("SEED") == main_config["SEED"]
    assert fig7_config.get("MASK_LEVELS") == main_config["MASK_LEVELS"]
    assert "jaxa_finetune_h{hour:02d}_realmask_clean/best_model.pth" in source
    assert "target_ratio=level" in source
    assert "pred = gauss_filter(" not in source
    assert "gaussian_filter" not in source


def test_fig7_diurnal_cycle_is_inferred_from_latest_hourly_models():
    source = FIG7_EVAL.read_text()

    assert "def compute_diurnal_for_hour(" in source
    assert "RECON_NC_ROOT" not in source
    assert "netCDF4" not in source
