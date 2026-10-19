import ast
from pathlib import Path


DATA_IMPUTATION = Path(__file__).resolve().parents[1]
MAIN_EVAL = DATA_IMPUTATION / "baselines/eval_all_methods.py"
PROXY_EVAL = DATA_IMPUTATION / "visualization/paper_figs/eval_proxy_ablation.py"


def literal_assignments(path):
    tree = ast.parse(path.read_text())
    assignments = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                assignments[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return assignments


def test_proxy_ablation_uses_main_seed_levels_and_sample_count():
    main_config = literal_assignments(MAIN_EVAL)
    proxy_config = literal_assignments(PROXY_EVAL)

    assert proxy_config["SEED"] == main_config["SEED"]
    assert proxy_config["NUM_SAMPLES"] == main_config["NUM_SAMPLES_PER_LEVEL"]
    assert proxy_config["LEVELS"] == main_config["MASK_LEVELS"]


def test_proxy_ablation_uses_main_reference_fields():
    source = PROXY_EVAL.read_text()

    assert "gt_h5 = REPO / 'experiments/hourly_data/h12/jaxa_knn_filled_08.h5'" in source
    assert "(obs_all[i] * ocean).sum() > 4000" in source
