import ast
from pathlib import Path


DATA_IMPUTATION = Path(__file__).resolve().parents[1]
MAIN_EVAL = DATA_IMPUTATION / "baselines/eval_all_methods.py"
ABLATION_EVAL = DATA_IMPUTATION / "visualization/paper_figs/gen_ablation_table.py"


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


def test_model_ablation_uses_main_evaluation_protocol():
    main_config = literal_assignments(MAIN_EVAL)
    ablation_config = literal_assignments(ABLATION_EVAL)

    assert ablation_config["SEED"] == main_config["SEED"]
    assert ablation_config["NUM_SMALL"] == main_config["NUM_SAMPLES_PER_LEVEL"]
    assert ablation_config["MASK_LEVELS"] == main_config["MASK_LEVELS"]
    assert "(obs_all[i] * ocean).sum() > 4000" in ABLATION_EVAL.read_text()
