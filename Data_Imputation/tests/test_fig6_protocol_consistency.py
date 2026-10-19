import ast
from pathlib import Path


DATA_IMPUTATION = Path(__file__).resolve().parents[1]
MAIN_EVAL = DATA_IMPUTATION / "baselines/eval_all_methods.py"
FIG6_EVAL = DATA_IMPUTATION / "visualization/paper_figs/gen_fig6_stats.py"


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
        elif isinstance(target, ast.Tuple):
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for element, value in zip(target.elts, values):
                if isinstance(element, ast.Name):
                    assignments[element.id] = value
    return assignments


def test_fig6_uses_main_evaluation_seed_and_mask_levels():
    main_config = literal_assignments(MAIN_EVAL)
    fig6_config = literal_assignments(FIG6_EVAL)

    assert fig6_config["SEED"] == main_config["SEED"]
    assert fig6_config["NUM_SAMPLES_PER_LEVEL"] == main_config["NUM_SAMPLES_PER_LEVEL"]
    assert fig6_config["MASK_LEVELS"] == main_config["MASK_LEVELS"]
    assert fig6_config["NORM_MEAN"] == main_config["FNO_MEAN"]
    assert fig6_config["NORM_STD"] == main_config["FNO_STD"]


def test_fig6_uses_main_candidate_filter_and_no_output_smoothing():
    source = FIG6_EVAL.read_text()

    assert "(obs_all[i] * ocean).sum() > 4000" in source
    assert "pixel_rng = np.random.default_rng(" in source
    assert "pick = pixel_rng.choice(" in source
    assert "pred = gauss_filter(" not in source
