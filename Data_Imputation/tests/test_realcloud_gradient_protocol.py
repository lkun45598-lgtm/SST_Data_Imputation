import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "visualization/paper_figs/fig_realcloud_gradient.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("fig_realcloud_gradient", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_gaussian_option_skips_fno_postprocessing(monkeypatch):
    module = load_script()
    calls = []

    def fake_model(*args, **kwargs):
        return np.full((8, 8), 301.0, dtype=np.float32)

    def fake_gaussian(*args, **kwargs):
        calls.append(1)
        return np.full((8, 8), 999.0, dtype=np.float32)

    monkeypatch.setattr(module, "run_torch_model", fake_model)
    monkeypatch.setattr(module, "gauss_filter", fake_gaussian)
    sst_all = np.full((30, 8, 8), 300.0, dtype=np.float32)
    obs_all = np.ones((30, 8, 8), dtype=np.uint8)
    obs_all[-1, :4, :] = 0
    miss_all = np.zeros_like(obs_all)
    land = np.zeros((8, 8), dtype=np.uint8)
    models = {name: object() for name in ("fno", "dincae", "unet", "convlstm")}

    preds, cloud, _ = module.reconstruct_all(
        29, sst_all, obs_all, miss_all, land, land == 0, models, None,
        apply_gaussian=False,
    )

    assert calls == []
    assert np.all(preds["fno"][cloud > 0] == 301.0)
