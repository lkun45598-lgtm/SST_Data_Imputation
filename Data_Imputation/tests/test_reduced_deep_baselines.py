import sys
from pathlib import Path

import torch


DATA_IMPUTATION = Path(__file__).resolve().parents[1]
if str(DATA_IMPUTATION) not in sys.path:
    sys.path.insert(0, str(DATA_IMPUTATION))

from baselines.deep_models import build_model
from baselines.dincae import DINCAE


def test_reduced_factory_capacities():
    dincae = build_model("dincae", out_size=(32, 32))
    unet = build_model("unet", out_size=(32, 32))
    convlstm = build_model("convlstm", out_size=(32, 32))
    lama = build_model("lama", out_size=(32, 32))
    partialconv = build_model("partialconv", out_size=(32, 32))

    default_dincae = DINCAE(in_frames=30, out_size=(32, 32))
    assert default_dincae.enc[0][0].out_channels == 16
    assert len(default_dincae.enc) == 3
    assert dincae.enc[0][0].out_channels == 16
    assert len(dincae.enc) == 3
    assert unet.enc[0][0].out_channels == 16
    assert len(unet.enc) == 3
    assert convlstm.frame_enc[0].out_channels == 8
    assert convlstm.frame_enc[2].out_channels == 8
    assert convlstm.frame_enc[4].out_channels == 8
    assert convlstm.cell.hid == 8
    assert convlstm.dec[3].out_channels == 8
    assert convlstm.frame_enc(torch.zeros(1, 2, 32, 32)).shape == (1, 8, 4, 4)
    assert lama.stem[0].out_channels == 40
    assert len(lama.blocks) == 4
    assert partialconv.d1.pc.conv.out_channels == 20


def test_reduced_models_forward_shape_and_finiteness():
    x = torch.zeros(1, 30, 32, 32)
    mask = torch.zeros_like(x)
    for name in ("dincae", "unet", "convlstm", "lama", "partialconv"):
        model = build_model(name, out_size=(32, 32)).eval()
        with torch.no_grad():
            mean, _ = model(x, mask)
        assert mean.shape == (1, 1, 32, 32)
        assert torch.isfinite(mean).all()
