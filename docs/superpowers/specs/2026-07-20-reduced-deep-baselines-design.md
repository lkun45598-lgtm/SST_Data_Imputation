# Reduced Deep Baselines Design

## Goal

Create a slightly weaker but still normally trained deep-baseline family for SST gap filling by reducing model capacity, while preserving the input protocol, loss functions, masking, normalization, and validation metrics.

## Scope

Modify the model definitions in `Data_Imputation/baselines/dincae.py` and `Data_Imputation/baselines/deep_models.py`. Add a run tag to the two generic training entry points so reduced checkpoints are written beside, rather than over, the existing full-capacity checkpoints.

The reduced capacity is:

- DINCAE, UNet, and PartialConvUNet: base width `20` instead of `32`.
- ConvLSTM: hidden width `28` instead of `48`.
- LaMaFFC: base width `40` and four residual blocks instead of `64` and six.

No output perturbation, metric manipulation, input masking change, or loss change is introduced. The reduced models must be retrained before use.

## Training and artifact policy

The existing experiment directories and checkpoints remain untouched. Passing `--tag reduced` to pretraining creates `baselines/experiments/{model}_reduced_ostia_pretrain`; passing the same tag to fine-tuning creates `baselines/experiments/{model}_reduced_h12` and automatically loads the matching reduced OSTIA checkpoint.

Five OSTIA jobs will be assigned one per idle GPU (`0` through `4`). GPU `6` is left untouched because it is already occupied. Fine-tuning will use the same allocation after each model's pretraining checkpoint exists.

## Verification

Add a focused regression test that checks the factory builds the intended reduced widths and block count. Run the test before and after the model edit, then run a small CPU forward smoke test for all five factory names. Training logs and checkpoint files are the final runtime verification.
