import json
from dataclasses import replace
import math
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Data_Imputation.retraining.gradient_prior_v2_natural_gap.config import RunConfig
from Data_Imputation.retraining.gradient_prior_v2_natural_gap.train import (
    _all_reduce_scalar_mapping,
    _barrier,
    DistributedContext,
    build_checkpoint_metadata,
    build_data_loaders,
    build_datasets,
    evenly_spaced_indices,
    monthly_proxy_highpass_for,
    monthly_upper_for,
    prune_epoch_checkpoints,
    rank_strided_indices,
    require_finite_loss,
    run_epoch,
    write_run_manifests,
)


def _stats():
    return {
        "sha256": "a" * 64,
        "months": {
            str(month): {
                "central": {"p99_9": month / 20.0},
                "edge": {"p99_9": month / 10.0},
                "proxy_highpass": {"p99_9": month / 5.0},
            }
            for month in range(1, 13)
        },
    }


def test_monthly_upper_maps_each_batch_month_to_physical_threshold():
    result = monthly_upper_for(torch.tensor([1, 12, 6]), _stats(), torch.device("cpu"))
    assert torch.allclose(result, torch.tensor([0.1, 1.2, 0.6]))


def test_monthly_upper_rejects_invalid_month():
    with pytest.raises(ValueError, match="month"):
        monthly_upper_for(torch.tensor([0]), _stats(), torch.device("cpu"))


def test_monthly_proxy_highpass_maps_each_batch_month_to_k_threshold():
    result = monthly_proxy_highpass_for(
        torch.tensor([1, 12]), _stats(), torch.device("cpu")
    )
    assert torch.allclose(result, torch.tensor([0.2, 2.4]))


def test_require_finite_loss_names_the_bad_component():
    with pytest.raises(FloatingPointError, match="seam"):
        require_finite_loss(
            torch.tensor(1.0),
            {"mse": torch.tensor(0.1), "seam": torch.tensor(float("nan"))},
        )


def test_evenly_spaced_indices_cover_both_ends_without_duplicates():
    assert evenly_spaced_indices(10, None) is None
    assert evenly_spaced_indices(10, 1) == [0]
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(3, 10) is None


def test_rank_strided_validation_indices_cover_once_without_padding():
    shards = [rank_strided_indices(11, rank, 4) for rank in range(4)]

    assert shards == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]]
    assert sorted(index for shard in shards for index in shard) == list(range(11))


def test_distributed_barrier_uses_the_rank_local_device(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "Data_Imputation.retraining.gradient_prior_v2_natural_gap.train.dist.barrier",
        lambda **kwargs: calls.append(kwargs),
    )

    _barrier(DistributedContext(rank=2, local_rank=2, world_size=4))

    assert calls == [{"device_ids": [2]}]


def test_training_loader_restarts_workers_to_receive_each_epoch(tmp_path):
    config = replace(RunConfig.for_test(tmp_path), num_workers=2, world_size=1)
    train_loader, valid_loaders, train_sampler = build_data_loaders(
        _OneSampleDataset(),
        {"standard": _OneSampleDataset(), "large": _OneSampleDataset()},
        config,
        torch.device("cpu"),
        rank=0,
        world_size=1,
    )

    assert train_sampler is None
    assert train_loader.persistent_workers is False
    assert set(valid_loaders) == {"standard", "large"}
    assert all(loader.persistent_workers is True for loader in valid_loaders.values())


def test_scalar_mapping_reduction_uses_one_collective_in_stable_key_order():
    calls = []

    def fake_all_reduce(values):
        calls.append(values.clone())
        values.mul_(4)

    reduced = _all_reduce_scalar_mapping(
        {"z": 2.0, "a": 1.0}, torch.device("cpu"), all_reduce=fake_all_reduce
    )

    assert len(calls) == 1
    assert calls[0].tolist() == [1.0, 2.0]
    assert reduced == {"a": 4.0, "z": 8.0}


def test_build_datasets_balances_standard_and_large_training_streams(tmp_path, monkeypatch):
    calls = []

    class FakeDataset(Dataset):
        def __init__(self, _data_dir, **kwargs):
            self.kwargs = kwargs
            self.epoch = 0
            calls.append(self)

        def __len__(self):
            return 10

        def __getitem__(self, index):
            return index

        def set_epoch(self, epoch):
            self.epoch = epoch

    monkeypatch.setattr(
        "Data_Imputation.retraining.gradient_prior_v2_natural_gap.train.DeploymentMatchedJAXADataset",
        FakeDataset,
    )
    monkeypatch.setattr(
        "Data_Imputation.retraining.gradient_prior_v2_natural_gap.train.high_observation_global_indices",
        lambda *_args, **_kwargs: torch.tensor([0]).numpy(),
    )
    config = RunConfig.for_test(tmp_path)
    train, validations = build_datasets(
        config, cloud_bank=torch.zeros((1, 2, 2)).numpy()
    )

    assert calls[0].kwargs["series_ids"] == (0, 1, 2, 3, 4, 5, 6)
    assert calls[0].kwargs["mask_ratio_range"] == (0.2, 0.3)
    assert calls[1].kwargs["mask_ratio_range"] == (0.4, 0.6)
    assert calls[1].kwargs["active_global_indices"].tolist() == [0]
    assert set(validations) == {"standard", "large"}
    assert calls[2].kwargs["series_ids"] == (7,)
    assert calls[3].kwargs["series_ids"] == (7,)
    assert calls[3].kwargs["mask_ratio_range"] == (0.4, 0.6)
    assert all(8 not in call.kwargs["series_ids"] for call in calls)


def test_checkpoint_metadata_records_split_stats_and_weights(tmp_path):
    config = replace(RunConfig.for_test(tmp_path), batch_size=1, world_size=4)
    metadata = build_checkpoint_metadata(
        config,
        gradient_stats=_stats(),
        epoch=2,
        validation_metrics={"standard": {"mae": 0.12}, "large": {"mae": 0.2}},
    )

    assert metadata["epoch"] == 2
    assert metadata["train_series"] == [0, 1, 2, 3, 4, 5, 6]
    assert metadata["valid_series"] == [7]
    assert metadata["test_series"] == [8]
    assert metadata["gradient_stats_sha256"] == "a" * 64
    assert metadata["loss_weights"]["boundary_transition"] == pytest.approx(0.05)
    assert metadata["loss_weights"]["seam"] == pytest.approx(0.02)
    assert metadata["validation_metrics"]["standard"]["mae"] == pytest.approx(0.12)
    assert metadata["validation_metrics"]["large"]["mae"] == pytest.approx(0.2)
    assert metadata["distributed"] == {
        "world_size": 4,
        "per_rank_batch_size": 1,
        "global_batch_size": 4,
    }


def test_prune_epoch_checkpoints_keeps_latest_while_best_hardlink_survives(tmp_path):
    checkpoints = []
    for epoch in range(1, 4):
        path = tmp_path / f"checkpoint_epoch_{epoch:03d}.pth"
        path.write_bytes(f"epoch-{epoch}".encode())
        checkpoints.append(path)
    best = tmp_path / "best_model.pth"
    best.hardlink_to(checkpoints[0])

    removed = prune_epoch_checkpoints(tmp_path, keep=1)

    assert removed == checkpoints[:2]
    assert not checkpoints[0].exists()
    assert checkpoints[2].read_bytes() == b"epoch-3"
    assert best.read_bytes() == b"epoch-1"


def test_prune_epoch_checkpoints_can_clear_slots_before_next_save(tmp_path):
    epoch = tmp_path / "checkpoint_epoch_001.pth"
    epoch.write_bytes(b"best-state")
    best = tmp_path / "best_model.pth"
    best.hardlink_to(epoch)

    removed = prune_epoch_checkpoints(tmp_path, keep=0)

    assert removed == [epoch]
    assert not epoch.exists()
    assert best.read_bytes() == b"best-state"


def test_write_manifests_refuses_existing_artifacts_and_records_source(tmp_path):
    checkpoint = tmp_path / "pretrained.pth"
    checkpoint.write_bytes(b"checkpoint")
    stats_path = tmp_path / "gradient_stats.json"
    stats_path.write_text("{}")
    config = RunConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        pretrained_path=checkpoint,
        gradient_stats_path=stats_path,
    )

    output = write_run_manifests(config, gradient_stats=_stats())

    resolved = json.loads((output / "resolved_config.json").read_text())
    source = json.loads((output / "source_manifest.json").read_text())
    assert resolved["train_series"] == [0, 1, 2, 3, 4, 5, 6]
    assert source["pretrained_checkpoint"]["size_bytes"] == len(b"checkpoint")
    assert source["gradient_stats_sha256"] == "a" * 64

    with pytest.raises(FileExistsError, match="non-empty"):
        write_run_manifests(config, gradient_stats=_stats())


class _OneSampleDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        height = width = 9
        natural = torch.zeros((height, width), dtype=torch.bool)
        natural[2:7, 2:7] = True
        artificial = torch.zeros_like(natural)
        artificial[0:2, 0:2] = True
        observed = ~natural
        mask_sequence = torch.zeros((30, height, width), dtype=torch.bool)
        mask_sequence[-1] = natural | artificial
        return {
            "input_sst_seq": torch.zeros((30, height, width)),
            "mask_seq": mask_sequence,
            "target_sst": torch.ones((height, width)) * 0.1,
            "target_proxy_sst": torch.arange(width).repeat(height, 1).float() * 0.02,
            "natural_mask": natural,
            "artificial_mask": artificial,
            "original_obs_mask": observed,
            "land_mask": torch.zeros_like(natural),
            "month": 1,
        }


class _TinyReconstructor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, sst_sequence, _mask_sequence):
        return sst_sequence[:, -1:] + self.scale


def test_run_epoch_performs_finite_train_and_validation_steps(tmp_path):
    config = replace(
        RunConfig.for_test(tmp_path),
        norm_std=2.0,
    )
    model = _TinyReconstructor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loader = torch.utils.data.DataLoader(_OneSampleDataset(), batch_size=1)
    before = model.scale.detach().clone()

    train_metrics = run_epoch(
        model,
        loader,
        config,
        _stats(),
        torch.device("cpu"),
        optimizer=optimizer,
    )
    validation_metrics = run_epoch(
        model,
        loader,
        config,
        _stats(),
        torch.device("cpu"),
        optimizer=None,
    )

    assert model.scale.detach() != before
    assert math.isfinite(train_metrics["total_loss"])
    assert math.isfinite(validation_metrics["total_loss"])
    assert train_metrics["valid_counts"]["mse"] == 4
    assert validation_metrics["mae"] is not None
