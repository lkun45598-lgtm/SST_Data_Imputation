"""Configuration and artifact-isolation rules for gradient-prior retraining."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRAIN_SERIES = (0, 1, 2, 3, 4, 5, 6)
VALID_SERIES = (7,)
TEST_SERIES = (8,)


def prepare_new_output_dir(path: Path | str) -> Path:
    """Create an output directory, refusing to reuse one containing artifacts."""
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


@dataclass(frozen=True)
class RunConfig:
    data_dir: Path
    output_dir: Path
    pretrained_path: Path
    gradient_stats_path: Path
    experiment_name: str = "h12_pilot"
    hour: int = 12
    train_series: tuple[int, ...] = TRAIN_SERIES
    valid_series: tuple[int, ...] = VALID_SERIES
    test_series: tuple[int, ...] = TEST_SERIES
    epochs: int = 5
    batch_size: int = 1
    world_size: int = 1
    num_workers: int = 0
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    mask_ratio_low: float = 0.2
    mask_ratio_high: float = 0.3
    train_seed: int = 42
    valid_seed: int = 123
    lambda_grad: float = 0.2
    lambda_boundary_transition: float = 0.05
    lambda_seam: float = 0.02
    large_mask_ratio_low: float = 0.4
    large_mask_ratio_high: float = 0.6
    large_mask_min_observed_fraction: float = 0.7
    norm_mean: float = 299.9221
    norm_std: float = 2.6919
    device: str = "cuda:0"
    train_limit: int | None = None
    valid_limit: int | None = None
    max_epoch_checkpoints: int = 1

    @classmethod
    def for_test(cls, root: Path | str) -> "RunConfig":
        root = Path(root)
        return cls(
            data_dir=root / "data",
            output_dir=root / "output",
            pretrained_path=root / "pretrained.pth",
            gradient_stats_path=root / "gradient_stats.json",
        )

    def validate(self, require_inputs: bool = False) -> None:
        splits = [set(self.train_series), set(self.valid_series), set(self.test_series)]
        if any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("train, validation, and test series overlap")
        if tuple(self.train_series) != TRAIN_SERIES:
            raise ValueError(f"gradient-prior v1 requires training series {TRAIN_SERIES}")
        if tuple(self.valid_series) != VALID_SERIES or tuple(self.test_series) != TEST_SERIES:
            raise ValueError("gradient-prior v1 requires Series 7 validation and Series 8 test")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.max_epoch_checkpoints <= 0:
            raise ValueError("max_epoch_checkpoints must be positive")
        if self.batch_size <= 0 or self.num_workers < 0 or self.world_size <= 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer hyperparameters are invalid")
        if not 0 < self.mask_ratio_low <= self.mask_ratio_high <= 1:
            raise ValueError("mask ratio range must lie in (0, 1]")
        if not 0 < self.large_mask_ratio_low <= self.large_mask_ratio_high <= 1:
            raise ValueError("large mask ratio range must lie in (0, 1]")
        if not 0 < self.large_mask_min_observed_fraction <= 1:
            raise ValueError("large mask coverage threshold must lie in (0, 1]")
        if self.norm_std <= 0:
            raise ValueError("norm_std must be positive")
        weights = (
            self.lambda_grad,
            self.lambda_boundary_transition,
            self.lambda_seam,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if require_inputs:
            for label, path in (
                ("data directory", self.data_dir),
                ("pretrained checkpoint", self.pretrained_path),
                ("gradient statistics", self.gradient_stats_path),
            ):
                if not Path(path).exists():
                    raise FileNotFoundError(f"missing {label}: {path}")

    def to_dict(self) -> dict[str, Any]:
        resolved = asdict(self)
        for key in ("data_dir", "output_dir", "pretrained_path", "gradient_stats_path"):
            resolved[key] = str(Path(resolved[key]).resolve())
        for key in ("train_series", "valid_series", "test_series"):
            resolved[key] = list(resolved[key])
        resolved["global_batch_size"] = self.batch_size * self.world_size
        return resolved
