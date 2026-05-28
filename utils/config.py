from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_SEED = 16


@dataclass
class DatasetConfig:
    name: str = "babyai"
    action_size: int = 7
    max_seq_len: int = 96
    pretrain_size: int = 500
    pretrain_valid_size: int = 100
    train_size: int = 50
    valid_size: int = 100
    inference_size: int = 1000
    min_episode_len: int = 3
    mission_encoding: str = "surface"
    generator_backend: str = "auto"
    seed: int | None = None
    pretrain_envs: tuple[str, ...] = ("BabyAI-GoToObj-v0", "BabyAI-Pickup-v0")
    neupsl_envs: tuple[str, ...] = ("BabyAI-PickupLoc-v0", "BabyAI-OpenDoor-v0")
    action_vocab: tuple[str, ...] = field(
        default_factory=lambda: ("left", "right", "forward", "pickup", "drop", "toggle", "done")
    )

    def __post_init__(self) -> None:
        if self.name != "babyai":
            raise ValueError(f"Unknown dataset: {self.name}")
        if self.mission_encoding not in {"surface", "structured"}:
            raise ValueError("mission_encoding must be 'surface' or 'structured'.")
        if self.generator_backend not in {"auto", "minigrid", "synthetic"}:
            raise ValueError("generator_backend must be 'auto', 'minigrid', or 'synthetic'.")
        self.action_size = len(self.action_vocab)
        if self.seed is None:
            self.seed = DEFAULT_SEED * 10000 + 100 * self.train_size + self.inference_size

    def to_metadata_dict(self) -> dict:
        data = asdict(self)
        data["pretrain_envs"] = list(self.pretrain_envs)
        data["neupsl_envs"] = list(self.neupsl_envs)
        data["action_vocab"] = list(self.action_vocab)
        data["data-source"] = "babyai"
        return data


def dataset_dir(dataset: str = "babyai", data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / f"experiment_{dataset}"


def pretrain_dir(config: DatasetConfig, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return (
        dataset_dir(config.name, data_root)
        / "pretrain"
        / f"size_{config.pretrain_size:04d}-valid_{config.pretrain_valid_size:04d}"
    )


def neupsl_train_dir(config: DatasetConfig, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return (
        dataset_dir(config.name, data_root)
        / "neupsl-train"
        / f"size_{config.train_size:04d}-valid_{config.valid_size:04d}"
    )


def inference_dir(config: DatasetConfig, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return dataset_dir(config.name, data_root) / "inference" / f"size_{config.inference_size:04d}"


def rules_dir(config: DatasetConfig, split: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return dataset_dir(config.name, data_root) / split / "rules"


def experiment_id(config: DatasetConfig) -> str:
    return f"babyai_train_{config.train_size}_infer_{config.inference_size}"


def result_dir(config: DatasetConfig, results_root: str | Path = DEFAULT_RESULTS_ROOT) -> Path:
    return Path(results_root) / experiment_id(config)
