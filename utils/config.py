from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_SEED = 16


@dataclass
class DatasetConfig:
    name: str = "mnist-1"
    num_digits: int = 1
    class_size: int = 10
    pretrain_size: int = 20000
    pretrain_valid_size: int = 5000
    pretrain_overlap: float = 0.4
    train_size: int = 1000
    valid_size: int = 1000
    inference_size: int = 5000
    overlap: float = 0.0
    seed: int | None = None
    max_number: int = field(init=False)
    max_sum: int = field(init=False)

    def __post_init__(self) -> None:
        if self.name != "mnist-1":
            raise ValueError(f"Unknown dataset: {self.name}")

        self.num_digits = 1

        self.max_number = 10**self.num_digits - 1
        self.max_sum = 2 * self.max_number

        if self.seed is None:
            self.seed = DEFAULT_SEED * 10000 + 100 * self.train_size + self.inference_size + self.num_digits

    def to_metadata_dict(self) -> dict:
        data = asdict(self)
        return {
            "name": data["name"],
            "num-digits": data["num_digits"],
            "class-size": data["class_size"],
            "pretrain-size": data["pretrain_size"],
            "pretrain-valid-size": data["pretrain_valid_size"],
            "pretrain-overlap": data["pretrain_overlap"],
            "train-size": data["train_size"],
            "valid-size": data["valid_size"],
            "inference-size": data["inference_size"],
            "overlap": data["overlap"],
            "seed": data["seed"],
            "max-number": data["max_number"],
            "max-sum": data["max_sum"],
            "data-source": "mnist",
        }


def dataset_dir(dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / f"experiment_{dataset}"


def pretrain_dir(config: DatasetConfig, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return (
        dataset_dir(config.name, data_root)
        / "pretrain"
        / f"size_{config.pretrain_size:05d}-valid_{config.pretrain_valid_size:04d}-overlap_{config.pretrain_overlap:.2f}"
    )


def neupsl_train_dir(config: DatasetConfig, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return (
        dataset_dir(config.name, data_root)
        / "neupsl-train"
        / f"size_{config.train_size:04d}-valid_{config.valid_size:04d}-overlap_{config.overlap:.2f}"
    )


def inference_dir(config: DatasetConfig, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return (
        dataset_dir(config.name, data_root)
        / "inference"
        / f"size_{config.inference_size:04d}"
    )


def experiment_id(config: DatasetConfig) -> str:
    return f"train_{config.train_size}_infer_{config.inference_size}"


def result_dir(config: DatasetConfig, results_root: str | Path = DEFAULT_RESULTS_ROOT) -> Path:
    return (
        Path(results_root)
        / experiment_id(config)
    )
