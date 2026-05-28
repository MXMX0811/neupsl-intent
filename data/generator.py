from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.config import DEFAULT_DATA_ROOT, DatasetConfig, dataset_dir, inference_dir, neupsl_train_dir, pretrain_dir
from utils.io import ensure_dir, write_json, write_psl


def digits_to_number(digits) -> int:
    value = 0
    for digit in digits:
        value = value * 10 + int(digit)
    return value


def digits_to_sum(digits, num_digits: int) -> int:
    return digits_to_number(digits[:num_digits]) + digits_to_number(digits[num_digits:])


def generate_split(config: DatasetConfig, labels, indexes, overlap: float | None = None):
    indexes = np.asarray(indexes, dtype=np.int64)
    original_size = len(indexes)
    overlap = config.overlap if overlap is None else overlap

    if original_size == 0:
        return np.empty((0, 2 * config.num_digits), dtype=np.int64), np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(config.seed)
    if overlap > 0:
        extra = rng.choice(indexes[:original_size], size=int(original_size * overlap), replace=True)
        indexes = np.append(indexes, extra)

    block_size = 2 * config.num_digits
    indexes = indexes[: len(indexes) - (len(indexes) % block_size)]
    if len(indexes) == 0:
        return np.empty((0, block_size), dtype=np.int64), np.empty((0,), dtype=np.int64)

    entities = np.unique(indexes.reshape(-1, block_size), axis=0)
    sum_labels = np.array([digits_to_sum(labels[row], config.num_digits) for row in entities], dtype=np.int64)
    return entities, sum_labels


def create_entity_data_map(features, labels, entities):
    entities = np.asarray(entities, dtype=np.int64).reshape(-1)
    rows = []
    for entity in entities:
        rows.append([int(entity), *features[entity].astype(float).tolist(), int(labels[entity])])
    return rows


def create_image_data(config: DatasetConfig, entities):
    return [[int(entity), digit] for entity in np.asarray(entities).reshape(-1) for digit in range(config.class_size)]


def create_image_sum_data(config: DatasetConfig, sum_entities, sum_labels):
    targets = []
    truth = []
    for example, label in zip(sum_entities, sum_labels):
        prefix = [int(value) for value in example]
        for sum_value in range(config.max_sum + 1):
            targets.append(prefix + [sum_value])
            truth.append(prefix + [sum_value, int(sum_value == int(label))])
    return np.unique(np.asarray(targets, dtype=np.int64), axis=0).tolist(), np.unique(np.asarray(truth, dtype=np.int64), axis=0).tolist()


def create_sum_data_add1(config: DatasetConfig):
    number_sum = []
    possible_digits = []
    for x_value in range(config.class_size):
        for y_value in range(config.class_size):
            z_value = x_value + y_value
            number_sum.append([x_value, y_value, z_value])
            possible_digits.append([x_value, z_value])
    return number_sum, possible_digits


def write_shared_data(config: DatasetConfig, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    number_sum, possible_digits = create_sum_data_add1(config)

    write_psl(out_dir / "number-sum.txt", number_sum)
    write_psl(out_dir / "possible-digits.txt", possible_digits)

    write_json(out_dir / "config.json", config.to_metadata_dict())


def write_partition_data(config: DatasetConfig, out_dir: str | Path, features, labels, partition_indexes: dict[str, np.ndarray], overlaps: dict[str, float] | None = None) -> None:
    out_dir = ensure_dir(out_dir)
    overlaps = overlaps or {}

    total_entities = []
    for partition, indexes in partition_indexes.items():
        sum_entities, sum_labels = generate_split(config, labels, indexes, overlap=overlaps.get(partition, 0.0))
        image_sum_target, image_sum_truth = create_image_sum_data(config, sum_entities, sum_labels)

        image_entities = np.unique(sum_entities.reshape(-1)).reshape(-1, 1)
        total_entities.extend(image_entities.reshape(-1).tolist())

        write_psl(out_dir / f"image-sum-block-{partition}.txt", sum_entities.tolist())
        write_psl(out_dir / f"image-sum-target-{partition}.txt", image_sum_target)
        write_psl(out_dir / f"image-sum-truth-{partition}.txt", image_sum_truth)
        write_psl(out_dir / f"image-target-{partition}.txt", create_image_data(config, image_entities))

    write_psl(out_dir / "entity-data-map.txt", create_entity_data_map(features, labels, np.unique(total_entities)))
    write_json(out_dir / "config.json", config.to_metadata_dict())


def generate_experiment_datasets(config: DatasetConfig, features, labels, data_root: str | Path = DEFAULT_DATA_ROOT) -> dict[str, Path]:
    shared_dir = dataset_dir(config.name, data_root)
    pretrain_path = pretrain_dir(config, data_root)
    neupsl_path = neupsl_train_dir(config, data_root)
    inference_path = inference_dir(config, data_root)

    required_size = (
        config.pretrain_size
        + config.pretrain_valid_size
        + config.train_size
        + config.valid_size
        + config.inference_size
    )
    if required_size > len(features):
        raise ValueError(f"Requested {required_size} MNIST images, but only {len(features)} are available.")

    rng = np.random.default_rng(config.seed)
    all_indexes = np.arange(len(features), dtype=np.int64)
    rng.shuffle(all_indexes)

    cursor = 0

    def take(size: int) -> np.ndarray:
        nonlocal cursor
        subset = all_indexes[cursor : cursor + size]
        cursor += size
        return subset

    pretrain_indexes = {
        "train": take(config.pretrain_size),
        "valid": take(config.pretrain_valid_size),
    }
    neupsl_train_indexes = {
        "train": take(config.train_size),
        "valid": take(config.valid_size),
    }
    inference_indexes = {
        "inference": take(config.inference_size),
    }

    write_shared_data(config, shared_dir)
    write_partition_data(config, pretrain_path, features, labels, pretrain_indexes, {"train": config.pretrain_overlap})
    write_partition_data(config, neupsl_path, features, labels, neupsl_train_indexes, {"train": config.overlap})
    write_partition_data(config, inference_path, features, labels, inference_indexes)

    return {"pretrain": pretrain_path, "neupsl_train": neupsl_path, "inference": inference_path}
