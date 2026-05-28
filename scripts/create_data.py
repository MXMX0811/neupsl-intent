#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.rule_extraction import write_rule_observations, write_split_rule_library
from data.sequence_extraction import build_action_vocab, build_token_vocab, build_type_vocab, encode_example, generate_step_examples
from utils.config import DEFAULT_DATA_ROOT, DatasetConfig, dataset_dir, inference_dir, neupsl_train_dir, pretrain_dir, rules_dir
from utils.io import ensure_dir, write_json, write_psl


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BabyAI action-sequence data.")
    parser.add_argument("--dataset", choices=["babyai"], default="babyai")
    parser.add_argument("--pretrain-size", type=int, default=500)
    parser.add_argument("--pretrain-valid-size", type=int, default=100)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--inference-size", type=int, default=1000)
    parser.add_argument("--min-episode-len", type=int, default=3)
    parser.add_argument("--mission-encoding", choices=["surface", "structured"], default="surface")
    parser.add_argument("--generator-backend", choices=["auto", "minigrid", "synthetic"], default="auto")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--random-seed", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        name=args.dataset,
        pretrain_size=args.pretrain_size,
        pretrain_valid_size=args.pretrain_valid_size,
        train_size=args.train_size,
        valid_size=args.valid_size,
        inference_size=args.inference_size,
        min_episode_len=args.min_episode_len,
        mission_encoding=args.mission_encoding,
        generator_backend=args.generator_backend,
        seed=args.random_seed,
    )
    paths = generate_experiment_datasets(config, data_root=args.data_root)
    print(f"Generated BabyAI pretrain data at {paths['pretrain']}")
    print(f"Generated BabyAI NeuPSL train data at {paths['neupsl_train']}")
    print(f"Generated BabyAI inference data at {paths['inference']}")


def generate_experiment_datasets(config: DatasetConfig, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Path]:
    root = ensure_dir(dataset_dir(config.name, data_root))
    action_vocab = build_action_vocab()
    type_vocab = build_type_vocab()

    pretrain_train = generate_step_examples(
        env_ids=config.pretrain_envs,
        episode_count=config.pretrain_size,
        start_step_id=0,
        seed=config.seed,
        min_episode_len=config.min_episode_len,
        mission_encoding=config.mission_encoding,
        backend=config.generator_backend,
    )
    pretrain_valid = generate_step_examples(
        env_ids=config.pretrain_envs,
        episode_count=config.pretrain_valid_size,
        start_step_id=len(pretrain_train),
        seed=config.seed + 1,
        min_episode_len=config.min_episode_len,
        mission_encoding=config.mission_encoding,
        backend=config.generator_backend,
    )
    neupsl_train = generate_step_examples(
        env_ids=config.neupsl_envs,
        episode_count=config.train_size,
        start_step_id=len(pretrain_train) + len(pretrain_valid),
        seed=config.seed + 2,
        min_episode_len=config.min_episode_len,
        mission_encoding=config.mission_encoding,
        backend=config.generator_backend,
    )
    neupsl_valid = generate_step_examples(
        env_ids=config.neupsl_envs,
        episode_count=config.valid_size,
        start_step_id=len(pretrain_train) + len(pretrain_valid) + len(neupsl_train),
        seed=config.seed + 3,
        min_episode_len=config.min_episode_len,
        mission_encoding=config.mission_encoding,
        backend=config.generator_backend,
    )
    inference = generate_step_examples(
        env_ids=config.neupsl_envs,
        episode_count=config.inference_size,
        start_step_id=len(pretrain_train) + len(pretrain_valid) + len(neupsl_train) + len(neupsl_valid),
        seed=config.seed + 4,
        min_episode_len=config.min_episode_len,
        mission_encoding=config.mission_encoding,
        backend=config.generator_backend,
    )

    all_examples = pretrain_train + pretrain_valid + neupsl_train + neupsl_valid + inference
    token_vocab = build_token_vocab(all_examples)
    write_json(root / "action-vocab.json", action_vocab)
    write_json(root / "type-vocab.json", type_vocab)
    write_json(root / "token-vocab.json", token_vocab)

    pretrain_path = pretrain_dir(config, data_root)
    neupsl_path = neupsl_train_dir(config, data_root)
    inference_path = inference_dir(config, data_root)
    _write_supervised_split(pretrain_path, config, {"train": pretrain_train, "valid": pretrain_valid}, token_vocab, type_vocab, action_vocab)
    _write_supervised_split(neupsl_path, config, {"train": neupsl_train, "valid": neupsl_valid}, token_vocab, type_vocab, action_vocab)
    _write_supervised_split(inference_path, config, {"inference": inference}, token_vocab, type_vocab, action_vocab)

    pretrain_rules = write_split_rule_library(rules_dir(config, "pretrain", data_root), pretrain_train + pretrain_valid)
    neupsl_rules = write_split_rule_library(rules_dir(config, "neupsl-train", data_root), neupsl_train + neupsl_valid, inherited_rules=pretrain_rules)
    write_split_rule_library(rules_dir(config, "inference", data_root), inference, inherited_rules=neupsl_rules)
    return {"pretrain": pretrain_path, "neupsl_train": neupsl_path, "inference": inference_path}


def _write_supervised_split(
    out_dir: Path,
    config: DatasetConfig,
    partitions: dict[str, list],
    token_vocab: dict[str, int],
    type_vocab: dict[str, int],
    action_vocab: dict[str, int],
) -> None:
    ensure_dir(out_dir)
    encoded_by_partition = {}
    for partition, examples in partitions.items():
        encoded = [encode_example(example, token_vocab, type_vocab, action_vocab, config.max_seq_len) for example in examples]
        encoded_by_partition[partition] = encoded
        _write_jsonl(out_dir / f"sequence-data-{partition}.jsonl", encoded)
        _write_targets(out_dir, partition, encoded, action_vocab)
        report = write_rule_observations(out_dir, partition, examples)
        write_json(out_dir / f"generation-report-{partition}.json", {"examples": len(encoded), **report})

    all_encoded = [row for rows in encoded_by_partition.values() for row in rows]
    write_psl(out_dir / "entity-data-map.txt", [[row["step_id"], *row["token_ids"], row["action_id"]] for row in all_encoded])
    write_psl(out_dir / "entity-type-map.txt", [[row["step_id"], *row["type_ids"]] for row in all_encoded])
    write_json(out_dir / "config.json", config.to_metadata_dict())


def _write_targets(out_dir: Path, partition: str, rows: list[dict], action_vocab: dict[str, int]) -> None:
    target_rows = []
    truth_rows = []
    for row in rows:
        for action, action_id in action_vocab.items():
            target_rows.append([row["step_id"], action])
            truth_rows.append([row["step_id"], action, int(action_id == row["action_id"])])
    write_psl(out_dir / f"action-target-{partition}.txt", target_rows)
    write_psl(out_dir / f"action-truth-{partition}.txt", truth_rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
