#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psl.runner import PSLRunner
from utils.config import DEFAULT_DATA_ROOT, DEFAULT_RESULTS_ROOT, DatasetConfig, inference_dir, neupsl_train_dir, result_dir as build_result_dir
from utils.io import ensure_dir, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuPSL joint fine-tuning and inference for BabyAI action prediction.")
    parser.add_argument("--dataset", choices=["babyai"], default="babyai")
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--inference-size", type=int, default=1000)
    parser.add_argument("--mission-encoding", choices=["surface", "structured"], default="surface")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=250)
    parser.add_argument("--admm-iterations", type=int, default=200)
    parser.add_argument("--checkpoint-frequency", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--random-seed", type=int, default=16)
    parser.add_argument("--rules-path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-class", default="BabyAIActionModel")
    parser.add_argument("--pretrained-path", type=Path)
    parser.add_argument("--backend", choices=["runtime", "cli"], default="runtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        name=args.dataset,
        train_size=args.train_size,
        valid_size=args.valid_size,
        inference_size=args.inference_size,
        mission_encoding=args.mission_encoding,
        seed=args.random_seed,
    )
    train_data_dir = neupsl_train_dir(config, args.data_root)
    inference_data_dir = inference_dir(config, args.data_root)
    for data_dir in (train_data_dir, inference_data_dir):
        if not (data_dir / "entity-data-map.txt").exists():
            raise FileNotFoundError(f"Missing prepared data: {data_dir}. Run scripts/create_data.py first.")
    if load_json(train_data_dir / "config.json").get("data-source") != "babyai":
        raise RuntimeError(f"Generated data at {train_data_dir} is not BabyAI data.")

    result_dir = ensure_dir(build_result_dir(config, args.results_root))
    checkpoint_dir = result_dir / "checkpoints"
    runner = PSLRunner()
    config_path = runner.build_config(
        config,
        data_root=args.data_root,
        output_path=result_dir / "psl-config.json",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        admm_iterations=args.admm_iterations,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        log_level=args.log_level,
        rule_path=args.rules_path,
        model_path=args.model_path,
        model_class=args.model_class,
        random_seed=args.random_seed,
        pretrained_path=args.pretrained_path,
    )

    if args.backend == "runtime":
        result = runner.run_runtime(config_path, result_dir)
        trained_model = result_dir / "saved-networks" / "nesy-trained-pt" / "model.pt"
        if trained_model.exists():
            shutil.copy2(trained_model, result_dir / "model.pt")
        print(f"NeuPSL runtime complete. output={result_dir / 'runtime-output.json'}")
        print(f"Trained model: {result_dir / 'model.pt'}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Evaluations: {result.get('evaluations', [])}")
    else:
        completed = runner.run(config_path, result_dir / "inferred-predicates")
        print(f"NeuPSL CLI complete. returncode={completed.returncode}")


if __name__ == "__main__":
    main()
