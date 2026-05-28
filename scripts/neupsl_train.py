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
    parser = argparse.ArgumentParser(description="Run NeuPSL joint fine-tuning and inference for MNIST addition.")
    parser.add_argument("--dataset", choices=["mnist-1"], default="mnist-1")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--inference-size", type=int, default=5000)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256, help="Neural classifier batch size used inside each full-graph NeuPSL step.")
    parser.add_argument("--gradient-steps", type=int, default=500, help="PSL gradient descent steps; each step can update neural classifier parameters.")
    parser.add_argument("--admm-iterations", type=int, default=100, help="Override ADMM iterations for NeuPSL learning and inference.")
    parser.add_argument("--checkpoint-frequency", type=int, default=10, help="Save a Transformer checkpoint every N NeuPSL gradient steps; use 0 to disable.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--random-seed", type=int, default=16, help="Seed passed to the PyTorch DeepModel.")
    parser.add_argument("--rules-path", type=Path, help="Optional replacement PSL rule/config JSON.")
    parser.add_argument("--model-path", type=Path, help="Optional replacement DeepModel Python file.")
    parser.add_argument("--model-class", default="MNISTAdditionModel", help="DeepModel class in --model-path.")
    parser.add_argument("--pretrained-path", type=Path, help="Optional pretrained Transformer checkpoint to initialize NeuPSL learning.")
    parser.add_argument("--backend", choices=["runtime", "cli"], default="runtime", help="runtime uses pslpython's bundled JVM bridge; cli uses psl-cli jar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        name=args.dataset,
        train_size=args.train_size,
        valid_size=args.valid_size,
        inference_size=args.inference_size,
        overlap=args.overlap,
    )
    train_data_dir = neupsl_train_dir(config, args.data_root)
    inference_data_dir = inference_dir(config, args.data_root)
    for data_dir in (train_data_dir, inference_data_dir):
        if not (data_dir / "entity-data-map.txt").exists():
            raise FileNotFoundError(f"Missing prepared data: {data_dir}. Run scripts/create_data.py first.")
    data_config_path = train_data_dir / "config.json"
    if not data_config_path.exists() or load_json(data_config_path).get("data-source") != "mnist":
        raise RuntimeError(
            f"Generated data at {train_data_dir} is missing the real-MNIST data-source marker. "
            "Regenerate it with scripts/create_data.py."
        )

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
        print(f"Learned PSL rules: {result_dir / 'learned-rules.txt'}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Evaluations: {result.get('evaluations', [])}")
        print(f"Atoms: {len(result.get('atoms', []))}")
    else:
        completed = runner.run(config_path, result_dir / "inferred-predicates")
        print(f"NeuPSL CLI complete. returncode={completed.returncode}")

if __name__ == "__main__":
    main()
