#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DEFAULT_RESULTS_ROOT, DatasetConfig, result_dir as build_result_dir
from utils.io import load_json, write_json


METRIC_RE = re.compile(r"(categorical_accuracy|sum_accuracy|digit_categorical_accuracy)['\":= ]+([0-9]*\.?[0-9]+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize NeuPSL runtime output and text logs.")
    parser.add_argument("--dataset", choices=["mnist-1"], default="mnist-1")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--inference-size", type=int, default=5000)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--results-dir", type=Path, help="Explicit result directory; overrides dataset/size options.")
    parser.add_argument("--no-write", action="store_true", help="Print summary without writing metrics.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.results_dir or build_result_dir(
        DatasetConfig(
            name=args.dataset,
            train_size=args.train_size,
            valid_size=args.valid_size,
            inference_size=args.inference_size,
            overlap=args.overlap,
        ),
        args.results_root,
    )
    metrics = summarize_results(result_dir)
    if not metrics:
        raise FileNotFoundError(f"No NeuPSL output found in {result_dir}. Run scripts/neupsl_train.py first.")

    if not args.no_write:
        write_json(result_dir / "metrics.json", metrics)

    print(json.dumps(metrics, indent=2, sort_keys=True))


def summarize_results(result_dir: Path) -> dict:
    metrics = {}

    runtime_output = result_dir / "runtime-output.json"
    if runtime_output.exists():
        payload = load_json(runtime_output)
        metrics["evaluations"] = payload.get("evaluations", [])
        metrics["num_atoms"] = len(payload.get("atoms", []))
        metrics["learned_rules"] = payload.get("rules", [])

    for log_path in result_dir.glob("*.txt"):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for name, value in METRIC_RE.findall(text):
            metrics[name] = float(value)

    return metrics


if __name__ == "__main__":
    main()
