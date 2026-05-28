'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2026-05-20 15:32:54
LastEditors: Mingxin Zhang
LastEditTime: 2026-05-27 20:26:16
Copyright (c) 2026 by Mingxin Zhang, All Rights Reserved. 
'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generator import generate_experiment_datasets
from data.loader import load_mnist_arrays
from utils.config import DEFAULT_DATA_ROOT, DatasetConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PSL-compatible MNIST addition data.")
    parser.add_argument("--dataset", choices=["mnist-1"], default="mnist-1")
    parser.add_argument("--pretrain-size", type=int, default=500)
    parser.add_argument("--pretrain-valid-size", type=int, default=100)
    parser.add_argument("--pretrain-overlap", type=float, default=0.4)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--inference-size", type=int, default=1000)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--no-download", action="store_true", help="Do not download MNIST if it is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        name=args.dataset,
        pretrain_size=args.pretrain_size,
        pretrain_valid_size=args.pretrain_valid_size,
        pretrain_overlap=args.pretrain_overlap,
        train_size=args.train_size,
        valid_size=args.valid_size,
        inference_size=args.inference_size,
        overlap=args.overlap,
    )

    features, labels = load_mnist_arrays(download=not args.no_download)

    out_dirs = generate_experiment_datasets(config, features, labels, data_root=args.data_root)
    print(f"Generated {config.name} pretrain data at {out_dirs['pretrain']}")
    print(f"Generated {config.name} NeuPSL train data at {out_dirs['neupsl_train']}")
    print(f"Generated {config.name} inference data at {out_dirs['inference']}")


if __name__ == "__main__":
    main()
