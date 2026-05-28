#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer import MNISTTransformerClassifier, save_checkpoint
from utils.config import DEFAULT_DATA_ROOT, DatasetConfig, pretrain_dir
from utils.io import ensure_dir, load_json, load_psl

DEFAULT_CKPT_ROOT = PROJECT_ROOT / "ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised pretraining for the MNIST Transformer classifier.")
    parser.add_argument("--dataset", choices=["mnist-1"], default="mnist-1")
    parser.add_argument("--pretrain-size", type=int, default=500)
    parser.add_argument("--pretrain-valid-size", type=int, default=100)
    parser.add_argument("--pretrain-overlap", type=float, default=0.4)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-path", type=Path, help="Checkpoint path; defaults to ckpt/pretrained-transformer.pt.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm; use 0 or less to disable.")
    parser.add_argument("--random-seed", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or any torch device string.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    config = DatasetConfig(
        name=args.dataset,
        pretrain_size=args.pretrain_size,
        pretrain_valid_size=args.pretrain_valid_size,
        pretrain_overlap=args.pretrain_overlap,
    )
    data_dir = pretrain_dir(config, args.data_root)
    _validate_data_dir(data_dir)

    entity_ids, features, labels = _load_entity_data(data_dir / "entity-data-map.txt")
    train_ids = _target_entity_ids(data_dir / "image-target-train.txt")
    valid_ids = _target_entity_ids(data_dir / "image-target-valid.txt")
    train_x, train_y = _select_entities(entity_ids, features, labels, train_ids)
    valid_x, valid_y = _select_entities(entity_ids, features, labels, valid_ids)

    device = _resolve_device(args.device)
    model = MNISTTransformerClassifier(num_classes=config.class_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = _make_loader(train_x, train_y, args.batch_size, shuffle=True)
    valid_loader = _make_loader(valid_x, valid_y, args.batch_size, shuffle=False)
    history = []
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        valid_metrics = _evaluate(model, valid_loader, criterion, device)
        is_best = valid_metrics["accuracy"] > best_accuracy
        if is_best:
            best_accuracy = valid_metrics["accuracy"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        row = {"epoch": epoch, "is_best": is_best, **_prefix(train_metrics, "train"), **_prefix(valid_metrics, "valid")}
        history.append(row)
        print(
            f"epoch={epoch} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} valid_acc={row['valid_accuracy']:.4f}"
            f"{' best' if is_best else ''}"
        )

    output_path = args.output_path or _default_output_path(args.ckpt_root)
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(output_path, model, {
        "num_classes": config.class_size,
        "dataset": config.name,
        "pretrain_size": config.pretrain_size,
        "pretrain_valid_size": config.pretrain_valid_size,
        "pretrain_overlap": config.pretrain_overlap,
        "pretrain_epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_valid_accuracy": best_accuracy,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
    })
    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved pretrained model: {output_path}")
    print(f"Saved history: {history_path}")


def _validate_data_dir(data_dir: Path) -> None:
    if not (data_dir / "entity-data-map.txt").exists():
        raise FileNotFoundError(f"Missing prepared data: {data_dir}. Run scripts/create_data.py first.")
    config_path = data_dir / "config.json"
    if not config_path.exists() or load_json(config_path).get("data-source") != "mnist":
        raise RuntimeError(f"Generated data at {data_dir} is missing the real-MNIST data-source marker.")


def _load_entity_data(path: Path):
    rows = load_psl(path, dtype=float)
    data = np.asarray(rows, dtype=np.float32)
    return data[:, 0].astype(np.int64), data[:, 1:-1], data[:, -1].astype(np.int64)


def _target_entity_ids(path: Path) -> np.ndarray:
    rows = load_psl(path, dtype=int)
    return np.unique(np.asarray([row[0] for row in rows], dtype=np.int64))


def _select_entities(entity_ids, features, labels, selected_ids):
    positions = {int(entity_id): index for index, entity_id in enumerate(entity_ids)}
    selected_positions = [positions[int(entity_id)] for entity_id in selected_ids]
    return features[selected_positions], labels[selected_positions]


def _make_loader(features, labels, batch_size: int, *, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model, loader, criterion, optimizer, device, grad_clip: float):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model.logits(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_y)
        total_correct += int((torch.argmax(logits, dim=1) == batch_y).sum().item())
        total_seen += len(batch_y)
    return _metrics(total_loss, total_correct, total_seen)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model.logits(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item()) * len(batch_y)
        total_correct += int((torch.argmax(logits, dim=1) == batch_y).sum().item())
        total_seen += len(batch_y)
    return _metrics(total_loss, total_correct, total_seen)


def _metrics(total_loss: float, total_correct: int, total_seen: int) -> dict:
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "examples": total_seen,
    }


def _prefix(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_output_path(ckpt_root: Path) -> Path:
    return ensure_dir(ckpt_root) / "pretrained-transformer.pt"


if __name__ == "__main__":
    main()
