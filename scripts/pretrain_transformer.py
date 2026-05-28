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

from data.loader import load_babyai_partition, load_vocabularies
from models.transformer import BabyAIPredictor, save_checkpoint
from utils.config import DEFAULT_DATA_ROOT, DatasetConfig, dataset_dir, pretrain_dir
from utils.io import ensure_dir, load_json

DEFAULT_CKPT_ROOT = PROJECT_ROOT / "ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised pretraining for the BabyAI action predictor.")
    parser.add_argument("--dataset", choices=["babyai"], default="babyai")
    parser.add_argument("--pretrain-size", type=int, default=500)
    parser.add_argument("--pretrain-valid-size", type=int, default=100)
    parser.add_argument("--mission-encoding", choices=["surface", "structured"], default="surface")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-path", type=Path, help="Checkpoint path; defaults to ckpt/pretrained-babyai-transformer.pt.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    config = DatasetConfig(
        name=args.dataset,
        pretrain_size=args.pretrain_size,
        pretrain_valid_size=args.pretrain_valid_size,
        mission_encoding=args.mission_encoding,
        seed=args.random_seed,
    )
    data_dir = pretrain_dir(config, args.data_root)
    _validate_data_dir(data_dir)

    root = dataset_dir(config.name, args.data_root)
    token_vocab, type_vocab, action_vocab = load_vocabularies(root)
    train_tokens, train_types, train_labels = load_babyai_partition(data_dir, "train")
    valid_tokens, valid_types, valid_labels = load_babyai_partition(data_dir, "valid")

    device = _resolve_device(args.device)
    model = BabyAIPredictor(
        vocab_size=len(token_vocab),
        type_vocab_size=len(type_vocab),
        num_actions=len(action_vocab),
        max_seq_len=config.max_seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = _make_loader(train_tokens, train_types, train_labels, args.batch_size, shuffle=True)
    valid_loader = _make_loader(valid_tokens, valid_types, valid_labels, args.batch_size, shuffle=False)
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

    output_path = args.output_path or ensure_dir(args.ckpt_root) / "pretrained-babyai-transformer.pt"
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(
        output_path,
        model,
        {
            "dataset": config.name,
            "mission_encoding": config.mission_encoding,
            "action_vocab": action_vocab,
            "token_vocab": token_vocab,
            "type_vocab": type_vocab,
            "pretrain_size": config.pretrain_size,
            "pretrain_valid_size": config.pretrain_valid_size,
            "pretrain_epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_accuracy,
        },
    )
    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved pretrained model: {output_path}")
    print(f"Saved history: {history_path}")


def _validate_data_dir(data_dir: Path) -> None:
    if not (data_dir / "entity-data-map.txt").exists() or not (data_dir / "entity-type-map.txt").exists():
        raise FileNotFoundError(f"Missing prepared BabyAI data: {data_dir}. Run scripts/create_data.py first.")
    config_path = data_dir / "config.json"
    if not config_path.exists() or load_json(config_path).get("data-source") != "babyai":
        raise RuntimeError(f"Generated data at {data_dir} is missing the BabyAI data-source marker.")


def _make_loader(tokens, types, labels, batch_size: int, *, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(tokens, dtype=torch.long),
        torch.as_tensor(types, dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model, loader, criterion, optimizer, device, grad_clip: float):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_tokens, batch_types, batch_y in loader:
        batch_tokens = batch_tokens.to(device)
        batch_types = batch_types.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model.logits(batch_tokens, batch_types)
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
    for batch_tokens, batch_types, batch_y in loader:
        batch_tokens = batch_tokens.to(device)
        batch_types = batch_types.to(device)
        batch_y = batch_y.to(device)
        logits = model.logits(batch_tokens, batch_types)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item()) * len(batch_y)
        total_correct += int((torch.argmax(logits, dim=1) == batch_y).sum().item())
        total_seen += len(batch_y)
    return _metrics(total_loss, total_correct, total_seen)


def _metrics(total_loss: float, total_correct: int, total_seen: int) -> dict:
    return {"loss": total_loss / max(total_seen, 1), "accuracy": total_correct / max(total_seen, 1), "examples": total_seen}


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


if __name__ == "__main__":
    main()
