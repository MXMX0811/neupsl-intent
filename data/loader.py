from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.io import load_json, load_psl


def load_babyai_partition(data_dir: str | Path, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    entity_rows = load_psl(data_dir / "entity-data-map.txt", dtype=int)
    type_rows = load_psl(data_dir / "entity-type-map.txt", dtype=int)
    targets = _target_step_ids(data_dir / f"action-target-{partition}.txt")

    token_by_step = {row[0]: row[1:-1] for row in entity_rows}
    label_by_step = {row[0]: row[-1] for row in entity_rows}
    type_by_step = {row[0]: row[1:] for row in type_rows}

    token_ids = []
    type_ids = []
    labels = []
    for step_id in targets:
        token_ids.append(token_by_step[step_id])
        type_ids.append(type_by_step[step_id])
        labels.append(label_by_step[step_id])
    return (
        np.asarray(token_ids, dtype=np.int64),
        np.asarray(type_ids, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )


def load_vocabularies(root: str | Path) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    root = Path(root)
    return (
        load_json(root / "token-vocab.json"),
        load_json(root / "type-vocab.json"),
        load_json(root / "action-vocab.json"),
    )


def _target_step_ids(path: Path) -> list[int]:
    rows = load_psl(path, dtype=str)
    return sorted({int(row[0]) for row in rows})
