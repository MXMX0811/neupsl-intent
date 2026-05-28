import json
from pathlib import Path
from typing import Iterable, List, Sequence


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data, indent: int | None = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent)


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_psl(path: str | Path, rows: Iterable[Sequence]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write("\t".join(str(item) for item in row) + "\n")


def load_psl(path: str | Path, dtype=str) -> List[List]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append([dtype(value) for value in line.split("\t")])
    return rows
