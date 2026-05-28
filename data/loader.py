from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np


def normalize_images(images) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images.reshape(images.shape[0], -1)
    return np.round(images / 255.0, 4)


def load_mnist_arrays(root: str | Path | None = None, download: bool = True):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data" / "raw"

    try:
        from torchvision.datasets import MNIST
        from torchvision.datasets.mnist import read_image_file, read_label_file
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load MNIST data.") from exc

    _ensure_flat_mnist_files(root, download=download, mnist_cls=MNIST)

    train_images = read_image_file(str(root / "train-images-idx3-ubyte")).numpy()
    train_labels = read_label_file(str(root / "train-labels-idx1-ubyte")).numpy()
    test_images = read_image_file(str(root / "t10k-images-idx3-ubyte")).numpy()
    test_labels = read_label_file(str(root / "t10k-labels-idx1-ubyte")).numpy()

    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    return normalize_images(images), labels.astype(np.int64)


def _ensure_flat_mnist_files(root: Path, *, download: bool, mnist_cls) -> None:
    root.mkdir(parents=True, exist_ok=True)
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    if all((root / name).exists() for name in required):
        return

    legacy_raw = root / "MNIST" / "raw"
    if not all((legacy_raw / name).exists() for name in required) and download:
        mnist_cls(root=str(root), train=True, download=True)
        mnist_cls(root=str(root), train=False, download=True)

    if all((legacy_raw / name).exists() for name in required):
        for name in required:
            shutil.copy2(legacy_raw / name, root / name)

    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing MNIST raw files in {root}: {missing}. "
            "Run with download=True or place the IDX files directly under data/raw/."
        )
