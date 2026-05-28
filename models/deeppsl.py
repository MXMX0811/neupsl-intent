from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer import MNISTTransformerClassifier, load_checkpoint, save_checkpoint
from utils.metrics import categorical_accuracy, one_hot

try:
    import pslpython.deeppsl.model
    _DeepModelBase = pslpython.deeppsl.model.DeepModel
except ImportError:
    _DeepModelBase = object


class MNISTDeepPSL(_DeepModelBase):
    def __init__(self):
        super().__init__()
        self.application = None
        self.model: MNISTTransformerClassifier | None = None
        self.optimizer = None
        self.features = None
        self.labels = None
        self.label_ids = None
        self.predictions = None
        self.device = torch.device("cpu")
        self.step = 0
        self.checkpoint_dir = None
        self.checkpoint_frequency = 0
        self.batch_size = 256
        self.supervised_loss_weight = 0.0

    def internal_init_model(self, application, options=None):
        options = options or {}
        self.application = application
        self.device = torch.device(options.get("device", "cpu"))
        class_size = int(options.get("class-size", 10))
        random_seed = options.get("random-seed")
        if random_seed is not None:
            seed = int(random_seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.checkpoint_dir = Path(options["checkpoint-dir"]) if options.get("checkpoint-dir") else None
        self.checkpoint_frequency = int(options.get("checkpoint-frequency", 0))

        resume_path = _latest_checkpoint(self.checkpoint_dir) if application != "inference" else None
        pretrained_path = options.get("pretrained-path")
        if resume_path is not None:
            self.model = load_checkpoint(resume_path, map_location=self.device)
            self.step = int(_checkpoint_metadata(resume_path).get("neupsl_step", 0))
        elif application == "inference" and options.get("save-path"):
            self.model = load_checkpoint(_checkpoint_path(options["save-path"]), map_location=self.device)
        elif pretrained_path:
            self.model = load_checkpoint(_checkpoint_path(pretrained_path), map_location=self.device)
        else:
            self.model = MNISTTransformerClassifier(num_classes=class_size)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(options.get("learning-rate", 1e-3)))
        self.batch_size = int(options.get("batch-size", self.batch_size))
        self.supervised_loss_weight = float(options.get("supervised-loss-weight", 0.0))
        return {}

    def internal_fit(self, data, gradients, options=None):
        options = options or {}
        if self.model is None:
            self.internal_init_model("learning", options)

        self._prepare_data(data, options)
        self.model.train()

        native_gradients = np.array(gradients, dtype=np.float32, copy=True)
        gradient_tensor = torch.as_tensor(native_gradients, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad()
        for start, end in self._batch_ranges(len(self.features)):
            batch_probs = self.model(self.features[start:end])
            batch_probs.backward(gradient=gradient_tensor[start:end])
        supervised_loss = self._supervised_loss()
        if supervised_loss is not None:
            (self.supervised_loss_weight * supervised_loss).backward()
        self.optimizer.step()
        self.predictions = None
        self.step += 1

        probs = self._predict_batched(training=False)

        results = {
            "step": self.step,
            "categorical_accuracy": categorical_accuracy(probs, self.label_ids.cpu().numpy()),
            "mean_gradient_abs": float(torch.mean(torch.abs(gradient_tensor)).detach().cpu().item()),
            "supervised_loss": float(supervised_loss.detach().cpu().item()) if supervised_loss is not None else None,
            "sample_prediction": probs[0].round(4).tolist() if len(probs) else [],
            "sample_label": int(self.label_ids[0].detach().cpu().item()) if len(self.label_ids) else None,
        }
        self._save_step_checkpoint(results)
        return results

    def internal_predict(self, data, options=None):
        options = options or {}
        if self.model is None:
            self.internal_init_model("inference" if self.application == "inference" else "learning", options)

        self._prepare_data(data, options)
        if self.application == "learning" or options.get("learn", False):
            self.model.train()
            self.predictions = None
            probs = self._predict_batched(training=True)
            return probs, {}

        probs = self._predict_batched(training=False)
        metrics = {"categorical_accuracy": categorical_accuracy(probs, self.label_ids.cpu().numpy())}
        return probs, {"metrics": metrics}

    def internal_eval(self, data, options=None):
        probs, _ = self.internal_predict(data, options or {})
        return {"metrics": {"categorical_accuracy": categorical_accuracy(probs, self.label_ids.cpu().numpy())}}

    def internal_save(self, options=None):
        options = options or {}
        if self.model is None:
            return {}
        save_path = options.get("save-path", "saved-networks/nesy-trained-pt")
        save_checkpoint(_checkpoint_path(save_path), self.model, {"num_classes": self.model.num_classes})
        return {"save_path": str(_checkpoint_path(save_path))}

    def _prepare_data(self, data, options):
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Expected data rows with 784 features and one label column.")

        class_size = int(options.get("class-size", 10))
        self.features = torch.as_tensor(data[:, :-1], dtype=torch.float32, device=self.device)
        self.label_ids = torch.as_tensor(data[:, -1].astype(np.int64), dtype=torch.long, device=self.device)
        self.labels = torch.as_tensor(one_hot(self.label_ids.cpu().numpy(), class_size), dtype=torch.float32, device=self.device)

    def _batch_ranges(self, size: int):
        batch_size = max(int(self.batch_size), 1)
        for start in range(0, size, batch_size):
            yield start, min(start + batch_size, size)

    def _predict_batched(self, *, training: bool) -> np.ndarray:
        if self.model is None or self.features is None:
            return np.empty((0, 0), dtype=np.float32)

        self.model.train(training)
        batches = []
        with torch.no_grad():
            for start, end in self._batch_ranges(len(self.features)):
                batch_probs = self.model(self.features[start:end])
                batches.append(batch_probs.detach().cpu())
        if not batches:
            return np.empty((0, int(getattr(self.model, "num_classes", 0))), dtype=np.float32)
        return torch.cat(batches, dim=0).numpy()

    def _supervised_loss(self):
        if self.supervised_loss_weight <= 0.0 or self.features is None or self.label_ids is None:
            return None

        losses = []
        for start, end in self._batch_ranges(len(self.features)):
            batch_probs = self.model(self.features[start:end])
            losses.append(F.nll_loss(torch.log(batch_probs.clamp_min(1e-8)), self.label_ids[start:end]))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _save_step_checkpoint(self, metrics: dict) -> None:
        if self.model is None or self.checkpoint_dir is None:
            return
        if self.checkpoint_frequency <= 0 or self.step % self.checkpoint_frequency != 0:
            return

        save_checkpoint(
            self.checkpoint_dir / f"step_{self.step:04d}.pt",
            self.model,
            {
                "num_classes": self.model.num_classes,
                "neupsl_step": self.step,
                "categorical_accuracy": metrics.get("categorical_accuracy"),
                "mean_gradient_abs": metrics.get("mean_gradient_abs"),
            },
        )


def _checkpoint_path(save_path: str | Path) -> Path:
    path = Path(save_path)
    if path.suffix in {".pt", ".pth"}:
        return path
    return path / "model.pt"


def _latest_checkpoint(checkpoint_dir: str | Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    path = Path(checkpoint_dir)
    if not path.exists():
        return None
    checkpoints = sorted(path.glob("step_*.pt"), key=_checkpoint_step)
    return checkpoints[-1] if checkpoints else None


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def _checkpoint_metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return payload.get("metadata", {})


MNISTAdditionModel = MNISTDeepPSL
