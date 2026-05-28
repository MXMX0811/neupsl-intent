from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer import BabyAIPredictor, load_checkpoint, save_checkpoint
from utils.metrics import categorical_accuracy, one_hot

try:
    import pslpython.deeppsl.model

    _DeepModelBase = pslpython.deeppsl.model.DeepModel
except ImportError:
    _DeepModelBase = object


class BabyAIActionModel(_DeepModelBase):
    def __init__(self):
        super().__init__()
        self.application = None
        self.model: BabyAIPredictor | None = None
        self.optimizer = None
        self.token_ids = None
        self.type_ids = None
        self.labels = None
        self.label_ids = None
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
            self.model = BabyAIPredictor(
                vocab_size=int(options["vocab-size"]),
                type_vocab_size=int(options["type-vocab-size"]),
                num_actions=int(options.get("action-size", 7)),
                max_seq_len=int(options.get("max-seq-len", 96)),
            )

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

        gradient_tensor = torch.as_tensor(np.asarray(gradients, dtype=np.float32), dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad()
        for start, end in self._batch_ranges(len(self.token_ids)):
            batch_probs = self.model(self.token_ids[start:end], self.type_ids[start:end])
            batch_probs.backward(gradient=gradient_tensor[start:end])
        supervised_loss = self._supervised_loss()
        if supervised_loss is not None:
            (self.supervised_loss_weight * supervised_loss).backward()
        self.optimizer.step()
        self.step += 1

        probs = self._predict_batched(training=False)
        results = {
            "step": self.step,
            "categorical_accuracy": categorical_accuracy(probs, self.label_ids.cpu().numpy()),
            "mean_gradient_abs": float(torch.mean(torch.abs(gradient_tensor)).detach().cpu().item()),
            "supervised_loss": float(supervised_loss.detach().cpu().item()) if supervised_loss is not None else None,
        }
        self._save_step_checkpoint(results)
        return results

    def internal_predict(self, data, options=None):
        options = options or {}
        if self.model is None:
            self.internal_init_model("inference" if self.application == "inference" else "learning", options)
        self._prepare_data(data, options)
        probs = self._predict_batched(training=self.application == "learning" or options.get("learn", False))
        return probs, {"metrics": {"categorical_accuracy": categorical_accuracy(probs, self.label_ids.cpu().numpy())}}

    def internal_eval(self, data, options=None):
        probs, metrics = self.internal_predict(data, options or {})
        return metrics

    def internal_save(self, options=None):
        options = options or {}
        if self.model is None:
            return {}
        save_path = options.get("save-path", "saved-networks/nesy-trained-pt")
        save_checkpoint(_checkpoint_path(save_path), self.model, {"num_actions": self.model.num_actions})
        return {"save_path": str(_checkpoint_path(save_path))}

    def _prepare_data(self, data, options):
        data = np.asarray(data, dtype=np.int64)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("Expected rows: step_id, token ids..., action_id. Type ids are loaded from entity-type-map-path.")
        type_map_path = options.get("entity-type-map-path")
        if not type_map_path:
            raise ValueError("BabyAIActionModel requires entity-type-map-path.")
        type_rows = np.asarray(_load_int_rows(type_map_path), dtype=np.int64)
        type_by_step = {int(row[0]): row[1:] for row in type_rows}

        step_ids = data[:, 0].astype(np.int64)
        self.token_ids = torch.as_tensor(data[:, 1:-1], dtype=torch.long, device=self.device)
        self.type_ids = torch.as_tensor(np.asarray([type_by_step[int(step_id)] for step_id in step_ids]), dtype=torch.long, device=self.device)
        self.label_ids = torch.as_tensor(data[:, -1].astype(np.int64), dtype=torch.long, device=self.device)
        self.labels = torch.as_tensor(one_hot(self.label_ids.cpu().numpy(), int(options.get("action-size", 7))), dtype=torch.float32, device=self.device)

    def _batch_ranges(self, size: int):
        batch_size = max(int(self.batch_size), 1)
        for start in range(0, size, batch_size):
            yield start, min(start + batch_size, size)

    @torch.no_grad()
    def _predict_batched(self, *, training: bool) -> np.ndarray:
        if self.model is None or self.token_ids is None:
            return np.empty((0, 0), dtype=np.float32)
        self.model.train(training)
        batches = []
        for start, end in self._batch_ranges(len(self.token_ids)):
            batches.append(self.model(self.token_ids[start:end], self.type_ids[start:end]).detach().cpu())
        if not batches:
            return np.empty((0, int(getattr(self.model, "num_actions", 0))), dtype=np.float32)
        return torch.cat(batches, dim=0).numpy()

    def _supervised_loss(self):
        if self.supervised_loss_weight <= 0.0 or self.token_ids is None or self.label_ids is None:
            return None
        losses = []
        for start, end in self._batch_ranges(len(self.token_ids)):
            batch_probs = self.model(self.token_ids[start:end], self.type_ids[start:end])
            losses.append(F.nll_loss(torch.log(batch_probs.clamp_min(1e-8)), self.label_ids[start:end]))
        return torch.stack(losses).mean() if losses else None

    def _save_step_checkpoint(self, metrics: dict) -> None:
        if self.model is None or self.checkpoint_dir is None:
            return
        if self.checkpoint_frequency <= 0 or self.step % self.checkpoint_frequency != 0:
            return
        save_checkpoint(
            self.checkpoint_dir / f"step_{self.step:04d}.pt",
            self.model,
            {
                "neupsl_step": self.step,
                "categorical_accuracy": metrics.get("categorical_accuracy"),
                "mean_gradient_abs": metrics.get("mean_gradient_abs"),
            },
        )


def _load_int_rows(path: str | Path) -> list[list[int]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append([int(value) for value in line.split("\t")])
    return rows


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
