from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BabyAIPredictor(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        type_vocab_size: int,
        num_actions: int = 7,
        max_seq_len: int = 96,
        embedding_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.num_actions = num_actions
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(type_vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_actions),
        )

    def _prepare(self, token_ids, type_ids=None) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(token_ids):
            token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        token_ids = token_ids.long()
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids[:, : self.max_seq_len]

        if type_ids is None:
            type_ids = torch.zeros_like(token_ids)
        elif not torch.is_tensor(type_ids):
            type_ids = torch.as_tensor(type_ids, dtype=torch.long, device=token_ids.device)
        type_ids = type_ids.long()
        if type_ids.ndim == 1:
            type_ids = type_ids.unsqueeze(0)
        type_ids = type_ids[:, : self.max_seq_len]
        return token_ids, type_ids

    def logits(self, token_ids, type_ids=None):
        token_ids, type_ids = self._prepare(token_ids, type_ids)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.type_embedding(type_ids) + self.position_embedding(positions)
        padding_mask = token_ids.eq(0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)

    def forward(self, token_ids, type_ids=None):
        return torch.softmax(self.logits(token_ids, type_ids), dim=1)

    @torch.no_grad()
    def predict(self, token_ids, type_ids=None, device: str | torch.device = "cpu") -> np.ndarray:
        self.eval()
        self.to(device)
        token_tensor = torch.as_tensor(token_ids, dtype=torch.long, device=device)
        type_tensor = None if type_ids is None else torch.as_tensor(type_ids, dtype=torch.long, device=device)
        return self.forward(token_tensor, type_tensor).detach().cpu().numpy()


def save_checkpoint(path: str | Path, model: BabyAIPredictor, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": {
            "vocab_size": model.vocab_size,
            "type_vocab_size": model.type_vocab_size,
            "num_actions": model.num_actions,
            "max_seq_len": model.max_seq_len,
            "embedding_dim": model.embedding_dim,
            **(metadata or {}),
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> BabyAIPredictor:
    payload = torch.load(path, map_location=map_location)
    metadata = payload.get("metadata", {})
    model = BabyAIPredictor(
        vocab_size=int(metadata["vocab_size"]),
        type_vocab_size=int(metadata["type_vocab_size"]),
        num_actions=int(metadata.get("num_actions", 7)),
        max_seq_len=int(metadata.get("max_seq_len", 96)),
        embedding_dim=int(metadata.get("embedding_dim", 64)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
