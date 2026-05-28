from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class MNISTTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = 28 * 28
        self.embed_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.output_tokens = 4
        self.output_dim = self.embed_dim * self.output_tokens

        self.input_embedding = nn.Linear(1, self.embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.output_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.reshape(-1, self.seq_len)
        elif x.ndim == 3:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            x = x.reshape(-1, self.seq_len)
        elif x.ndim == 4:
            x = x.reshape(-1, self.seq_len)
        else:
            raise ValueError(f"Unexpected input tensor shape {x.shape}")

        x = x.unsqueeze(-1)
        x = self.input_embedding(x)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.output_tokens)
        x = x.reshape(-1, self.output_dim)
        return x

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        x = self._prepare_input(x)
        logits = self.classifier(x)
        return torch.softmax(logits, dim=1)

    def logits(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        x = self._prepare_input(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict(self, x, device: str | torch.device = "cpu") -> np.ndarray:
        self.eval()
        self.to(device)
        probs = self.forward(torch.as_tensor(x, dtype=torch.float32, device=device))
        return probs.detach().cpu().numpy()

def save_checkpoint(path: str | Path, model: MNISTTransformerClassifier, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {"num_classes": model.num_classes}}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> MNISTTransformerClassifier:
    payload = torch.load(path, map_location=map_location)
    metadata = payload.get("metadata", {})
    model = MNISTTransformerClassifier(num_classes=int(metadata.get("num_classes", 10)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
