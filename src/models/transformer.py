from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


class SASRecModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 0,
        item_to_category: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        if item_to_category is not None:
            category_tensor = torch.tensor(item_to_category, dtype=torch.long)
            category_vocab_size = int(category_tensor.max().item()) + 1
            self.register_buffer("item_to_category", category_tensor)
            self.category_embedding = nn.Embedding(
                category_vocab_size, embedding_dim, padding_idx=0
            )
        else:
            self.item_to_category = None  # type: ignore[assignment]
            self.category_embedding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        if self.category_embedding is not None:
            category_ids = self.item_to_category[input_ids]
            x = x + self.category_embedding(category_ids)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = input_ids.eq(self.pad_id)
        encoded = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output(encoded)


@dataclass
class TransformerRecommender:
    model: SASRecModel
    max_seq_len: int
    vocab_size: int
    pad_id: int = 0
    mask_id: int = 1
    popularity_scores: np.ndarray | None = None
    device: str = "cpu"

    def recommend(
        self,
        user_idx: int | None = None,
        history: Sequence[int] | None = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        history = [int(item) for item in (history or []) if 2 <= int(item) < self.vocab_size]
        if not history:
            return self._popular(top_k, [], exclude_seen)

        truncated = history[-self.max_seq_len :]
        input_ids = truncated + [self.pad_id] * (self.max_seq_len - len(truncated))
        last_position = len(truncated) - 1

        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            logits = self.model(tensor)[0, last_position].detach().cpu().numpy()

        return self._top_from_scores(logits, top_k, history if exclude_seen else [])

    def _popular(
        self, top_k: int, seen: Sequence[int], exclude_seen: bool
    ) -> list[tuple[int, float]]:
        if self.popularity_scores is None:
            return []
        scores = self.popularity_scores.copy()
        return self._top_from_scores(scores, top_k, seen if exclude_seen else [])

    def _top_from_scores(
        self, scores: np.ndarray, top_k: int, seen: Sequence[int]
    ) -> list[tuple[int, float]]:
        scores = scores.astype(np.float32, copy=True)
        scores[:2] = -np.inf
        for item in seen:
            if 0 <= item < len(scores):
                scores[item] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            return []

        k = min(top_k, int(finite.sum()))
        top_indices = np.argpartition(-scores, kth=np.arange(k))[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(item), float(scores[item])) for item in top_indices]


def load_transformer_recommender(
    checkpoint_path: str | Path,
    config: dict | None = None,
    device: str = "cpu",
) -> TransformerRecommender:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Transformer checkpoint not found at {path}. "
            "Run `python -m src.training.train_transformer` first."
        )

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model_config = checkpoint["model_config"]
    model = SASRecModel(
        vocab_size=int(model_config["vocab_size"]),
        max_seq_len=int(model_config["max_seq_len"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        pad_id=int(model_config.get("pad_id", 0)),
        item_to_category=checkpoint.get("item_to_category"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    popularity = checkpoint.get("popularity_scores")
    popularity_scores = None if popularity is None else np.asarray(popularity, dtype=np.float32)

    return TransformerRecommender(
        model=model,
        max_seq_len=int(model_config["max_seq_len"]),
        vocab_size=int(model_config["vocab_size"]),
        pad_id=int(model_config.get("pad_id", 0)),
        mask_id=int(model_config.get("mask_id", 1)),
        popularity_scores=popularity_scores,
        device=device,
    )
