from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.evaluate import evaluate_recommender_on_split, load_jsonl
from src.models.popularity import PopularityRecommender
from src.models.transformer import SASRecModel, TransformerRecommender
from src.training.train_baselines import _train_interactions_from_sequences
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


logger = logging.getLogger(__name__)


class NextItemDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], max_seq_len: int, pad_id: int = 0) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        for row in rows:
            items = [int(item) for item in row["items"]]
            if len(items) < 2:
                continue
            sequence = items[-(max_seq_len + 1) :]
            input_ids = sequence[:-1]
            labels = sequence[1:]
            pad_len = max_seq_len - len(input_ids)
            self.examples.append(
                (
                    input_ids + [pad_id] * pad_len,
                    labels + [pad_id] * pad_len,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.examples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    return json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))


def _build_item_to_category(processed_dir: Path, vocab_size: int) -> list[int]:
    item_to_category = np.zeros(vocab_size, dtype=np.int64)
    items = pd.read_csv(processed_dir / "items_processed.csv")
    for row in items.itertuples(index=False):
        item_to_category[int(row.item_idx)] = int(row.category_idx)
    return item_to_category.tolist()


def _build_popularity_scores(processed_dir: Path, vocab_size: int) -> np.ndarray:
    train_interactions = _train_interactions_from_sequences(processed_dir)
    popularity = PopularityRecommender(vocab_size).fit(train_interactions)
    if popularity.scores is None:
        return np.zeros(vocab_size, dtype=np.float32)
    scores = popularity.scores.copy()
    scores[~np.isfinite(scores)] = 0.0
    return scores.astype(np.float32)


def train_transformer(config: dict) -> dict[str, float]:
    set_seed(int(config["seed"]))
    processed_dir = Path(config["paths"]["processed_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(processed_dir)
    model_cfg = config["model"]
    pad_id = int(metadata["pad_id"])
    mask_id = int(metadata["mask_id"])
    vocab_size = int(metadata["vocab_size"])
    max_seq_len = int(model_cfg["max_seq_len"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = load_jsonl(processed_dir / "train_sequences.jsonl")
    validation_rows = load_jsonl(processed_dir / "validation.jsonl")
    dataset = NextItemDataset(train_rows, max_seq_len=max_seq_len, pad_id=pad_id)
    dataloader = DataLoader(
        dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )

    item_to_category = _build_item_to_category(processed_dir, vocab_size)
    model = SASRecModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embedding_dim=int(model_cfg["embedding_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        pad_id=pad_id,
        item_to_category=item_to_category,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    logger.info("Training Transformer on %s with %s examples.", device, len(dataset))
    for epoch in range(1, int(model_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        logger.info("Epoch %s loss %.4f", epoch, total_loss / max(total_batches, 1))

    popularity_scores = _build_popularity_scores(processed_dir, vocab_size)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "embedding_dim": int(model_cfg["embedding_dim"]),
            "num_heads": int(model_cfg["num_heads"]),
            "num_layers": int(model_cfg["num_layers"]),
            "dropout": float(model_cfg["dropout"]),
            "pad_id": pad_id,
            "mask_id": mask_id,
        },
        "item_to_category": item_to_category,
        "popularity_scores": popularity_scores.tolist(),
    }
    checkpoint_path = models_dir / "transformer.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved Transformer checkpoint to %s", checkpoint_path)

    recommender = TransformerRecommender(
        model=model,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        pad_id=pad_id,
        mask_id=mask_id,
        popularity_scores=popularity_scores,
        device=device,
    )
    metrics = evaluate_recommender_on_split(
        recommender,
        validation_rows,
        ks=[int(k) for k in config["evaluation"]["ks"]],
        exclude_seen=bool(config["evaluation"]["exclude_seen"]),
    )
    output_path = metrics_dir / "transformer_validation.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Transformer validation metrics: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SASRec-style Transformer.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    train_transformer(config)


if __name__ == "__main__":
    main()

