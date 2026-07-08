from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluate import evaluate_recommender_on_split, load_jsonl
from src.models.als import ALSRecommender
from src.models.item_knn import ItemKNNRecommender
from src.models.popularity import PopularityRecommender
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


logger = logging.getLogger(__name__)


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    return json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))


def _train_interactions_from_sequences(processed_dir: Path) -> pd.DataFrame:
    rows = []
    for row in load_jsonl(processed_dir / "train_sequences.jsonl"):
        items = [int(item) for item in row["items"]]
        for position, item_idx in enumerate(items):
            rows.append(
                {
                    "user_idx": int(row["user_idx"]),
                    "item_idx": item_idx,
                    "event_weight": 1.0 + (position / max(len(items), 1)),
                }
            )
    return pd.DataFrame(rows)


def _save_pickle(model: object, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(model, file)


def train_baselines(config: dict) -> dict[str, dict[str, float]]:
    set_seed(int(config["seed"]))
    processed_dir = Path(config["paths"]["processed_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(processed_dir)
    train_interactions = _train_interactions_from_sequences(processed_dir)
    validation_rows = load_jsonl(processed_dir / "validation.jsonl")
    ks = [int(k) for k in config["evaluation"]["ks"]]
    exclude_seen = bool(config["evaluation"]["exclude_seen"])

    num_users = int(metadata["num_users"])
    vocab_size = int(metadata["vocab_size"])

    models = {
        "popularity": PopularityRecommender(vocab_size).fit(train_interactions),
        "itemknn": ItemKNNRecommender(
            num_users=num_users,
            num_items=vocab_size,
            top_neighbors=int(config["baselines"]["item_knn_neighbors"]),
        ).fit(train_interactions),
        "als": ALSRecommender(
            num_users=num_users,
            num_items=vocab_size,
            factors=int(config["baselines"]["als_factors"]),
            iterations=int(config["baselines"]["als_iterations"]),
        ).fit(train_interactions),
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        _save_pickle(model, models_dir / f"{name}.pkl")
        metrics[name] = evaluate_recommender_on_split(
            model, validation_rows, ks, exclude_seen=exclude_seen
        )
        logger.info("%s validation metrics: %s", name, metrics[name])

    output_path = metrics_dir / "baselines_validation.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved baseline metrics to %s", output_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline recommenders.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    train_baselines(config)


if __name__ == "__main__":
    main()

