from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.metrics import ranking_metrics_at_k
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def evaluate_recommender_on_split(
    recommender: Any,
    split_rows: list[dict[str, Any]],
    ks: Sequence[int],
    exclude_seen: bool = True,
) -> dict[str, float]:
    max_k = max(ks)
    recommendations: dict[int, list[int]] = {}
    targets: dict[int, int] = {}

    for row in split_rows:
        user_idx = int(row["user_idx"])
        context = [int(item) for item in row["context"]]
        recs = recommender.recommend(
            user_idx=user_idx,
            history=context,
            top_k=max_k,
            exclude_seen=exclude_seen,
        )
        recommendations[user_idx] = [int(item) for item, _ in recs]
        targets[user_idx] = int(row["target_item"])

    return ranking_metrics_at_k(recommendations, targets, ks)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def load_model(model_name: str, models_dir: Path, config: dict) -> Any:
    if model_name in {"popularity", "itemknn", "als"}:
        return _load_pickle(models_dir / f"{model_name}.pkl")
    if model_name == "transformer":
        from src.models.transformer import load_transformer_recommender

        return load_transformer_recommender(models_dir / "transformer.pt", config)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_models(
    config: dict,
    model_names: Sequence[str],
    split_name: str = "test",
) -> dict[str, dict[str, float]]:
    processed_dir = Path(config["paths"]["processed_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    split_rows = load_jsonl(processed_dir / f"{split_name}.jsonl")
    ks = [int(k) for k in config["evaluation"]["ks"]]
    exclude_seen = bool(config["evaluation"]["exclude_seen"])

    all_metrics: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        model = load_model(model_name, models_dir, config)
        metrics = evaluate_recommender_on_split(model, split_rows, ks, exclude_seen)
        all_metrics[model_name] = metrics
        logger.info("%s metrics on %s: %s", model_name, split_name, metrics)

    output_path = metrics_dir / f"evaluation_{split_name}.json"
    output_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    logger.info("Saved evaluation metrics to %s", output_path)
    return all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved recommenders.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["popularity", "itemknn", "als", "transformer"],
        help="Model names to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    evaluate_models(config, args.models, args.split)


if __name__ == "__main__":
    main()

