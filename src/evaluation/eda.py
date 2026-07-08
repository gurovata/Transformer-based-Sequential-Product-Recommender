from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def run_eda(config: dict) -> dict:
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_csv(raw_dir / "interactions.csv")
    items = pd.read_csv(raw_dir / "items.csv")
    sequences = []
    with (processed_dir / "sequences.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            sequences.append(json.loads(line))

    sequence_lengths = pd.Series([len(row["items"]) for row in sequences], name="length")
    item_counts = interactions["item_id"].value_counts()

    n_users = interactions["user_id"].nunique()
    n_items = items["item_id"].nunique()
    n_interactions = len(interactions)
    sparsity = 1.0 - (n_interactions / max(n_users * n_items, 1))
    cold_start_ratio = float((item_counts <= 1).sum() / max(n_items, 1))

    summary = {
        "num_users": int(n_users),
        "num_items": int(n_items),
        "num_interactions": int(n_interactions),
        "avg_sequence_length": float(sequence_lengths.mean()),
        "median_sequence_length": float(sequence_lengths.median()),
        "cold_start_ratio_items_with_1_or_fewer_events": cold_start_ratio,
        "sparsity": float(sparsity),
        "top_categories": interactions["item_category"].value_counts().head(10).to_dict(),
        "top_items": item_counts.head(10).astype(int).to_dict(),
    }

    sequence_lengths.value_counts().sort_index().rename_axis("sequence_length").reset_index(
        name="num_users"
    ).to_csv(reports_dir / "sequence_length_distribution.csv", index=False)
    (reports_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info("Saved EDA summary to %s", reports_dir / "eda_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight EDA.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    run_eda(config)


if __name__ == "__main__":
    main()

