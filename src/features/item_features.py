from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def build_item_features(config: dict) -> pd.DataFrame:
    processed_dir = Path(config["paths"]["processed_dir"])
    items = pd.read_csv(processed_dir / "items_processed.csv")
    interactions = pd.read_csv(processed_dir / "interactions_processed.csv")

    popularity = (
        interactions.groupby("item_idx")["event_weight"]
        .sum()
        .rename("weighted_popularity")
        .reset_index()
    )

    features = items.merge(popularity, on="item_idx", how="left")
    features["weighted_popularity"] = features["weighted_popularity"].fillna(0.0)
    features["log_weighted_popularity"] = np.log(
        features["weighted_popularity"].astype(float).clip(lower=0) + 1.0
    )
    features.to_csv(processed_dir / "item_features.csv", index=False)
    logger.info("Saved item features to %s", processed_dir / "item_features.csv")
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build item-level features.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    build_item_features(config)


if __name__ == "__main__":
    main()
