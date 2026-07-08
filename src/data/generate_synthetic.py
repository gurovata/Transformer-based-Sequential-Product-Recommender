from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


logger = logging.getLogger(__name__)


RELATED_CATEGORIES = {
    "mobile": ["internet", "devices", "finance"],
    "internet": ["tv", "entertainment", "cloud"],
    "tv": ["entertainment", "internet"],
    "finance": ["mobile", "travel"],
    "entertainment": ["tv", "mobile"],
    "cloud": ["internet", "devices"],
    "devices": ["mobile", "internet"],
    "travel": ["finance", "mobile"],
}


def _build_items(
    rng: np.random.Generator,
    num_items: int,
    categories: list[str],
    price_buckets: list[str],
) -> pd.DataFrame:
    category_probs = rng.dirichlet(np.ones(len(categories)) * 1.3)
    item_categories = rng.choice(categories, size=num_items, p=category_probs)
    popularity = rng.lognormal(mean=0.0, sigma=1.0, size=num_items)
    popularity = popularity / popularity.sum()

    price_by_category = {
        "mobile": [0.05, 0.45, 0.4, 0.1],
        "internet": [0.1, 0.5, 0.35, 0.05],
        "tv": [0.05, 0.35, 0.5, 0.1],
        "finance": [0.35, 0.4, 0.2, 0.05],
        "entertainment": [0.25, 0.5, 0.2, 0.05],
        "cloud": [0.05, 0.25, 0.45, 0.25],
        "devices": [0.0, 0.15, 0.45, 0.4],
        "travel": [0.05, 0.35, 0.45, 0.15],
    }

    rows = []
    for item_id, category, pop in zip(range(1, num_items + 1), item_categories, popularity):
        probs = price_by_category.get(category, [0.15, 0.4, 0.35, 0.1])
        price_bucket = rng.choice(price_buckets, p=probs)
        rows.append(
            {
                "item_id": item_id,
                "item_category": category,
                "item_price_bucket": price_bucket,
                "item_description": f"{category} product plan {item_id}",
                "base_popularity": float(pop),
            }
        )

    return pd.DataFrame(rows)


def _sample_next_category(
    rng: np.random.Generator,
    previous_category: str | None,
    preferred_categories: list[str],
    preference_weights: np.ndarray,
    categories: list[str],
) -> str:
    if previous_category and rng.random() < 0.42:
        related = RELATED_CATEGORIES.get(previous_category, [])
        if related:
            return str(rng.choice(related))

    if rng.random() < 0.78:
        return str(rng.choice(preferred_categories, p=preference_weights))

    return str(rng.choice(categories))


def _sample_item_from_category(
    rng: np.random.Generator,
    items: pd.DataFrame,
    category: str,
) -> int:
    category_items = items.loc[items["item_category"] == category]
    if category_items.empty:
        category_items = items
    probs = category_items["base_popularity"].to_numpy(dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice(category_items["item_id"].to_numpy(), p=probs))


def generate_synthetic_data(config: dict, output_dir: str | Path | None = None) -> None:
    seed = int(config["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    data_cfg = config["data_generation"]
    raw_dir = Path(output_dir or config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    categories = list(data_cfg["categories"])
    price_buckets = list(data_cfg["price_buckets"])
    num_users = int(data_cfg["num_users"])
    num_items = int(data_cfg["num_items"])

    items = _build_items(rng, num_items, categories, price_buckets)
    item_category_by_id = items.set_index("item_id")["item_category"].to_dict()

    event_types = np.array(["view", "cart", "purchase"])
    event_probs = np.array([0.58, 0.24, 0.18])

    interactions = []
    start_ts = pd.Timestamp("2025-01-01")

    for user_id in range(1, num_users + 1):
        seq_len = int(
            np.clip(
                rng.negative_binomial(n=8, p=0.35) + data_cfg["min_interactions"],
                data_cfg["min_interactions"],
                data_cfg["max_interactions"],
            )
        )
        n_preferences = int(rng.integers(2, min(4, len(categories)) + 1))
        preferred_categories = list(rng.choice(categories, size=n_preferences, replace=False))
        preference_weights = rng.dirichlet(np.ones(n_preferences) * 1.8)

        current_ts = start_ts + pd.Timedelta(days=int(rng.integers(0, 90)))
        previous_category: str | None = None

        for position in range(seq_len):
            category = _sample_next_category(
                rng,
                previous_category,
                preferred_categories,
                preference_weights,
                categories,
            )
            item_id = _sample_item_from_category(rng, items, category)
            previous_category = item_category_by_id[item_id]
            current_ts += pd.Timedelta(hours=int(rng.integers(2, 96)))
            event_type = str(rng.choice(event_types, p=event_probs))

            interactions.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": current_ts.isoformat(),
                    "event_type": event_type,
                    "item_category": previous_category,
                    "item_price_bucket": items.loc[
                        items["item_id"] == item_id, "item_price_bucket"
                    ].iloc[0],
                    "position": position,
                }
            )

    interactions_df = pd.DataFrame(interactions)
    interactions_df.to_csv(raw_dir / "interactions.csv", index=False)
    items.to_csv(raw_dir / "items.csv", index=False)
    logger.info(
        "Generated %s interactions for %s users and %s items in %s",
        len(interactions_df),
        num_users,
        num_items,
        raw_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic recommender data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    generate_synthetic_data(config, args.output_dir)


if __name__ == "__main__":
    main()

