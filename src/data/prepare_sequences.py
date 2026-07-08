from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.split import leave_two_out
from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging


PAD_ID = 0
MASK_ID = 1
FIRST_ITEM_ID = 2

logger = logging.getLogger(__name__)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _event_weight(event_type: str, event_weights: dict[str, float]) -> float:
    return float(event_weights.get(event_type, 1.0))


def prepare_sequences(config: dict) -> None:
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = raw_dir / "interactions.csv"
    items_path = raw_dir / "items.csv"
    if not interactions_path.exists() or not items_path.exists():
        raise FileNotFoundError(
            "Raw data is missing. Run `python -m src.data.generate_synthetic` first."
        )

    interactions = pd.read_csv(interactions_path, parse_dates=["timestamp"])
    items = pd.read_csv(items_path)

    allowed_events = set(config["preprocessing"]["interaction_event_types"])
    interactions = interactions.loc[interactions["event_type"].isin(allowed_events)].copy()
    interactions = interactions.sort_values(["user_id", "timestamp", "position"])

    user_ids = sorted(interactions["user_id"].unique())
    item_ids = sorted(items["item_id"].unique())

    user_mapping = pd.DataFrame(
        {"user_id": user_ids, "user_idx": list(range(len(user_ids)))}
    )
    item_mapping = pd.DataFrame(
        {
            "item_id": item_ids,
            "item_idx": list(range(FIRST_ITEM_ID, FIRST_ITEM_ID + len(item_ids))),
        }
    )

    category_ids = {
        category: idx + 1
        for idx, category in enumerate(sorted(items["item_category"].dropna().unique()))
    }
    price_bucket_ids = {
        bucket: idx + 1
        for idx, bucket in enumerate(sorted(items["item_price_bucket"].dropna().unique()))
    }

    items_processed = items.merge(item_mapping, on="item_id", how="left")
    items_processed["category_idx"] = items_processed["item_category"].map(category_ids).fillna(0)
    items_processed["price_bucket_idx"] = (
        items_processed["item_price_bucket"].map(price_bucket_ids).fillna(0)
    )

    event_weights = config["data_generation"]["event_weights"]
    interactions_processed = interactions.merge(user_mapping, on="user_id", how="left").merge(
        item_mapping, on="item_id", how="left"
    )
    interactions_processed["event_weight"] = interactions_processed["event_type"].map(
        lambda value: _event_weight(str(value), event_weights)
    )

    min_sequence_length = int(config["preprocessing"]["min_sequence_length"])

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []

    grouped = interactions_processed.groupby(["user_idx", "user_id"], sort=True)
    for (user_idx, user_id), group in grouped:
        sequence = group["item_idx"].astype(int).tolist()
        if len(sequence) < min_sequence_length:
            continue

        split = leave_two_out(sequence, min_train_length=max(1, min_sequence_length - 2))
        train_rows.append(
            {"user_idx": int(user_idx), "user_id": int(user_id), "items": split.train}
        )
        validation_rows.append(
            {
                "user_idx": int(user_idx),
                "user_id": int(user_id),
                "context": split.train,
                "target_item": split.validation,
            }
        )
        test_rows.append(
            {
                "user_idx": int(user_idx),
                "user_id": int(user_id),
                "context": split.train + [split.validation],
                "target_item": split.test,
            }
        )
        sequence_rows.append(
            {
                "user_idx": int(user_idx),
                "user_id": int(user_id),
                "items": sequence,
                "validation_item": split.validation,
                "test_item": split.test,
            }
        )

    user_mapping.to_csv(processed_dir / "user_mapping.csv", index=False)
    item_mapping.to_csv(processed_dir / "item_mapping.csv", index=False)
    items_processed.to_csv(processed_dir / "items_processed.csv", index=False)
    interactions_processed.to_csv(processed_dir / "interactions_processed.csv", index=False)
    _write_jsonl(processed_dir / "train_sequences.jsonl", train_rows)
    _write_jsonl(processed_dir / "validation.jsonl", validation_rows)
    _write_jsonl(processed_dir / "test.jsonl", test_rows)
    _write_jsonl(processed_dir / "sequences.jsonl", sequence_rows)

    metadata = {
        "pad_id": PAD_ID,
        "mask_id": MASK_ID,
        "first_item_id": FIRST_ITEM_ID,
        "num_users": len(train_rows),
        "num_raw_users": len(user_ids),
        "num_items": len(item_ids),
        "vocab_size": FIRST_ITEM_ID + len(item_ids),
        "max_item_idx": FIRST_ITEM_ID + len(item_ids) - 1,
        "num_categories": len(category_ids),
        "num_price_buckets": len(price_bucket_ids),
        "min_sequence_length": min_sequence_length,
    }
    (processed_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info(
        "Prepared %s user sequences, vocab_size=%s, processed_dir=%s",
        len(train_rows),
        metadata["vocab_size"],
        processed_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leave-two-out sequences.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    prepare_sequences(config)


if __name__ == "__main__":
    main()

