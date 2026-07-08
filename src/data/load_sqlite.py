from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config import ensure_dirs, load_config
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def load_sqlite(config: dict) -> Path:
    raw_dir = Path(config["paths"]["raw_dir"])
    sqlite_path = Path(config["paths"]["sqlite_path"])
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_csv(raw_dir / "interactions.csv")
    interactions = interactions.reset_index().rename(columns={"index": "interaction_id"})
    interactions["interaction_id"] = interactions["interaction_id"] + 1

    items = pd.read_csv(raw_dir / "items.csv")
    users = interactions[["user_id"]].drop_duplicates().sort_values("user_id")

    schema = Path("sql/schema.sql").read_text(encoding="utf-8")
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(schema)
        for table in ["recommendations", "interactions", "items", "users"]:
            connection.execute(f"DELETE FROM {table}")

        users.to_sql("users", connection, if_exists="append", index=False)
        items.to_sql("items", connection, if_exists="append", index=False)
        interactions.to_sql("interactions", connection, if_exists="append", index=False)

    logger.info("Loaded synthetic data into SQLite database at %s", sqlite_path)
    return sqlite_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load synthetic data into SQLite.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    load_sqlite(config)


if __name__ == "__main__":
    main()

