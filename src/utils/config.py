from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must contain a mapping.")
    return data


def ensure_dirs(config: dict[str, Any]) -> None:
    for key, value in config.get("paths", {}).items():
        if key.endswith("_dir"):
            Path(value).mkdir(parents=True, exist_ok=True)
