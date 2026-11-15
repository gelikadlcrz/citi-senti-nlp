"""Load models/logistic_regression/config.yaml so scripts share one source of settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("model") != "logistic_regression":
        raise ValueError(f"{path} is not a logistic_regression config")
    return data