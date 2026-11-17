"""Load SVM-only configuration; does not affect Logistic Regression."""
from __future__ import annotations
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if cfg.get('model') != 'linear_svm':
        raise ValueError(f'{path}: model must be linear_svm')
    return cfg
