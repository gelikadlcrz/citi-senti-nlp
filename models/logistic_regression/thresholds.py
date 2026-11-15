"""Per-aspect decision thresholds for the one-vs-rest Logistic Regression detector.

OneVsRestClassifier.predict() uses a fixed 0.5 probability cutoff for every
aspect. With few positive examples, rare aspects may never cross 0.5, which
pulls down macro-F1. Here each aspect gets its own cutoff, chosen on VALIDATION
only and saved next to the model so evaluation and inference use the same one.

Note: validation scores computed with thresholds tuned on validation are
optimistic. The held-out Testing partition gives the unbiased number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from shared.labels import ASPECT_CODES

THRESHOLDS_FILE = "aspect_thresholds.json"
DEFAULT_THRESHOLD = 0.5
DEFAULT_GRID = tuple(round(t, 2) for t in np.arange(0.05, 0.951, 0.05))


def aspect_probabilities(model, x) -> np.ndarray:
    """Positive-class probability per aspect, shape (n_samples, 11)."""
    proba = model.predict_proba(x)
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != len(ASPECT_CODES):
        raise ValueError("Aspect model must return one probability column per aspect")
    return proba


def tune_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid: tuple[float, ...] = DEFAULT_GRID,
) -> dict[str, float]:
    """Pick the F1-maximising cutoff for each aspect independently.

    Aspects with no positive validation examples keep the default 0.5, since
    there is no evidence to tune on. Ties go to the cutoff closest to 0.5,
    which is the least aggressive change.
    """
    y_true = np.asarray(y_true)
    thresholds: dict[str, float] = {}
    for j, code in enumerate(ASPECT_CODES):
        if y_true[:, j].sum() == 0:
            thresholds[code] = DEFAULT_THRESHOLD
            continue
        best_t, best_f1 = DEFAULT_THRESHOLD, -1.0
        for t in sorted(grid, key=lambda v: abs(v - DEFAULT_THRESHOLD)):
            f1 = f1_score(y_true[:, j], (proba[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1 + 1e-12:
                best_t, best_f1 = float(t), f1
        thresholds[code] = best_t
    return thresholds


def apply_thresholds(proba: np.ndarray, thresholds: dict[str, float] | None) -> np.ndarray:
    """Turn probabilities into a 0/1 multi-hot matrix using per-aspect cutoffs."""
    cutoffs = np.array([
        (thresholds or {}).get(code, DEFAULT_THRESHOLD) for code in ASPECT_CODES
    ])
    return (proba >= cutoffs).astype(np.int8)


def save_thresholds(directory: str | Path, thresholds: dict[str, float]) -> Path:
    path = Path(directory) / THRESHOLDS_FILE
    path.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    return path


def load_thresholds(directory: str | Path) -> dict[str, float] | None:
    """Return saved thresholds, or None if the artifacts were never tuned."""
    path = Path(directory) / THRESHOLDS_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - set(ASPECT_CODES)
    if unknown:
        raise ValueError(f"{path} has unknown aspect codes: {sorted(unknown)}")
    return {code: float(value) for code, value in data.items()}