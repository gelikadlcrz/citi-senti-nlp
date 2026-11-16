"""Per-aspect decision thresholds for the one-vs-rest Logistic Regression detector.

OneVsRestClassifier.predict() uses a fixed 0.5 probability cutoff for every
aspect. Where that cutoff lands relative to the model's probabilities depends
on regularisation and class weighting, not only on how well the model separates
aspects, so each aspect gets its own cutoff instead.

Cutoffs are tuned on OUT-OF-FOLD probabilities from the Training partition
(5-fold cross-validation), so Validation never influences them and Validation
scores stay an honest estimate. Two guardrails stop degenerate cutoffs:
  * aspects with fewer than `min_positives` training examples keep 0.5;
  * a cutoff may not flag more than `max_positive_rate` of comments, which
    rules out "say yes to every comment" solutions that F1 can reward.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold

from shared.labels import ASPECT_CODES

THRESHOLDS_FILE = "aspect_thresholds.json"
DEFAULT_THRESHOLD = 0.5
DEFAULT_GRID = tuple(round(t, 2) for t in np.arange(0.05, 0.951, 0.05))
DEFAULT_MIN_POSITIVES = 10
DEFAULT_MAX_POSITIVE_RATE = 0.5


def aspect_probabilities(model, x) -> np.ndarray:
    """Positive-class probability per aspect, shape (n_samples, 11)."""
    proba = np.asarray(model.predict_proba(x))
    if proba.ndim != 2 or proba.shape[1] != len(ASPECT_CODES):
        raise ValueError("Aspect model must return one probability column per aspect")
    return proba


def out_of_fold_probabilities(
    make_model: Callable[[], object],
    x,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Probabilities for every Training row from a model that never saw that row.

    With `groups`, rows sharing a group (the same comment text) always fall in
    the same fold, so a model is never scored on text it was trained on.
    """
    y = np.asarray(y)
    if n_splits < 2 or n_splits > len(y):
        raise ValueError("n_splits must be between 2 and the number of training rows")
    proba = np.zeros(y.shape, dtype=float)
    if groups is not None:
        folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, held_idx in folds.split(x, groups=groups):
        model = make_model()
        model.fit(x[train_idx], y[train_idx])
        proba[held_idx] = aspect_probabilities(model, x[held_idx])
    return proba


def tune_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid: tuple[float, ...] = DEFAULT_GRID,
    *,
    min_positives: int = DEFAULT_MIN_POSITIVES,
    max_positive_rate: float = DEFAULT_MAX_POSITIVE_RATE,
) -> dict[str, float]:
    """Pick the F1-maximising cutoff for each aspect independently.

    Aspects with too few positive examples keep the default 0.5. Cutoffs that
    would flag more than `max_positive_rate` of rows are skipped. Ties go to
    the cutoff closest to 0.5, the least aggressive change.
    """
    y_true = np.asarray(y_true)
    if not 0 < max_positive_rate <= 1:
        raise ValueError("max_positive_rate must be in (0, 1]")
    thresholds: dict[str, float] = {}
    for j, code in enumerate(ASPECT_CODES):
        if y_true[:, j].sum() < min_positives:
            thresholds[code] = DEFAULT_THRESHOLD
            continue
        best_t, best_f1 = DEFAULT_THRESHOLD, -1.0
        for t in sorted(grid, key=lambda v: abs(v - DEFAULT_THRESHOLD)):
            pred = (proba[:, j] >= t).astype(int)
            if pred.mean() > max_positive_rate:
                continue
            f1 = f1_score(y_true[:, j], pred, zero_division=0)
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