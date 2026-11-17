"""Decision-margin cutoffs for LinearSVC ACD; margins are NOT probabilities."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from shared.labels import ASPECT_CODES

THRESHOLDS_FILE = 'aspect_thresholds.json'

def aspect_margins(model, x) -> np.ndarray:
    scores = np.asarray(model.decision_function(x), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(ASPECT_CODES):
        raise ValueError('Expected one signed margin per aspect category')
    return scores

def apply_thresholds(margins: np.ndarray, thresholds: dict[str, float] | None) -> np.ndarray:
    cutoffs = np.array([(thresholds or {}).get(c, 0.0) for c in ASPECT_CODES])
    return (np.asarray(margins) >= cutoffs).astype(np.int8)

def tune_thresholds(
    gold: np.ndarray, margins: np.ndarray, *,
    min_positives: int = 10, max_positive_rate: float = 0.5,
) -> dict[str, float]:
    """Use TRAINING out-of-fold margins; cutoffs are candidate margin percentiles.

    Rare labels retain the native decision boundary 0; every label's prediction
    rate is capped to avoid selecting an almost-all-positive classifier.
    """
    gold = np.asarray(gold)
    margins = np.asarray(margins)
    if gold.shape != margins.shape or gold.shape[1] != len(ASPECT_CODES):
        raise ValueError('Gold and margins must have shape (n, 11)')
    if not 0 < max_positive_rate <= 1:
        raise ValueError('max_positive_rate must be in (0,1]')
    output = {}
    for j, code in enumerate(ASPECT_CODES):
        column = margins[:, j]
        if gold[:, j].sum() < min_positives:
            output[code] = 0.0
            continue
        choices = np.unique(np.concatenate(([0.0], np.percentile(column, np.arange(10, 100, 5)))))
        # Prefer the native boundary at ties.
        choices = sorted(choices, key=lambda value: (abs(value), value))
        best, best_f1 = 0.0, -1.0
        for cut in choices:
            prediction = (column >= cut).astype(np.int8)
            if prediction.mean() > max_positive_rate:
                continue
            score = f1_score(gold[:, j], prediction, zero_division=0)
            if score > best_f1 + 1e-12:
                best, best_f1 = float(cut), score
        output[code] = best
    return output

def save_thresholds(directory: str | Path, values: dict[str, float]) -> Path:
    path = Path(directory) / THRESHOLDS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
    return path

def load_thresholds(directory: str | Path) -> dict[str, float] | None:
    path = Path(directory) / THRESHOLDS_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if set(data) != set(ASPECT_CODES):
        raise ValueError('Threshold file must have exactly A01–A11')
    return {k: float(v) for k, v in data.items()}
