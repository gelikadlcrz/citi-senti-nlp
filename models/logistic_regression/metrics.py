"""Fixed-label 33-pair metrics and group-bootstrap CIs for LR research protocol."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from shared.labels import ASPECT_CODES, SENTIMENT_LABELS

PAIRS = tuple((code, sentiment) for code in ASPECT_CODES for sentiment in SENTIMENT_LABELS)
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}


def pair_arrays(gold, predicted):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Gold and prediction must be equally sized nonempty lists")
    truth = np.zeros((len(gold), len(PAIRS)), dtype=np.int8)
    pred = np.zeros_like(truth)
    for row, (g, p) in enumerate(zip(gold, predicted)):
        for pair in set(g):
            truth[row, PAIR_INDEX[tuple(pair)]] = 1
        for pair in set(p):
            pred[row, PAIR_INDEX[tuple(pair)]] = 1
    return truth, pred


def metrics(gold, predicted):
    y, p = pair_arrays(gold, predicted)
    supported = np.flatnonzero(y.sum(axis=0) > 0)
    return {
        "macro_f1_33_pairs": float(f1_score(y, p, average="macro", zero_division=0)),
        "supported_macro_f1": float(f1_score(y[:, supported], p[:, supported], average="macro", zero_division=0)) if supported.size else 0.0,
        "supported_pairs": int(supported.size),
        "micro_f1": float(f1_score(y, p, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(y, p, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, p, average="micro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y, p)),
        "gold_pairs": int(y.sum()), "predicted_pairs": int(p.sum()),
        "support_by_pair": {f"{a}|{s}": int(y[:, j].sum()) for j, (a, s) in enumerate(PAIRS)},
        "f1_by_pair": _per_pair_f1(y, p),
    }


def _per_pair_f1(y, p):
    tp = np.sum((y == 1) & (p == 1), axis=0)
    fp = np.sum((y == 0) & (p == 1), axis=0)
    fn = np.sum((y == 1) & (p == 0), axis=0)
    denominator = 2*tp + fp + fn
    score = np.divide(2*tp, denominator, out=np.zeros(len(PAIRS)), where=denominator != 0)
    return {f"{a}|{s}": float(score[j]) for j, (a, s) in enumerate(PAIRS)}


def group_bootstrap_ci(gold, predicted, groups, *, replicates=10000, seed=42):
    """Group-level unadjusted 95% percentile interval, resampling merged groups.

    Reuses the same fixed 33-label denominator per bootstrap sample.
    Paired comparisons require common resampled groups and must be performed
    across the three model families in the team's shared analysis.
    """
    y, p = pair_arrays(gold, predicted)
    groups = np.asarray(groups)
    if groups.shape != (len(y),):
        raise ValueError("One group identifier per comment is required")
    unique, inv = np.unique(groups, return_inverse=True)
    n = len(unique)
    tp = np.zeros((n, len(PAIRS)), dtype=np.int32)
    fp = tp.copy()
    fn = tp.copy()
    for idx in range(n):
        current = inv == idx
        tp[idx] = np.sum((y[current] == 1) & (p[current] == 1), axis=0)
        fp[idx] = np.sum((y[current] == 0) & (p[current] == 1), axis=0)
        fn[idx] = np.sum((y[current] == 1) & (p[current] == 0), axis=0)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        draws = rng.integers(0, n, size=n)
        a, b, c = tp[draws].sum(axis=0), fp[draws].sum(axis=0), fn[draws].sum(axis=0)
        denominator = 2*a + b + c
        values.append(float(np.divide(2*a, denominator, out=np.zeros(len(PAIRS)), where=denominator > 0).mean()))
    return {"group_count": n, "replicates": replicates, "seed": seed,
            "unadjusted_percentile_95_ci": [float(v) for v in np.quantile(values, [.025, .975])]}
