"""Metric correctness: hand-computed expected values on tiny examples."""
from __future__ import annotations

import numpy as np
import pytest

from shared.train_common import _metrics


def test_aspect_metrics_match_hand_computed_values():
    # 2 comments x 11 aspects. Comment 1: A01 correct, A02 missed.
    # Comment 2: A03 correct, A04 a false alarm. 2 wrong cells out of 22.
    y_true = np.zeros((2, 11), dtype=int)
    pred = np.zeros((2, 11), dtype=int)
    y_true[0, [0, 1]] = 1
    pred[0, 0] = 1
    y_true[1, 2] = 1
    pred[1, [2, 3]] = 1
    m = _metrics("aspect", y_true, pred)
    assert m["hamming_loss"] == pytest.approx(2 / 22)
    assert m["exact_match_accuracy"] == 0.0
    # micro: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3
    assert m["micro_f1"] == pytest.approx(2 / 3)
    # macro over 11 aspects: A01=1, A02=0, A03=1, A04=0, rest 0 (zero_division=0)
    assert m["macro_f1"] == pytest.approx(2 / 11)


def test_perfect_aspect_predictions():
    y = np.eye(11, dtype=int)
    m = _metrics("aspect", y, y.copy())
    assert m["hamming_loss"] == 0.0
    assert m["exact_match_accuracy"] == 1.0
    assert m["macro_f1"] == pytest.approx(1.0)


def test_sentiment_confusion_matrix_and_scores():
    # Labels: 0=Negative, 1=Neutral, 2=Positive
    y_true = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    m = _metrics("sentiment", y_true, pred)
    assert m["confusion_matrix"]["labels"] == ["Negative", "Neutral", "Positive"]
    assert m["confusion_matrix"]["matrix"] == [
        [2, 1, 0],   # true Negative: 2 right, 1 called Neutral
        [0, 1, 1],   # true Neutral: 1 right, 1 called Positive
        [0, 0, 1],   # true Positive: 1 right
    ]
    assert m["accuracy"] == pytest.approx(4 / 6)
    # F1 per class: Neg 0.8, Neu 0.5, Pos 2/3
    assert m["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_confusion_matrix_keeps_all_three_classes_when_one_is_missing():
    m = _metrics("sentiment", np.array([0, 0, 2]), np.array([0, 2, 2]))
    matrix = np.array(m["confusion_matrix"]["matrix"])
    assert matrix.shape == (3, 3)
    assert matrix[1].sum() == 0 and matrix[:, 1].sum() == 0