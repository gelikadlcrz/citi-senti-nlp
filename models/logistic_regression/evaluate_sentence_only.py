"""Validation-only diagnostics for Logistic Regression's optional sentence-only path.

This is an END-TO-END (predicted aspect + category sentiment) PAIR evaluation,
not a replacement for either original comparison task. Never reads Testing.

Usage:
    python -m models.logistic_regression.evaluate_sentence_only \\
        --artifacts models/logistic_regression/artifacts/tuned
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.logistic_regression.category_sentiment import MODEL_FILE
from models.logistic_regression.predict import LogisticRegressionPredictor
from shared.data import WORKBOOK_DEFAULT, read_annotation_sheet
from shared.labels import ASPECT_CODES, SENTIMENT_LABELS

PAIR_LABELS = tuple((code, sentiment) for code in ASPECT_CODES for sentiment in SENTIMENT_LABELS)
PAIR_TO_INDEX = {pair: i for i, pair in enumerate(PAIR_LABELS)}


def pair_metrics(gold: list[set[tuple[str, str]]], pred: list[set[tuple[str, str]]]) -> dict:
    """Compare all (aspect, sentiment) pairs per comment, including mixed labels."""
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold and pred must be equally sized non-empty comment lists")
    y_true = np.zeros((len(gold), len(PAIR_LABELS)), dtype=np.int8)
    y_pred = np.zeros_like(y_true)
    for i, (g_pairs, p_pairs) in enumerate(zip(gold, pred)):
        for pair in g_pairs:
            if pair not in PAIR_TO_INDEX:
                raise ValueError(f"Unknown gold pair {pair}")
            y_true[i, PAIR_TO_INDEX[pair]] = 1
        for pair in p_pairs:
            if pair not in PAIR_TO_INDEX:
                raise ValueError(f"Unknown predicted pair {pair}")
            y_pred[i, PAIR_TO_INDEX[pair]] = 1
    return {
        "comment_count": len(gold),
        "gold_pairs": int(y_true.sum()),
        "predicted_pairs": int(y_pred.sum()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1_33_pairs": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_sentence_only(
    *,
    artifacts: str | Path = "models/logistic_regression/artifacts/tuned",
    features: str | Path = "models/tfidf",
    workbook: str | Path = WORKBOOK_DEFAULT,
) -> dict:
    """Evaluate the *whole* LR sentence-only path using VALIDATION only."""
    if not (Path(artifacts) / MODEL_FILE).is_file():
        raise FileNotFoundError("Train the optional LR category sentiment head first")
    val = read_annotation_sheet(workbook, "Validation")
    comments: dict[str, str] = {}
    gold_by_comment: dict[str, set[tuple[str, str]]] = {}
    for row in val.itertuples(index=False):
        cid = str(row.comment_id)
        text = str(row.model_text)
        if cid in comments and comments[cid] != text:
            raise ValueError(f"Comment {cid!r} has inconsistent model_text")
        comments[cid] = text
        gold_by_comment.setdefault(cid, set()).add((str(row.aspect_code), str(row.sentiment)))
    predictor = LogisticRegressionPredictor(artifacts=artifacts, features=features)
    predicted = []
    for comment in comments.values():
        response = predictor.predict_comment(comment)
        pairs = {
            (aspect["code"], sentiment)
            for aspect in response["aspects"]
            for sentiment in aspect["sentiments"]
        }
        predicted.append(pairs)
    report = {
        "model_type": "logreg",
        "task": "sentence_to_aspect_category_and_category_sentiment_pairs",
        "evaluation_partition": "Validation ONLY",
        "is_original_benchmark_metric": False,
        "warning": "Predicted aspect-category/sentiment pairs, not extracted target phrases. "
                   "Do not compare these numbers to the original target-conditioned sentiment macro-F1.",
        "metrics": pair_metrics(list(gold_by_comment.values()), predicted),
    }
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="models/logistic_regression/artifacts/tuned")
    parser.add_argument("--features", default="models/tfidf")
    parser.add_argument("--workbook", default=str(WORKBOOK_DEFAULT))
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    try:
        report = evaluate_sentence_only(
            artifacts=args.artifacts, features=args.features, workbook=args.workbook,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Sentence-only evaluation failed: {exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
