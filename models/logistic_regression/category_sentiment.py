"""Optional LR-only category-level sentiment head for sentence-only inference.

The existing benchmark sentiment model is TARGET-CONDITIONED and is NOT changed.
This additional model learns sentiment labels per (comment, aspect category),
including mixed sentiment when distinct targets share one category. It uses the
already fitted shared sentiment TF-IDF vectorizer without refitting it.

Run after training the existing LR classifiers:
    python -m models.logistic_regression.category_sentiment \\
        --artifacts models/logistic_regression/artifacts/tuned

Validation metrics saved here are a DIFFERENT task from the target-conditioned
sentiment macro-F1 and must never be substituted into the model comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, hamming_loss,
    precision_score, recall_score,
)
from sklearn.multiclass import OneVsRestClassifier

from shared.data import WORKBOOK_DEFAULT, read_annotation_sheet
from shared.labels import ASPECT_NAMES, SENTIMENT_LABELS, SENTIMENT_TO_INDEX, normalize_aspect
from shared.tfidf import load_sentiment_vectorizer

MODEL_FILE = "category_sentiment_model.joblib"
METRICS_FILE = "category_sentiment_validation.json"
DEFAULT_SENTIMENT_THRESHOLD = 0.5


def make_category_input(model_text: str, aspect_code: str) -> str:
    """Training/inference-aligned category input, never a fabricated target phrase."""
    code = normalize_aspect(aspect_code)
    text = str(model_text).strip()
    if not text:
        raise ValueError("model_text must not be blank")
    return f"COMMENT: {text}  ASPECT_CATEGORY: {ASPECT_NAMES[code]}"


def category_examples(frame: pd.DataFrame) -> tuple[list[str], np.ndarray, list[dict]]:
    """Aggregate (comment, category) labels; keep conflicting target sentiments.

    A comment can have multiple categories, and one category can have both
    Positive and Negative targets. We use multi-hot sentiment labels rather
    than choosing one arbitrarily or dropping conflicting annotations.
    """
    examples: list[str] = []
    labels: list[np.ndarray] = []
    manifest: list[dict] = []
    for (comment_id, aspect_code), group in frame.groupby(
        ["comment_id", "aspect_code"], sort=False
    ):
        versions = group["model_text"].unique()
        if len(versions) != 1:
            raise ValueError(f"Comment {comment_id!r} has inconsistent model_text")
        code = normalize_aspect(str(aspect_code))
        y = np.zeros(len(SENTIMENT_LABELS), dtype=np.int8)
        for sentiment in group["sentiment"]:
            y[SENTIMENT_TO_INDEX[str(sentiment)]] = 1
        examples.append(make_category_input(str(versions[0]), code))
        labels.append(y)
        manifest.append({
            "comment_id": str(comment_id),
            "aspect_code": code,
            "gold_sentiments": [name for name, flag in zip(SENTIMENT_LABELS, y) if flag],
        })
    if not labels:
        raise ValueError("No (comment, aspect) examples found")
    return examples, np.asarray(labels, dtype=np.int8), manifest


def select_sentiments(proba: np.ndarray, threshold: float = DEFAULT_SENTIMENT_THRESHOLD) -> np.ndarray:
    """Allow mixed sentiments; ensure at least one category sentiment per row."""
    scores = np.asarray(proba, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(SENTIMENT_LABELS):
        raise ValueError("Category sentiment model must return three probabilities")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Category sentiment probabilities must be finite and in [0, 1]")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    pred = (scores >= threshold).astype(np.int8)
    empty = ~pred.any(axis=1)
    if empty.any():
        pred[np.flatnonzero(empty), scores[empty].argmax(axis=1)] = 1
    return pred


def category_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "per_sentiment": classification_report(
            y_true, y_pred, target_names=list(SENTIMENT_LABELS),
            output_dict=True, zero_division=0,
        ),
    }


def train_category_sentiment(
    *,
    artifacts: str | Path = "models/logistic_regression/artifacts/tuned",
    features: str | Path = "models/tfidf",
    workbook: str | Path = WORKBOOK_DEFAULT,
    c: float = 1.0,
    max_iter: int = 5000,
    class_weight: str | None = "balanced",
    threshold: float = DEFAULT_SENTIMENT_THRESHOLD,
) -> dict:
    """Train category head on Training; report separate category-level Validation."""
    if c <= 0 or max_iter < 1:
        raise ValueError("C and max_iter must be positive")
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be 'balanced' or None")
    artifacts_dir = Path(artifacts)
    for required in ("aspect_model.joblib", "sentiment_model.joblib"):
        if not (artifacts_dir / required).is_file():
            raise FileNotFoundError(f"Train the original LR benchmark first: {artifacts_dir / required}")
    vectorizer = load_sentiment_vectorizer(features)
    train_df = read_annotation_sheet(workbook, "Training")
    val_df = read_annotation_sheet(workbook, "Validation")
    train_texts, y_train, train_manifest = category_examples(train_df)
    val_texts, y_val, val_manifest = category_examples(val_df)
    x_train = vectorizer.transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    if any((y_train.sum(axis=0) == 0) | (y_train.sum(axis=0) == len(y_train))):
        raise ValueError("Every sentiment must have positive and negative Training examples")

    classifier = OneVsRestClassifier(LogisticRegression(
        C=c, max_iter=max_iter, class_weight=class_weight, random_state=42,
    ))
    classifier.fit(x_train, y_train)
    pred = select_sentiments(classifier.predict_proba(x_val), threshold)
    result = {
        "task": "category_level_multilabel_sentiment",
        "scope": "Conditional on GOLD aspect categories (not end-to-end aspect detection)",
        "evaluation_partition": "Validation",
        "is_original_benchmark_metric": False,
        "warning": "Do not compare to target-conditioned sentiment macro-F1 or claim target-level sentiment extraction.",
        "hyperparameters": {"C": c, "max_iter": max_iter, "class_weight": class_weight, "threshold": threshold},
        "train_category_rows": len(train_texts),
        "val_category_rows": len(val_texts),
        "train_mixed_sentiment_categories": sum(len(row["gold_sentiments"]) > 1 for row in train_manifest),
        "val_mixed_sentiment_categories": sum(len(row["gold_sentiments"]) > 1 for row in val_manifest),
        "metrics": category_metrics(y_val, pred),
    }
    joblib.dump(classifier, artifacts_dir / MODEL_FILE)
    (artifacts_dir / METRICS_FILE).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="models/logistic_regression/artifacts/tuned")
    parser.add_argument("--features", default="models/tfidf")
    parser.add_argument("--workbook", default=str(WORKBOOK_DEFAULT))
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SENTIMENT_THRESHOLD)
    args = parser.parse_args(argv)
    try:
        report = train_category_sentiment(
            artifacts=args.artifacts, features=args.features, workbook=args.workbook,
            c=args.C, max_iter=args.max_iter,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            threshold=args.threshold,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Category sentiment training failed: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
