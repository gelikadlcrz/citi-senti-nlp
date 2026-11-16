"""Baseline training utility using PRECOMPUTED SHARED TF-IDF matrices.

This is a reproducible starting experiment (not automatic hyperparameter search).
Both models receive precisely the same sparse X matrices for each task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from shared.features import load_features
from shared.labels import ASPECT_CODES, SENTIMENT_LABELS


CLASS_WEIGHT_CHOICES = ("none", "balanced")


def parse_class_weight(value: str | None) -> str | None:
    """Map CLI/YAML spellings ('none', None, 'balanced') to sklearn's argument."""
    if value is None or str(value).strip().lower() in ("none", "null", ""):
        return None
    if str(value).strip().lower() == "balanced":
        return "balanced"
    raise ValueError(f"class_weight must be one of {CLASS_WEIGHT_CHOICES}, got {value!r}")


def _estimator(model_type: str, c: float, max_iter: int, class_weight: str | None = None):
    if model_type == "logreg":
        return LogisticRegression(C=c, max_iter=max_iter, random_state=42, class_weight=class_weight)
    if model_type == "linear_svm":
        return LinearSVC(C=c, max_iter=max_iter, random_state=42, class_weight=class_weight)
    raise ValueError(f"Unknown model type: {model_type}")


def build_model(model_type: str, task: str, c: float, max_iter: int, class_weight: str | None = None):
    """One-vs-rest for multi-label aspects; a single multiclass model for sentiment."""
    base = _estimator(model_type, c, max_iter, class_weight)
    return OneVsRestClassifier(base) if task == "aspect" else base


def _metrics(task: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    if task == "aspect":
        # Each sample has 11 binary decisions. Exact-match is strict.
        return {
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
            "sample_f1": float(f1_score(y_true, pred, average="samples", zero_division=0)),
            "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
            "exact_match_accuracy": float(accuracy_score(y_true, pred)),
            # Share of the 11 yes/no aspect decisions that are wrong (lower is better).
            "hamming_loss": float(hamming_loss(y_true, pred)),
            "per_aspect": classification_report(
                y_true, pred, target_names=list(ASPECT_CODES),
                output_dict=True, zero_division=0,
            ),
        }
    return {
        "macro_f1": float(f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        # Rows = true label, columns = predicted label, in SENTIMENT_LABELS order.
        "confusion_matrix": {
            "labels": list(SENTIMENT_LABELS),
            "matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
        },
        "per_sentiment": classification_report(
            y_true, pred, labels=[0, 1, 2], target_names=list(SENTIMENT_LABELS),
            output_dict=True, zero_division=0,
        ),
    }


def run_training(
    model_type: str,
    *,
    features: str | Path,
    output: str | Path,
    c: float,
    max_iter: int,
    class_weight: str | None = None,
) -> dict:
    """Fit both tasks on Training, score on Validation, save models + metrics."""
    if c <= 0 or max_iter < 1:
        raise ValueError("C and max_iter must be positive")
    class_weight = parse_class_weight(class_weight)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "model_type": model_type,
        "C": c,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "sklearn_version": sklearn.__version__,
        "evaluation_partition": "validation",
        "tasks": {},
    }
    for task in ("aspect", "sentiment"):
        x_train, y_train = load_features(features, task=task, split="train")
        x_val, y_val = load_features(features, task=task, split="val")
        if task == "sentiment" and len(np.unique(y_train)) < 2:
            raise ValueError("Sentiment training requires at least two classes")
        model = build_model(model_type, task, c, max_iter, class_weight)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        results["tasks"][task] = _metrics(task, y_val, pred)
        joblib.dump(model, output / f"{task}_model.joblib")
        print(f"{model_type} {task} validation macro-F1: {results['tasks'][task]['macro_f1']:.4f}")

    (output / "validation_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved models and validation metrics to {output}")
    return results


def train(model_type: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train CITI-Senti {model_type} baseline")
    parser.add_argument("--features", default="models/tfidf", help="Shared TF-IDF feature directory")
    parser.add_argument("--output", default=f"models/{'logistic_regression' if model_type == 'logreg' else 'linear_svm'}/artifacts")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--class-weight", choices=CLASS_WEIGHT_CHOICES, default="none")
    args = parser.parse_args()
    if args.C <= 0 or args.max_iter < 1:
        parser.error("--C and --max-iter must be positive")
    run_training(
        model_type, features=args.features, output=args.output,
        c=args.C, max_iter=args.max_iter, class_weight=args.class_weight,
    )