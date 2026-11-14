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
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from shared.features import load_features
from shared.labels import ASPECT_CODES, SENTIMENT_LABELS


def _estimator(model_type: str, c: float, max_iter: int):
    if model_type == "logreg":
        return LogisticRegression(C=c, max_iter=max_iter, random_state=42)
    if model_type == "linear_svm":
        return LinearSVC(C=c, max_iter=max_iter, random_state=42)
    raise ValueError(f"Unknown model type: {model_type}")


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
        "per_sentiment": classification_report(
            y_true, pred, labels=[0, 1, 2], target_names=list(SENTIMENT_LABELS),
            output_dict=True, zero_division=0,
        ),
    }


def train(model_type: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train CITI-Senti {model_type} baseline")
    parser.add_argument("--features", default="models/tfidf", help="Shared TF-IDF feature directory")
    parser.add_argument("--output", default=f"models/{'logistic_regression' if model_type == 'logreg' else 'linear_svm'}/artifacts")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--max-iter", type=int, default=5000)
    args = parser.parse_args()
    if args.C <= 0 or args.max_iter < 1:
        parser.error("--C and --max-iter must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "model_type": model_type,
        "C": args.C,
        "max_iter": args.max_iter,
        "sklearn_version": sklearn.__version__,
        "evaluation_partition": "validation",
        "tasks": {},
    }
    for task in ("aspect", "sentiment"):
        x_train, y_train = load_features(args.features, task=task, split="train")
        x_val, y_val = load_features(args.features, task=task, split="val")
        base = _estimator(model_type, args.C, args.max_iter)
        model = OneVsRestClassifier(base) if task == "aspect" else base
        if task == "sentiment" and len(np.unique(y_train)) < 2:
            raise ValueError("Sentiment training requires at least two classes")
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        results["tasks"][task] = _metrics(task, y_val, pred)
        joblib.dump(model, output / f"{task}_model.joblib")
        print(f"{model_type} {task} validation macro-F1: {results['tasks'][task]['macro_f1']:.4f}")

    (output / "validation_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved models and validation metrics to {output}")
