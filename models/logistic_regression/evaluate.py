"""Evaluate SAVED Logistic Regression classifiers on the shared TF-IDF matrices.

From the project root:
    python -m models.logistic_regression.evaluate \
        --artifacts models/logistic_regression/artifacts/baseline_C1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from shared.features import load_features
from shared.train_common import _metrics

TASKS = ("aspect", "sentiment")


def evaluate_saved_models(
    *,
    artifacts: str | Path,
    features: str | Path = "models/tfidf",
    split: str = "val",
    allow_test: bool = False,
) -> dict:
    """Return comparable metrics without retraining or refitting TF-IDF."""
    if split not in ("val", "test"):
        raise ValueError("Only 'val' and 'test' are supported for evaluation")
    if split == "test" and not allow_test:
        raise ValueError(
            "Testing is held out. Freeze the configuration and threshold choices "
            "first, then explicitly pass allow_test=True / --allow-test."
        )

    artifact_dir = Path(artifacts)
    feature_dir = Path(features)
    report = {"model_type": "logreg", "split": split, "tasks": {}}
    for task in TASKS:
        model_file = artifact_dir / f"{task}_model.joblib"
        if not model_file.is_file():
            raise FileNotFoundError(f"Missing saved model: {model_file}")
        # joblib loads Python objects: only open your team's trusted artifacts.
        model = joblib.load(model_file)
        x, y = load_features(feature_dir, task=task, split=split)
        if model.n_features_in_ != x.shape[1]:
            raise ValueError(f"{task} model and {split} features have different widths")
        pred = model.predict(x)
        report["tasks"][task] = _metrics(task, y, pred)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="models/logistic_regression/artifacts")
    parser.add_argument("--features", default="models/tfidf")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true", help="Explicitly unlock held-out testing only after model selection is frozen")
    parser.add_argument("--output", type=Path, help="Save separate evaluation JSON (does not overwrite training metrics)")
    args = parser.parse_args()
    try:
        report = evaluate_saved_models(
            artifacts=args.artifacts,
            features=args.features,
            split=args.split,
            allow_test=args.allow_test,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
