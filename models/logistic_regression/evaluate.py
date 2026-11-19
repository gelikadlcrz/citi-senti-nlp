"""Evaluate SAVED Logistic Regression classifiers on the shared TF-IDF matrices.

From the project root:
    python -m models.logistic_regression.evaluate \
        --artifacts models/logistic_regression/artifacts/tuned
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from models.logistic_regression.pipeline import (
    BUNDLE_FILE, _read_resolutions, category_records, audit_conflicts,
    aspect_mask, category_interactions, predict_text,
)
from models.logistic_regression.metrics import metrics as pair_metrics, group_bootstrap_ci
from shared.data import WORKBOOK_DEFAULT, read_annotation_sheet
from shared.labels import ASPECT_CODES, ASPECT_TO_INDEX, SENTIMENT_LABELS

from models.logistic_regression.thresholds import (
    apply_thresholds, aspect_probabilities, load_thresholds,
)
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
    thresholds = load_thresholds(artifact_dir)
    report = {
        "model_type": "logreg",
        "split": split,
        "aspect_thresholds": "tuned" if thresholds else "default_0.5",
        "tasks": {},
    }
    for task in TASKS:
        model_file = artifact_dir / f"{task}_model.joblib"
        if not model_file.is_file():
            raise FileNotFoundError(f"Missing saved model: {model_file}")
        # joblib loads Python objects: only open your team's trusted artifacts.
        model = joblib.load(model_file)
        x, y = load_features(feature_dir, task=task, split=split)
        if model.n_features_in_ != x.shape[1]:
            raise ValueError(f"{task} model and {split} features have different widths")
        if task == "aspect" and thresholds:
            pred = apply_thresholds(aspect_probabilities(model, x), thresholds)
        else:
            pred = model.predict(x)
        report["tasks"][task] = _metrics(task, y, pred)
    return report


def evaluate_frozen(*, artifacts: str | Path, workbook=WORKBOOK_DEFAULT,
                    manifest="data/splits/group_manifest.csv", resolutions=None,
                    allow_test=False):
    if not allow_test:
        raise ValueError("Held-out Testing is locked; pass --allow-test ONLY after freezing all models")
    artifact_dir = Path(artifacts)
    if not (artifact_dir / "development_report.json").is_file():
        raise FileNotFoundError("Missing frozen development selection report")
    with (artifact_dir / "development_report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("test_used_by_this_procedure") is not False:
        raise ValueError("Development report does not certify Testing was excluded from fitting")
    bundle = joblib.load(artifact_dir / BUNDLE_FILE)
    # Testing is first opened after the explicit allow_test guard.
    test = read_annotation_sheet(workbook, "Testing")
    comments = test.drop_duplicates("comment_id").reset_index(drop=True)
    # Verify group assignments against the trusted source manifest, without
    # regenerating a split. The data team must have verified group semantics.
    manifest_path = Path(manifest)
    m = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if not {"comment_id", "group_id", "split"}.issubset(m.columns):
        raise ValueError("Group manifest lacks required columns")
    ids = comments.comment_id.astype(str).tolist()
    by_id = m.set_index("comment_id").to_dict(orient="index")
    if any(cid not in by_id or by_id[cid]["split"] != "test" for cid in ids):
        raise ValueError("Test comment IDs disagree with the shared split manifest")
    training_groups = set(m.loc[m.split == "development", "group_id"])
    groups = [by_id[cid]["group_id"] for cid in ids]
    if training_groups.intersection(groups):
        raise ValueError("Test groups overlap development groups")
    all_resolutions = _read_resolutions(resolutions)
    test_conflicts = audit_conflicts(test)
    test_resolutions = {key: val for key, val in all_resolutions.items() if key in test_conflicts}
    category_rows = category_records(test, test_resolutions)
    gold_by_id = {}
    for cid, code, _, sentiment in category_rows:
        gold_by_id.setdefault(cid, set()).add((code, SENTIMENT_LABELS[sentiment]))
    gold = [gold_by_id.get(cid, set()) for cid in ids]
    pred = [set(predict_text(bundle, text)) for text in comments.model_text]
    scores = pair_metrics(gold, pred)
    intervals = group_bootstrap_ci(gold, pred, groups)
    # Task-specific independent diagnostic scores.
    true_a = np.zeros((len(ids), len(ASPECT_CODES)), dtype=np.int8)
    for i, pairs in enumerate(gold):
        for code, _ in pairs:
            true_a[i, ASPECT_TO_INDEX[code]] = 1
    aspect = bundle["aspect"]
    scores_a = aspect["classifier"].predict_proba(
        aspect["vectorizer"].transform(comments.model_text.astype(str).tolist()))
    predicted_a = aspect_mask(scores_a, aspect["thresholds"])
    aspect_f1 = float(f1_score(true_a, predicted_a, average="macro", zero_division=0))
    sentiment = bundle["sentiment"]
    texts = [text for _, _, text, _ in category_rows]
    x = sentiment["vectorizer"].transform(texts)
    cat = np.array([ASPECT_TO_INDEX[code] for _, code, _, _ in category_rows])
    y = np.array([value for _, _, _, value in category_rows])
    p = sentiment["classifier"].predict(category_interactions(x, cat))
    conditional = {
        "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        "labels": list(SENTIMENT_LABELS),
    }
    return {"split": "test", "frozen_model": str(artifact_dir),
            "end_to_end": scores, "end_to_end_group_ci": intervals,
            "aspect_macro_f1": aspect_f1, "conditional_category_sentiment": conditional,
            "predictions": [{"comment_id": cid, "group_id": group,
                             "gold_pairs": sorted([list(p) for p in g]),
                             "predicted_pairs": sorted([list(p) for p in s])}
                            for cid, group, g, s in zip(ids, groups, gold, pred)]}



def _frozen_test_cli(argv=None):
    """Evaluate a frozen grouped-CV model; testing stays explicitly locked."""
    parser = argparse.ArgumentParser(description="Frozen LR Testing evaluation")
    parser.add_argument("--artifacts", default="models/logistic_regression/artifacts/grouped_cv")
    parser.add_argument("--workbook", default=str(WORKBOOK_DEFAULT))
    parser.add_argument("--manifest", default="data/splits/group_manifest.csv")
    parser.add_argument("--resolutions")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = evaluate_frozen(artifacts=args.artifacts, workbook=args.workbook,
                                 manifest=args.manifest, resolutions=args.resolutions,
                                 allow_test=args.allow_test)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Held-out LR evaluation stopped: {exc}\n")
    output = Path(args.output) if args.output else Path(args.artifacts) / "frozen_test_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    return report


def main() -> None:
    argv = list(sys.argv[1:])
    if "--frozen-test" in argv:
        return _frozen_test_cli([arg for arg in argv if arg != "--frozen-test"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="models/logistic_regression/artifacts/tuned")
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