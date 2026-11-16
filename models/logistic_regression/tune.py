"""Select Logistic Regression settings by cross-validation on TRAINING.

For each C x class_weight combination in config.yaml:
  * aspect: 5-fold out-of-fold probabilities on Training -> per-aspect cutoffs
    -> cross-validated macro-F1 with those cutoffs (the same decision rule the
    final model uses);
  * sentiment: stratified 5-fold cross-validated macro-F1 on Training.
Folds are grouped by comment (text_group_id), so rows from one comment never
sit on both sides of a split.
The best combination per task is refit on all of Training. Validation is used
only to REPORT the selected models (and, for transparency, each combination),
so Validation scores are not inflated by selection or threshold tuning.
Testing is never touched.

Run from the repository root:
    python -m models.logistic_regression.tune
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score

from models.logistic_regression.config import load_config
from models.logistic_regression.thresholds import (
    DEFAULT_MAX_POSITIVE_RATE, DEFAULT_MIN_POSITIVES, DEFAULT_THRESHOLD,
    apply_thresholds, aspect_probabilities, out_of_fold_probabilities,
    save_thresholds, tune_thresholds,
)
from shared.features import load_features
from shared.train_common import _metrics, build_model, parse_class_weight

FIELDS = ["task", "C", "class_weight", "cv_macro_f1", "cv_macro_f1_std",
          "cv_macro_ap", "val_macro_f1", "selected"]


def _sort_key(row: dict) -> tuple:
    # Highest cross-validated macro-F1 first; on ties prefer smaller C
    # (stronger regularisation) and no reweighting (the simpler model).
    return (-row["cv_macro_f1"], row["C"], row["class_weight"] != "none")


def _load_groups(features: str | Path, task: str, n_rows: int) -> np.ndarray | None:
    """text_group_id per Training row, so CV keeps each comment's rows together."""
    path = Path(features) / f"{task}_train_rows.csv"
    if not path.is_file():
        print(f"WARNING: {path} not found; cross-validation will not be grouped by comment.")
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        groups = [row["text_group_id"] for row in csv.DictReader(handle)]
    if len(groups) != n_rows:
        raise ValueError(f"{path} has {len(groups)} rows but the {task} features have {n_rows}")
    return np.asarray(groups)


def _macro_ap(y: np.ndarray, proba: np.ndarray) -> float:
    cols = [j for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean([average_precision_score(y[:, j], proba[:, j]) for j in cols]))


def _tune_aspect(x_tr, y_tr, x_val, y_val, grid, max_iter, cv_folds, threshold_kwargs, use_thresholds, groups):
    rows, cache = [], {}
    for c, cw in grid:
        make = lambda c=c, cw=cw: build_model("logreg", "aspect", c, max_iter, cw)
        oof = out_of_fold_probabilities(make, x_tr, y_tr, n_splits=cv_folds, groups=groups)
        thresholds = tune_thresholds(y_tr, oof, **threshold_kwargs) if use_thresholds else None
        cv_f1 = f1_score(y_tr, apply_thresholds(oof, thresholds), average="macro", zero_division=0)
        model = make().fit(x_tr, y_tr)
        val_pred = apply_thresholds(aspect_probabilities(model, x_val), thresholds)
        row = {"task": "aspect", "C": c, "class_weight": cw or "none",
               "cv_macro_f1": float(cv_f1), "cv_macro_f1_std": "",
               "cv_macro_ap": _macro_ap(y_tr, oof),
               "val_macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
               "selected": False}
        rows.append(row)
        cache[(c, cw or "none")] = (model, thresholds)
        print(f"aspect    C={c:<7g} class_weight={cw or 'none':8s} "
              f"CV macro-F1={cv_f1:.4f}  (val {row['val_macro_f1']:.4f})")
    return rows, cache


def _tune_sentiment(x_tr, y_tr, x_val, y_val, grid, max_iter, cv_folds, groups):
    rows, cache = [], {}
    # One comment can yield several sentiment rows (one per aspect); grouping
    # keeps them in the same fold so CV never scores on text it trained on.
    if groups is not None:
        folds = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for c, cw in grid:
        model = build_model("logreg", "sentiment", c, max_iter, cw)
        scores = cross_val_score(model, x_tr, y_tr, cv=folds, scoring="f1_macro", groups=groups)
        model.fit(x_tr, y_tr)
        val_f1 = f1_score(y_val, model.predict(x_val), labels=[0, 1, 2], average="macro", zero_division=0)
        row = {"task": "sentiment", "C": c, "class_weight": cw or "none",
               "cv_macro_f1": float(scores.mean()), "cv_macro_f1_std": float(scores.std()),
               "cv_macro_ap": "", "val_macro_f1": float(val_f1), "selected": False}
        rows.append(row)
        cache[(c, cw or "none")] = (model, None)
        print(f"sentiment C={c:<7g} class_weight={cw or 'none':8s} "
              f"CV macro-F1={scores.mean():.4f} ± {scores.std():.4f}  (val {val_f1:.4f})")
    return rows, cache


def tune(
    *,
    features: str | Path,
    output: str | Path,
    c_grid: list[float],
    class_weight_grid: list[str],
    max_iter: int = 5000,
    tune_aspect_thresholds: bool = True,
    cv_folds: int = 5,
    min_positives: int = DEFAULT_MIN_POSITIVES,
    max_positive_rate: float = DEFAULT_MAX_POSITIVE_RATE,
) -> dict:
    if not c_grid or not class_weight_grid:
        raise ValueError("C_grid and class_weight_grid must not be empty")
    if any(c <= 0 for c in c_grid):
        raise ValueError("Every C in C_grid must be positive")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = [(float(c), parse_class_weight(cw)) for c in c_grid for cw in class_weight_grid]
    threshold_kwargs = {"min_positives": min_positives, "max_positive_rate": max_positive_rate}

    all_rows: list[dict] = []
    selected: dict[str, dict] = {}
    report = {
        "model_type": "logreg",
        "sklearn_version": sklearn.__version__,
        "selection": f"{cv_folds}-fold cross-validated macro-F1 on Training",
        "evaluation_partition": "validation (not used for selection or thresholds)",
        "tasks": {},
    }

    for task in ("aspect", "sentiment"):
        x_tr, y_tr = load_features(features, task=task, split="train")
        x_val, y_val = load_features(features, task=task, split="val")
        groups = _load_groups(features, task, x_tr.shape[0])
        report.setdefault("cv_grouped_by_comment", {})[task] = groups is not None
        if task == "aspect":
            rows, cache = _tune_aspect(x_tr, y_tr, x_val, y_val, grid, max_iter, cv_folds,
                                       threshold_kwargs, tune_aspect_thresholds, groups)
        else:
            rows, cache = _tune_sentiment(x_tr, y_tr, x_val, y_val, grid, max_iter, cv_folds, groups)
        best = sorted(rows, key=_sort_key)[0]
        best["selected"] = True
        all_rows.extend(rows)
        selected[task] = {"C": best["C"], "class_weight": best["class_weight"],
                          "cv_macro_f1": best["cv_macro_f1"]}
        model, thresholds = cache[(best["C"], best["class_weight"])]
        joblib.dump(model, output / f"{task}_model.joblib")

        if task == "aspect" and thresholds is not None:
            save_thresholds(output, thresholds)
            proba = aspect_probabilities(model, x_val)
            report["tasks"][task] = {
                "default_threshold": _metrics(task, y_val, apply_thresholds(proba, None)),
                "tuned_thresholds": _metrics(task, y_val, apply_thresholds(proba, thresholds)),
                "thresholds": thresholds,
                "note": "Thresholds were tuned on out-of-fold Training probabilities; Validation was not used.",
            }
        else:
            report["tasks"][task] = _metrics(task, y_val, model.predict(x_val))

    report["selected"] = selected
    (output / "validation_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "selected_config.json").write_text(
        json.dumps({"max_iter": max_iter, "cv_folds": cv_folds,
                    "default_threshold": DEFAULT_THRESHOLD,
                    "aspect_thresholds_tuned": tune_aspect_thresholds,
                    **threshold_kwargs, **selected}, indent=2) + "\n", encoding="utf-8")
    with (output / "tuning_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSelected (by CV on Training): {json.dumps(selected)}")
    if tune_aspect_thresholds:
        aspect = report["tasks"]["aspect"]
        print(f"Validation aspect macro-F1: {aspect['tuned_thresholds']['macro_f1']:.4f} "
              f"(0.5 cutoff: {aspect['default_threshold']['macro_f1']:.4f})")
    print(f"Validation sentiment macro-F1: {report['tasks']['sentiment']['macro_f1']:.4f}")
    print(f"Saved tuned models, thresholds and grid results to {output}")
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config (default: config.yaml beside this script)")
    parser.add_argument("--features")
    parser.add_argument("--output")
    parser.add_argument("--no-thresholds", action="store_true", help="Keep the fixed 0.5 aspect cutoff")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tuning = cfg.get("tuning", {})
    try:
        return tune(
            features=args.features or cfg["features"],
            output=args.output or tuning.get("output", "models/logistic_regression/artifacts/tuned"),
            c_grid=[float(c) for c in tuning.get("C_grid", [1.0])],
            class_weight_grid=[str(cw) for cw in tuning.get("class_weight_grid", ["none"])],
            max_iter=int(cfg.get("training", {}).get("max_iter", 5000)),
            tune_aspect_thresholds=bool(tuning.get("tune_aspect_thresholds", True)) and not args.no_thresholds,
            cv_folds=int(tuning.get("cv_folds", 5)),
            min_positives=int(tuning.get("min_positives", DEFAULT_MIN_POSITIVES)),
            max_positive_rate=float(tuning.get("max_positive_rate", DEFAULT_MAX_POSITIVE_RATE)),
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Tuning failed: {exc}\n")


if __name__ == "__main__":
    main()