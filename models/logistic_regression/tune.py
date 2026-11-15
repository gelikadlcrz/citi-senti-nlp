"""Select Logistic Regression settings on VALIDATION, then tune aspect thresholds.

Grid-searches C x class_weight from config.yaml for each task separately
(aspect detection and sentiment are independent classifiers), keeps the best
by validation macro-F1, and saves the selected models, per-aspect thresholds,
the full grid, and the selected settings. Testing is never touched.

Run from the repository root:
    python -m models.logistic_regression.tune
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import sklearn

from models.logistic_regression.config import load_config
from models.logistic_regression.thresholds import (
    DEFAULT_THRESHOLD, apply_thresholds, aspect_probabilities, save_thresholds, tune_thresholds,
)
from shared.features import load_features
from shared.train_common import _metrics, build_model, parse_class_weight

TASKS = ("aspect", "sentiment")


def _sort_key(row: dict) -> tuple:
    # Highest macro-F1 first; on ties prefer smaller C (stronger regularisation)
    # and no reweighting (the simpler model).
    return (-row["macro_f1"], row["C"], row["class_weight"] != "none")


def tune(
    *,
    features: str | Path,
    output: str | Path,
    c_grid: list[float],
    class_weight_grid: list[str],
    max_iter: int = 5000,
    tune_aspect_thresholds: bool = True,
) -> dict:
    if not c_grid or not class_weight_grid:
        raise ValueError("C_grid and class_weight_grid must not be empty")
    if any(c <= 0 for c in c_grid):
        raise ValueError("Every C in C_grid must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    grid_rows: list[dict] = []
    selected: dict[str, dict] = {}
    report = {
        "model_type": "logreg",
        "sklearn_version": sklearn.__version__,
        "evaluation_partition": "validation",
        "selection_metric": "macro_f1",
        "tasks": {},
    }

    for task in TASKS:
        x_train, y_train = load_features(features, task=task, split="train")
        x_val, y_val = load_features(features, task=task, split="val")
        task_rows = []
        for c in c_grid:
            for cw_name in class_weight_grid:
                cw = parse_class_weight(cw_name)
                model = build_model("logreg", task, float(c), max_iter, cw)
                model.fit(x_train, y_train)
                score = _metrics(task, y_val, model.predict(x_val))["macro_f1"]
                row = {"task": task, "C": float(c), "class_weight": cw or "none", "macro_f1": score}
                task_rows.append(row)
                print(f"{task:9s} C={c:<7g} class_weight={cw or 'none':8s} val macro-F1={score:.4f}")
        grid_rows.extend(task_rows)

        best = sorted(task_rows, key=_sort_key)[0]
        selected[task] = {"C": best["C"], "class_weight": best["class_weight"]}
        model = build_model("logreg", task, best["C"], max_iter, parse_class_weight(best["class_weight"]))
        model.fit(x_train, y_train)
        joblib.dump(model, output / f"{task}_model.joblib")

        if task == "aspect" and tune_aspect_thresholds:
            proba = aspect_probabilities(model, x_val)
            thresholds = tune_thresholds(y_val, proba)
            save_thresholds(output, thresholds)
            report["tasks"][task] = {
                "default_threshold": _metrics(task, y_val, apply_thresholds(proba, None)),
                "tuned_thresholds": _metrics(task, y_val, apply_thresholds(proba, thresholds)),
                "thresholds": thresholds,
                "note": "tuned_thresholds scores are optimistic: thresholds were chosen on this same Validation set.",
            }
        else:
            report["tasks"][task] = _metrics(task, y_val, model.predict(x_val))

    report["selected"] = selected
    (output / "validation_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "selected_config.json").write_text(
        json.dumps({"max_iter": max_iter, "default_threshold": DEFAULT_THRESHOLD,
                    "aspect_thresholds_tuned": tune_aspect_thresholds, **selected}, indent=2) + "\n",
        encoding="utf-8")
    with (output / "tuning_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "C", "class_weight", "macro_f1"])
        writer.writeheader()
        writer.writerows(grid_rows)

    print(f"\nSelected: {json.dumps(selected)}")
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
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Tuning failed: {exc}\n")


if __name__ == "__main__":
    main()