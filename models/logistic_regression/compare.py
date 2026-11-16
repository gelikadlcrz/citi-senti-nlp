"""Compare saved Logistic Regression runs side by side on VALIDATION.

Re-scores each artifacts folder with evaluate_saved_models (no retraining) and
prints a table, so the tuned model can be judged against baseline_C1 before
the held-out Testing partition is ever opened.

From the project root:
    python -m models.logistic_regression.compare
    python -m models.logistic_regression.compare --runs baseline_C1 tuned --output results/logreg_val_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from models.logistic_regression.evaluate import evaluate_saved_models

ARTIFACT_ROOT = Path("models/logistic_regression/artifacts")
COLUMNS = (
    ("aspect", "macro_f1"), ("aspect", "micro_f1"), ("aspect", "sample_f1"),
    ("aspect", "hamming_loss"), ("aspect", "exact_match_accuracy"),
    ("sentiment", "macro_f1"), ("sentiment", "accuracy"),
)


def compare_runs(runs: list[str], *, root: str | Path = ARTIFACT_ROOT,
                 features: str | Path = "models/tfidf") -> list[dict]:
    """One row per run: Validation metrics only; Testing is not accessible here."""
    rows = []
    for run in runs:
        folder = Path(run) if Path(run).is_dir() else Path(root) / run
        report = evaluate_saved_models(artifacts=folder, features=features, split="val")
        row = {"run": run, "aspect_thresholds": report["aspect_thresholds"]}
        for task, metric in COLUMNS:
            row[f"{task}_{metric}"] = report["tasks"][task][metric]
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    headers = list(rows[0])
    widths = [max(len(h), *(len(_fmt(r[h])) for r in rows)) for h in headers]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(_fmt(r[h]).ljust(w) for h, w in zip(headers, widths)))
    print("\nHigher is better for every column except hamming_loss (lower is better).")


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["baseline_C1", "tuned"],
                        help="Folder names under artifacts/ (or full paths)")
    parser.add_argument("--features", default="models/tfidf")
    parser.add_argument("--output", type=Path, help="Optional CSV path, e.g. results/logreg_val_comparison.csv")
    args = parser.parse_args(argv)
    try:
        rows = compare_runs(args.runs, features=args.features)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")
    _print_table(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved comparison to {args.output}")
    return rows


if __name__ == "__main__":
    main()