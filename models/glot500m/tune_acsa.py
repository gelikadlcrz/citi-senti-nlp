"""Run a small, reproducible full-fine-tuning ACSA search (no LoRA).

Each trial starts from the same pretrained checkpoint; never from an ACD
classifier or a previously trained ACSA trial. Hyperparameter selection uses
Validation; Testing is untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

TRIALS = [
    {"name": "base_sqrt_lr2e5", "learning_rate": 2e-5, "class_weighting": "sqrt", "max_length": 128},
    {"name": "sqrt_lr1e5", "learning_rate": 1e-5, "class_weighting": "sqrt", "max_length": 128},
    {"name": "sqrt_lr3e5", "learning_rate": 3e-5, "class_weighting": "sqrt", "max_length": 128},
    {"name": "unweighted_lr2e5", "learning_rate": 2e-5, "class_weighting": "none", "max_length": 128},
    {"name": "sqrt_length192", "learning_rate": 2e-5, "class_weighting": "sqrt", "max_length": 192},
]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Small ACSA full-fine-tuning parameter search")
    parser.add_argument("--workbook", type=Path,
                        default=ROOT / "data/splits/citizen_feedback_split.xlsx")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "models/glot500m/artifacts/acsa/search_v1")
    parser.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--optimizer", choices=["adafactor", "adamw", "adamw8bit"], default="adafactor")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--on-conflict", choices=["error", "drop"], default="error")
    parser.add_argument("--on-missing", choices=["error", "drop"], default="drop")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.trials <= len(TRIALS):
        raise ValueError(f"--trials must be in [1, {len(TRIALS)}]")
    if not args.workbook.exists() and not args.dry_run:
        raise FileNotFoundError(f"Workbook not found: {args.workbook}")
    data_hash = hash_file(args.workbook) if args.workbook.exists() else None
    trainer_hash = hash_file(HERE / "train_acsa.py")
    rows = []
    for index, trial in enumerate(TRIALS[:args.trials], start=1):
        directory = args.output_root / f"trial_{index:02d}_{trial['name']}"
        command = [
            sys.executable, str(HERE / "train_acsa.py"),
            "--workbook", str(args.workbook), "--output", str(directory),
            "--checkpoint", args.checkpoint,
            "--epochs", str(args.epochs), "--optimizer", args.optimizer,
            "--batch-size", str(args.batch_size),
            "--gradient-accumulation", str(args.gradient_accumulation),
            "--validation-batch-size", str(args.validation_batch_size),
            "--learning-rate", str(trial["learning_rate"]),
            "--class-weighting", trial["class_weighting"],
            "--max-length", str(trial["max_length"]),
            "--seed", str(args.seed),
            "--on-conflict", args.on_conflict,
            "--on-missing", args.on_missing,
        ]
        if args.cache_dir:
            command.extend(["--cache-dir", str(args.cache_dir)])
        print(f"\nTrial {index}/{args.trials}: {trial['name']}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        result_file = directory / "acsa_results.json"
        if result_file.is_file():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            params = result.get("training_parameters", {})
            valid = (
                result.get("checkpoint") == args.checkpoint
                and result.get("data_fingerprint") == data_hash
                and result.get("trainer_fingerprint") == trainer_hash
                and params.get("learning_rate") == trial["learning_rate"]
                and params.get("class_weighting") == trial["class_weighting"]
                and params.get("max_length") == trial["max_length"]
                and params.get("optimizer") == args.optimizer
                and params.get("epochs_max") == args.epochs
                and params.get("batch_size") == args.batch_size
                and params.get("gradient_accumulation") == args.gradient_accumulation
                and params.get("seed") == args.seed
                and (directory / "config.json").is_file()
            )
            if valid:
                print("  Using previous completed matching trial.", flush=True)
            else:
                raise ValueError(f"Existing results in {directory} do not match this search. "
                                 "Use another --output-root to avoid corrupt comparisons.")
        else:
            subprocess.run(command, check=True)
            result = json.loads(result_file.read_text(encoding="utf-8"))
        rows.append({
            "trial": index, "name": trial["name"], "validation_macro_f1": result["validation"]["macro_f1"],
            "validation_accuracy": result["validation"]["accuracy"],
            "best_epoch": result["best_epoch"], "checkpoint_directory": str(directory),
            **trial,
        })
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / "search_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        best = max(rows, key=lambda r: (r["validation_macro_f1"], r["validation_accuracy"]))
        (args.output_root / "best_configuration.json").write_text(
            json.dumps({"metric": "validation_macro_f1_gold_categories", **best,
                        "warning": "Selected on Validation, not independent Test."}, indent=2),
            encoding="utf-8")
        print(f"  Current best: {best['name']} Macro-F1={best['validation_macro_f1']:.4f}")
    if args.dry_run:
        print("Dry-run complete. No training or results were saved.")
    else:
        print(f"\nSaved search results to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
