"""Run a reproducible Glot500-m ACD hyperparameter search on fixed Training/Validation.

Run from the CITI-Senti repository root (PowerShell):
    python models/glot500m/tune_acd.py --dry-run --trials 3
    python models/glot500m/tune_acd.py --trials 3 --prune-rejected
    python models/glot500m/tune_acd.py --trials 12 --prune-rejected

This orchestrates the existing train_acd.py. No changes to data splits or ACSA.
It is a single-holdout search, NOT the paper's full nested group cross-validation.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "models" / "glot500m" / "train_acd.py"
DEFAULT_WORKBOOK = PROJECT_ROOT / "data" / "splits" / "citizen_feedback_split.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "glot500m" / "artifacts" / "acd" / "search_seed42"

SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "effective_batch_size": [8, 16],
    "epochs": [3, 5, 8],
    "max_length": [128, 256],
    "weight_decay": [0.0, 0.01],
    "warmup_ratio": [0.0, 0.10],
}
FIELDS = list(SEARCH_SPACE)


def create_candidates(seed: int) -> list[dict]:
    candidates = [
        dict(zip(FIELDS, values))
        for values in itertools.product(*(SEARCH_SPACE[field] for field in FIELDS))
    ]
    assert len(candidates) == 144
    random.Random(seed).shuffle(candidates)
    return candidates


def read_result(result_path: Path, expected_config: dict, seed: int, threshold_mode: str) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    actual = result.get("training_parameters", {})
    for name, expected in expected_config.items():
        if name == "effective_batch_size":
            observed = actual.get("effective_batch_size")
        else:
            key = {"learning_rate": "learning_rate", "epochs": "epochs_max",
                   "max_length": "max_length", "weight_decay": "weight_decay",
                   "warmup_ratio": "warmup_ratio"}[name]
            observed = actual.get(key)
        if observed != expected:
            raise ValueError(f"Existing {result_path} uses {name}={observed}, expected {expected}. "
                             "Choose a new output directory or restore matching settings.")
    if actual.get("seed") != seed or result.get("threshold_mode") != threshold_mode:
        raise ValueError(f"Existing {result_path} has different seed or threshold mode")
    return result


def save_summary(rows: list[dict], output_root: Path, seed: int) -> None:
    ranked = sorted(rows, key=lambda r: (-r["macro_f1"], r["trial"]))
    path = output_root / "search_results.csv"
    columns = ["rank", "trial", "macro_f1", "micro_f1", "exact_set_match",
               "best_epoch", *FIELDS, "checkpoint_directory"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for rank, record in enumerate(ranked, start=1):
            writer.writerow({key: (rank if key == "rank" else record[key]) for key in columns})
    if ranked:
        best = ranked[0]
        (output_root / "best_configuration.json").write_text(
            json.dumps({"trial": best["trial"], "seed": seed,
                        "selection_metric": "validation 11-label Macro-F1 after threshold tuning",
                        "score": best["macro_f1"],
                        "checkpoint_directory": best["checkpoint_directory"],
                        "hyperparameters": {key: best[key] for key in FIELDS},
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "warning": "Selection uses the fixed Validation sheet; not independent test performance. "
                                   "This does not implement nested CV."}, indent=2), encoding="utf-8"
        )


def build_command(args: argparse.Namespace, trial_dir: Path, config: dict) -> list[str]:
    # Microbatch of 1 + gradient accumulation achieves an effective batch of 8 or 16.
    command = [sys.executable, str(TRAIN_SCRIPT),
               "--workbook", str(args.workbook),
               "--output", str(trial_dir),
               "--checkpoint", args.checkpoint,
               "--batch-size", "1",
               "--gradient-accumulation", str(config["effective_batch_size"]),
               "--epochs", str(config["epochs"]),
               "--learning-rate", str(config["learning_rate"]),
               "--max-length", str(config["max_length"]),
               "--weight-decay", str(config["weight_decay"]),
               "--warmup-ratio", str(config["warmup_ratio"]),
               "--threshold-mode", args.threshold_mode,
               "--seed", str(args.seed),
               "--patience", str(args.patience)]
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperparameter optimization for CITI-Senti Glot500 ACD")
    p.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--threshold-mode", choices=["global", "eligible-category"], default="global")
    p.add_argument("--dry-run", action="store_true", help="Print selected experiments without training")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prune-rejected", action="store_true",
                   help="Remove large model weights from completed non-winning trials, keeping metrics")
    return p


def main() -> None:
    args = parser().parse_args()
    if not 1 <= args.trials <= 144:
        raise ValueError("--trials must be between 1 and 144")
    candidates = create_candidates(args.seed)[:args.trials]
    print(f"Sampling {len(candidates)} of {len(create_candidates(args.seed))} unique configurations (seed {args.seed})")
    for index, c in enumerate(candidates, 1):
        print(f"{index:02d}: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    if args.dry_run:
        print("Dry run only; no model training and no files written.")
        return
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")
    if not args.workbook.is_file():
        raise FileNotFoundError(f"Missing workbook: {args.workbook}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, config in enumerate(candidates, 1):
        trial_dir = args.output_root / f"trial_{index:02d}"
        result_file = trial_dir / "acd_results.json"
        if result_file.exists():
            print(f"\n[trial {index}/{args.trials}] Reusing completed trial in {trial_dir}", flush=True)
            result = read_result(result_file, config, args.seed, args.threshold_mode)
        else:
            print(f"\n[trial {index}/{args.trials}] Training {config}", flush=True)
            subprocess.run(build_command(args, trial_dir, config), check=True, cwd=PROJECT_ROOT)
            if not result_file.exists():
                raise RuntimeError(f"Training returned successfully without producing {result_file}")
            result = read_result(result_file, config, args.seed, args.threshold_mode)
        metric = result["validation_after_threshold_tuning"]
        record = dict(trial=index, macro_f1=float(metric["macro_f1"]),
                      micro_f1=float(metric["micro_f1"]),
                      exact_set_match=float(metric["exact_set_match"]),
                      best_epoch=int(result["best_epoch"]),
                      checkpoint_directory=str(trial_dir.resolve()), **config)
        rows.append(record)
        save_summary(rows, args.output_root, args.seed)
        print(f"Trial {index}: validation Macro-F1={record['macro_f1']:.4f}", flush=True)
        if args.prune_rejected:
            winning_trial = max(rows, key=lambda r: (r["macro_f1"], -r["trial"]))["trial"]
            for old in rows:
                if old["trial"] == winning_trial:
                    continue
                rejected_weight = Path(old["checkpoint_directory"]) / "model.safetensors"
                if rejected_weight.is_file():
                    rejected_weight.unlink()
                    print(f"Deleted non-winning model weights: {rejected_weight}", flush=True)
    best = sorted(rows, key=lambda r: (-r["macro_f1"], r["trial"]))[0]
    print("\nSearch complete.")
    print(f"Best trial: {best['trial']}; Macro-F1: {best['macro_f1']:.4f}")
    print("Best checkpoint:", best["checkpoint_directory"])
    print("Results:", args.output_root / "search_results.csv")
    print("Test data were not accessed. This is fixed-split tuning, not nested CV.")


if __name__ == "__main__":
    main()
