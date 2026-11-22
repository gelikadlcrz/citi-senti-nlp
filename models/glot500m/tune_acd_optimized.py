"""Safe, reproducible GLOT500-m ACD hyperparameter search (development only).

Uses the accompanying optimized train_acd.py. Unlike the old tuning script,
this refuses to reuse completed runs if the workbook, trainer, or settings differ.
It DOES NOT implement nested grouped cross-validation and NEVER reads Testing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "models/glot500m/train_acd.py"
DEFAULT_WORKBOOK = PROJECT_ROOT / "data/splits/citizen_feedback_split.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "models/glot500m/artifacts/acd/tuning_optimized"

SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "effective_batch_size": [8, 16],
    "epochs": [3, 5, 8],
    "max_length": [128, 256],
    "weight_decay": [0.0, 0.01],
    "warmup_ratio": [0.0, 0.10],
}
FIELDS = list(SEARCH_SPACE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates_for(seed: int, lengths: set[int]):
    options = [dict(zip(FIELDS, values)) for values in
               itertools.product(*(SEARCH_SPACE[field] for field in FIELDS))]
    options = [candidate for candidate in options if candidate["max_length"] in lengths]
    random.Random(seed).shuffle(options)
    return options


def build_command(args, output: Path, config: dict) -> list[str]:
    if config["effective_batch_size"] % args.batch_size:
        raise ValueError("--batch-size must divide selected effective batch sizes")
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--workbook", str(args.workbook),
        "--output", str(output),
        "--checkpoint", str(args.checkpoint),
        "--batch-size", str(args.batch_size),
        "--validation-batch-size", str(args.validation_batch_size),
        "--gradient-accumulation", str(config["effective_batch_size"] // args.batch_size),
        "--epochs", str(config["epochs"]),
        "--learning-rate", str(config["learning_rate"]),
        "--max-length", str(config["max_length"]),
        "--weight-decay", str(config["weight_decay"]),
        "--warmup-ratio", str(config["warmup_ratio"]),
        "--threshold-mode", str(args.threshold_mode),
        "--seed", str(args.seed),
        "--patience", str(args.patience),
        "--loss-weighting", str(args.loss_weighting),
        "--max-pos-weight", str(args.max_pos_weight),
        "--head-lr-multiplier", str(args.head_lr_multiplier),
        "--max-label-multiplier", str(args.max_label_multiplier),
        "--selection-metric", str(args.selection_metric),
        "--log-every", str(args.log_every),
        "--cache-dir", str(args.cache_dir),
    ]
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def validate_reuse(path: Path, args, config: dict, workbook_sha: str,
                   trainer_sha: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("data_fingerprint") != workbook_sha:
        raise ValueError(f"Old/different dataset in {path}. Use a new --output-root.")
    if result.get("trainer_fingerprint") != trainer_sha:
        raise ValueError(f"Trainer code changed since {path}. Use a new --output-root.")
    if result.get("checkpoint") != args.checkpoint or result.get("threshold_mode") != args.threshold_mode:
        raise ValueError(f"Model/threshold mode mismatch in {path}; use new output folder.")
    parameters = result.get("training_parameters", {})
    expected = {
        "effective_batch_size": config["effective_batch_size"],
        "epochs_max": config["epochs"],
        "learning_rate": config["learning_rate"],
        "max_length": config["max_length"],
        "weight_decay": config["weight_decay"],
        "warmup_ratio": config["warmup_ratio"],
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "loss_weighting": args.loss_weighting,
        "max_pos_weight": args.max_pos_weight,
        "head_lr_multiplier": args.head_lr_multiplier,
        "max_label_multiplier": args.max_label_multiplier,
        "selection_metric": args.selection_metric,
    }
    for name, value in expected.items():
        if parameters.get(name) != value:
            raise ValueError(f"Mismatched run: {name}={parameters.get(name)} "
                             f"rather than {value} in {path}; use new --output-root.")
    return result


def save_summary(rows, output_root: Path, seed: int, workbook_sha: str):
    ranked = sorted(rows, key=lambda row: (-row["macro_f1"], row["trial"]))
    cols = ["rank", "trial", "macro_f1", "macro_ap", "micro_f1", "exact_set_match",
            "best_epoch", *FIELDS, "checkpoint_directory"]
    with (output_root / "search_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        for rank, item in enumerate(ranked, 1):
            writer.writerow({col: (rank if col == "rank" else item[col]) for col in cols})
    if ranked:
        best = ranked[0]
        (output_root / "best_configuration.json").write_text(json.dumps({
            "trial": best["trial"], "seed": seed,
            "selection_metric": "validation Macro-F1 after threshold tuning",
            "score": best["macro_f1"],
            "checkpoint_directory": best["checkpoint_directory"],
            "hyperparameters": {key: best[key] for key in FIELDS},
            "workbook_sha256": workbook_sha,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "warning": "Development fixed-Validation tuning only; not nested grouped CV or Test scoring.",
        }, indent=2), encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description="Optimized GLOT500-m ACD tuning")
    p.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--validation-batch-size", type=int, default=4)
    p.add_argument("--loss-weighting", choices=("none", "sqrt", "full"), default="sqrt")
    p.add_argument("--max-pos-weight", type=float, default=5.0)
    p.add_argument("--head-lr-multiplier", type=float, default=5.0)
    p.add_argument("--max-label-multiplier", type=float, default=0.0)
    p.add_argument("--selection-metric", choices=("macro_f1", "macro_ap"), default="macro_f1")
    p.add_argument("--threshold-mode", choices=("global", "eligible-category"), default="global")
    p.add_argument("--lengths", nargs="+", type=int, choices=(128, 256), default=[128, 256],
                   help="Constrain token lengths, e.g. --lengths 128 for faster MPS tuning")
    p.add_argument("--cache-dir", type=Path,
                   default=PROJECT_ROOT / "models/glot500m/artifacts/token_cache")
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prune-rejected", action="store_true")
    return p


def main():
    args = parser().parse_args()
    choices = candidates_for(args.seed, set(args.lengths))
    if not 1 <= args.trials <= len(choices):
        raise ValueError(f"--trials must be between 1 and {len(choices)} with these lengths")
    selected = choices[:args.trials]
    print(f"Sampling {len(selected)} of {len(choices)} unique configurations (seed {args.seed})")
    for i, config in enumerate(selected, 1):
        print(f"{i:02d}: " + ", ".join(f"{k}={v}" for k, v in config.items()))
    if args.dry_run:
        print("Dry run only. No model training or output files written.")
        return
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing train_acd.py: {TRAIN_SCRIPT}")
    if not args.workbook.is_file():
        raise FileNotFoundError(f"Missing workbook: {args.workbook}")
    data_sha = sha256_file(args.workbook)
    trainer_sha = sha256_file(TRAIN_SCRIPT)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, config in enumerate(selected, 1):
        trial_dir = args.output_root / f"trial_{index:02d}"
        result_path = trial_dir / "acd_results.json"
        if result_path.exists():
            print(f"\n[trial {index}/{args.trials}] Verifying reusable result in {trial_dir}", flush=True)
            result = validate_reuse(result_path, args, config, data_sha, trainer_sha)
        else:
            print(f"\n[trial {index}/{args.trials}] Training {config}", flush=True)
            subprocess.run(build_command(args, trial_dir, config), cwd=PROJECT_ROOT, check=True)
            if not result_path.exists():
                raise RuntimeError(f"Trainer exited without writing results: {result_path}")
            result = validate_reuse(result_path, args, config, data_sha, trainer_sha)
        metric = result["validation_after_threshold_tuning"]
        row = dict(
            trial=index, macro_f1=float(metric["macro_f1"]),
            macro_ap=float(result["validation_macro_average_precision"]),
            micro_f1=float(metric["micro_f1"]),
            exact_set_match=float(metric["exact_set_match"]),
            best_epoch=int(result["best_epoch"]),
            checkpoint_directory=str(trial_dir.resolve()), **config,
        )
        rows.append(row)
        save_summary(rows, args.output_root, args.seed, data_sha)
        print(f"Trial {index}: validation Macro-F1={row['macro_f1']:.4f}; "
              f"Macro-AP={row['macro_ap']:.4f}", flush=True)
        if args.prune_rejected:
            best_trial = max(rows, key=lambda item: (item["macro_f1"], -item["trial"]))["trial"]
            for existing in rows:
                if existing["trial"] == best_trial:
                    continue
                for weight in Path(existing["checkpoint_directory"]).glob("model*.safetensors"):
                    weight.unlink()
                    print(f"Pruned non-winning weights: {weight}", flush=True)
    winner = sorted(rows, key=lambda row: (-row["macro_f1"], row["trial"]))[0]
    print("\nSearch complete.")
    print(f"Best trial: {winner['trial']}; Validation Macro-F1: {winner['macro_f1']:.4f}")
    print("Best checkpoint:", winner["checkpoint_directory"])
    print("Results:", args.output_root / "search_results.csv")
    print("Testing sheet was not read. This remains fixed-split tuning, not nested CV.")


if __name__ == "__main__":
    main()
