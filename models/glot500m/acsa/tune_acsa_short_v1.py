"""Short warm-start ACSA tuning using the ORIGINAL V1 train_acsa.py unchanged.

Train/Validation only. Each trial independently resumes *model weights* from the
same previously trained V1 checkpoint, not optimizer state. No LoRA or Experiment 2
input changes. Testing is never accessed here.
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
TRIALS = (
    {"name": "lr5e6_sqrt", "learning_rate": 5e-6, "class_weighting": "sqrt"},
    {"name": "lr1e5_sqrt", "learning_rate": 1e-5, "class_weighting": "sqrt"},
    {"name": "lr5e6_unweighted", "learning_rate": 5e-6, "class_weighting": "none"},
)
WEIGHTS = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_checkpoint(checkpoint: Path) -> dict:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"V1 checkpoint folder missing: {checkpoint}")
    if not (checkpoint / "config.json").is_file() or not any((checkpoint / x).is_file() for x in WEIGHTS):
        raise FileNotFoundError(f"V1 checkpoint is incomplete (config.json or model weights): {checkpoint}")
    report_path = checkpoint / "acsa_results.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing V1 training results: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("training_mode") != "full_fine_tuning_no_lora":
        raise ValueError("Source checkpoint must be a fully fine-tuned ACSA model")
    if report.get("input_format") != "tokenizer(text=model_text, text_pair='Aspect: CODE — NAME')":
        raise ValueError("Source checkpoint does not use the original V1 paired-text input")
    if report.get("label_order") != ["Negative", "Neutral", "Positive"]:
        raise ValueError("Wrong V1 sentiment label order")
    if not (checkpoint / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Tokenizer files missing in {checkpoint}")
    return report


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1-only short ACSA fine-tuning trials")
    p.add_argument("--baseline", type=Path, default=ROOT / "models/glot500m/artifacts/acsa/full_ft_v1")
    p.add_argument("--workbook", type=Path, default=ROOT / "data/splits/citizen_feedback_split.xlsx")
    p.add_argument("--output-root", type=Path, default=ROOT / "models/glot500m/artifacts/acsa/v1_short_search")
    p.add_argument("--epochs", type=int, default=2, help="Additional epochs from the saved V1 checkpoint")
    p.add_argument("--trials", type=int, default=2, help="1 or 2 LR trials; third trial tests unweighted CE")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--validation-batch-size", type=int, default=2)
    p.add_argument("--optimizer", choices=("adafactor", "adamw", "adamw8bit"), default="adafactor")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "models/glot500m/artifacts/acsa/token_cache")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--on-conflict", choices=("drop", "error"), default="drop")
    p.add_argument("--on-missing", choices=("drop", "error"), default="drop")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = cli()
    if not 1 <= args.trials <= len(TRIALS):
        raise ValueError(f"--trials must be 1 through {len(TRIALS)}")
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.validation_batch_size) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")
    baseline = args.baseline.resolve()
    out = args.output_root.resolve()
    wb = args.workbook.resolve()
    trainer = HERE / "train_acsa.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"Place this tuner beside the ORIGINAL V1 train_acsa.py: {trainer}")
    source = trainer.read_text(encoding="utf-8")
    if '"paired-text-v1"' not in source or 'mixed-sentiment-boost' in source:
        raise ValueError("This tuner requires original V1 train_acsa.py (not Experiment 2). Restore V1 trainer.")
    if out == baseline or out in baseline.parents or baseline in out.parents:
        raise ValueError("Output-root and V1 baseline directories must not overlap")
    if not wb.is_file() or not baseline.is_dir():
        raise FileNotFoundError("Missing workbook or V1 baseline checkpoint")
    source_report = check_checkpoint(baseline)
    data_hash = sha256_file(wb)
    if source_report.get("data_fingerprint") != data_hash:
        raise ValueError("Dataset workbook differs from V1 baseline; use the unchanged version to compare fairly")
    trainer_hash = sha256_file(trainer)
    checkpoint_arg = str(baseline)
    baseline_f1 = float(source_report["validation"]["macro_f1"])
    print(f"Baseline V1 Validation Macro-F1: {baseline_f1:.4f}", flush=True)
    print("IMPORTANT: This continues model weights from V1; each trial resets the optimizer/scheduler.", flush=True)
    rows = []
    for idx, trial in enumerate(TRIALS[:args.trials], 1):
        dest = out / f"trial_{idx:02d}_{trial['name']}"
        command = [
            sys.executable, str(trainer),
            "--workbook", str(wb), "--output", str(dest),
            "--checkpoint", checkpoint_arg, "--epochs", str(args.epochs),
            "--optimizer", args.optimizer,
            "--learning-rate", str(trial["learning_rate"]),
            "--head-lr-multiplier", "1",  # V1's classifier is already trained
            "--max-length", str(source_report["training_parameters"]["max_length"]),
            "--class-weighting", trial["class_weighting"],
            "--class-weight-cap", str(source_report["training_parameters"].get("class_weight_cap", 3.0)),
            "--batch-size", str(args.batch_size),
            "--gradient-accumulation", str(args.gradient_accumulation),
            "--validation-batch-size", str(args.validation_batch_size),
            "--seed", str(args.seed),
            "--on-conflict", args.on_conflict, "--on-missing", args.on_missing,
            "--cache-dir", str(args.cache_dir.resolve()),
            "--log-every", str(args.log_every),
            "--gradient-checkpointing",
        ]
        print(f"\nTrial {idx}/{args.trials}: {trial['name']} | {args.epochs} additional epochs", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        result_path = dest / "acsa_results.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            params = result.get("training_parameters", {})
            expected = {
                "learning_rate": trial["learning_rate"],
                "optimizer": args.optimizer,
                "epochs_max": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "max_length": source_report["training_parameters"]["max_length"],
                "class_weighting": trial["class_weighting"],
                "head_lr_multiplier": 1.0,
                "seed": args.seed,
                "gradient_checkpointing": True,
            }
            if not (result.get("checkpoint") == checkpoint_arg
                    and result.get("data_fingerprint") == data_hash
                    and result.get("trainer_fingerprint") == trainer_hash
                    and all(params.get(k) == v for k, v in expected.items())
                    and (dest / "config.json").is_file()
                    and any((dest / name).is_file() for name in WEIGHTS)):
                raise ValueError(f"Existing trial does not match requested run: {dest}. Use new --output-root.")
            print("  Reusing completed matching trial", flush=True)
        else:
            # Prevent accidental use of a partly written checkpoint.
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError(f"Incomplete trial folder: {dest}. Move it aside before retrying.")
            subprocess.run(command, check=True)
            if not result_path.is_file():
                raise RuntimeError(f"Trainer finished but did not write {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append({
            "trial": idx, "name": trial["name"], "validation_macro_f1": float(result["validation"]["macro_f1"]),
            "validation_accuracy": float(result["validation"]["accuracy"]),
            "best_epoch": result["best_epoch"], "checkpoint_directory": str(dest),
            "starting_checkpoint": checkpoint_arg, "additional_epochs_max": args.epochs,
            "learning_rate": trial["learning_rate"], "class_weighting": trial["class_weighting"],
        })
        out.mkdir(parents=True, exist_ok=True)
        with (out / "search_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        best = max(rows, key=lambda x: (x["validation_macro_f1"], x["validation_accuracy"]))
        winner = (best if (best["validation_macro_f1"], best["validation_accuracy"]) >
                  (baseline_f1, float(source_report["validation"]["accuracy"])) else
                  {"name": "baseline_v1", "validation_macro_f1": baseline_f1,
                   "validation_accuracy": float(source_report["validation"]["accuracy"]),
                   "checkpoint_directory": checkpoint_arg})
        (out / "best_configuration.json").write_text(json.dumps({
            "selected_by": "Validation 3-class Macro-F1; accuracy tie-break",
            "baseline_validation_macro_f1": baseline_f1,
            "winner": winner,
            "trials_completed": len(rows),
            "dataset_sha256": data_hash,
            "warning": "Validation selection, NOT independent Test; initial V1 was already trained on this Validation setup.",
        }, indent=2), encoding="utf-8")
        print(f"  Current best incl. baseline: {winner['name']} Macro-F1={winner['validation_macro_f1']:.4f}", flush=True)
    if args.dry_run:
        print("Dry-run complete: no model was trained.")
    else:
        print(f"\nSearch finished. Files: {out / 'search_results.csv'} and {out / 'best_configuration.json'}")


if __name__ == "__main__":
    main()
