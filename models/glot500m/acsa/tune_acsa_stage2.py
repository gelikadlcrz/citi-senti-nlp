"""CITI-Senti ACSA: focused stage-2 Validation-only experiments.

Runs the EXISTING train_acsa_research.py as a subprocess without editing it.
Uses the already trained V1 Trial 2 checkpoint as the default starting point.
Every trial restarts from identical weights and settings (except tested changes).
No Testing-sheet access and no changes to existing final_selection.json.
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
ROOT = HERE.parents[1]
TRAINER = HERE / "train_acsa_research.py"
MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")

# Identical 5e-6 LR, class weights, optimizer, batch size and epoch budget.
# The first is a continuation control; the others isolate/combine two changes.
EXPERIMENTS = (
    {"name": "control", "mixed_weight": 1.0, "weight_decay": 0.01},
    {"name": "mixed_1p5", "mixed_weight": 1.5, "weight_decay": 0.01},
    {"name": "mixed_1p5_decay_0p05", "mixed_weight": 1.5, "weight_decay": 0.05},
    {"name": "mixed_2p0_decay_0p05", "mixed_weight": 2.0, "weight_decay": 0.05},
)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_checkpoint(directory: Path) -> dict:
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise FileNotFoundError(f"Missing trained checkpoint config: {directory}")
    if not any((directory / name).is_file() for name in MODEL_WEIGHT_FILES):
        raise FileNotFoundError(f"Missing trained checkpoint weights: {directory}")
    info_file = directory / "acsa_results.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint results: {info_file}")
    result = json.loads(info_file.read_text(encoding="utf-8"))
    if result.get("training_mode") != "full_fine_tuning_no_lora":
        raise ValueError("Checkpoint is not from full fine-tuning")
    if result.get("input_format") != "tokenizer(text=model_text, text_pair='Aspect: CODE — NAME')":
        raise ValueError("Checkpoint uses a different input format from V1")
    if result.get("label_order") != ["Negative", "Neutral", "Positive"]:
        raise ValueError("Checkpoint has unexpected sentiment class order")
    return result


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 ACSA full fine-tuning trials (Validation only)")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models/glot500m/artifacts/acsa/v1_short_search/trial_02_lr1e5_sqrt")
    parser.add_argument("--workbook", type=Path,
                        default=ROOT / "data/splits/citizen_feedback_split.xlsx")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "models/glot500m/artifacts/acsa/stage2_refinement")
    parser.add_argument("--trials", type=int, default=3, choices=range(1, len(EXPERIMENTS)+1))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adafactor", "adamw", "adamw8bit"), default="adafactor")
    parser.add_argument("--on-conflict", choices=("error", "drop"), default="drop")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "models/glot500m/artifacts/acsa/token_cache")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(args: argparse.Namespace):
    if min(args.epochs, args.batch_size, args.gradient_accumulation,
           args.validation_batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch sizes and learning rate must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")
    if not TRAINER.is_file():
        raise FileNotFoundError(f"Copy this script alongside {TRAINER.name}")
    checkpoint = args.checkpoint.resolve()
    workbook = args.workbook.resolve()
    target_root = args.output_root.resolve()
    cache = args.cache_dir.resolve()
    if not workbook.is_file():
        raise FileNotFoundError(f"Missing data workbook: {workbook}")
    if target_root == checkpoint or target_root in checkpoint.parents or checkpoint in target_root.parents:
        raise ValueError("Output root overlaps the starting checkpoint")
    reference = read_checkpoint(checkpoint)
    workbook_hash = sha256_file(workbook)
    if reference.get("data_fingerprint") != workbook_hash:
        raise ValueError("Data workbook differs from checkpoint's training workbook. "
                         "Do not compare experiments with changed data partitions.")
    trainer_hash = sha256_file(TRAINER)
    checkpoint_results_hash = sha256_file(checkpoint / "acsa_results.json")
    ref_macro = float(reference["validation"]["macro_f1"])
    ref_accuracy = float(reference["validation"]["accuracy"])
    ref_mixed = reference.get("validation_mixed_sentiment", {}).get("macro_f1")
    print(f"Starting checkpoint: {checkpoint}", flush=True)
    print(f"Existing Validation Macro-F1: {ref_macro:.4f}", flush=True)
    print("Every trial starts from the same checkpoint. Testing is NEVER read.", flush=True)

    summaries = []
    for idx, variant in enumerate(EXPERIMENTS[:args.trials], start=1):
        run_folder = target_root / f"trial_{idx:02d}_{variant['name']}"
        signature = {
            "checkpoint": str(checkpoint), "checkpoint_results_hash": checkpoint_results_hash,
            "workbook_hash": workbook_hash, "trainer_hash": trainer_hash,
            "trial": variant, "epochs": args.epochs,
            "learning_rate": args.learning_rate, "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "validation_batch_size": args.validation_batch_size, "optimizer": args.optimizer,
            "seed": args.seed, "on_conflict": args.on_conflict,
            "max_length": int(reference.get("training_parameters", {}).get("max_length", 128)),
            "class_weighting": "sqrt", "class_weight_cap": 3.0,
        }
        command = [
            sys.executable, str(TRAINER),
            "--checkpoint", str(checkpoint),
            "--workbook", str(workbook),
            "--output", str(run_folder),
            "--epochs", str(args.epochs), "--patience", "2",
            "--learning-rate", str(args.learning_rate),
            "--head-lr-multiplier", "1",
            "--batch-size", str(args.batch_size),
            "--gradient-accumulation", str(args.gradient_accumulation),
            "--validation-batch-size", str(args.validation_batch_size),
            "--optimizer", args.optimizer,
            "--max-length", str(signature["max_length"]),
            "--class-weighting", "sqrt", "--class-weight-cap", "3",
            "--weight-decay", str(variant["weight_decay"]),
            "--mixed-sentiment-weight", str(variant["mixed_weight"]),
            "--real-weight", "1",
            "--on-conflict", args.on_conflict,
            "--seed", str(args.seed),
            "--cache-dir", str(cache),
            "--log-every", str(args.log_every),
            "--gradient-checkpointing",
        ]
        print(f"\nTrial {idx}/{args.trials}: {variant['name']}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        report_path = run_folder / "acsa_results.json"
        signature_path = run_folder / "_stage2_signature.json"
        if report_path.is_file() and signature_path.is_file():
            if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
                raise ValueError(f"Existing trial does not match settings: {run_folder}")
            result = read_checkpoint(run_folder)
            print("  Reusing completed matching trial", flush=True)
        else:
            if run_folder.is_dir() and any(run_folder.iterdir()):
                raise FileExistsError(f"Incomplete or unrelated folder: {run_folder}. "
                                      "Use a new output-root; previous files are never overwritten.")
            subprocess.run(command, check=True)
            result = read_checkpoint(run_folder)
            signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        mixed = result.get("validation_mixed_sentiment", {})
        summary = {
            "trial": idx, "name": variant["name"],
            "validation_macro_f1": float(result["validation"]["macro_f1"]),
            "validation_accuracy": float(result["validation"]["accuracy"]),
            "mixed_macro_f1": mixed.get("macro_f1"),
            "mixed_contrast_rate": mixed.get("contrast_detection_rate"),
            "best_epoch": result.get("best_epoch"),
            "learning_rate": args.learning_rate,
            "mixed_sentiment_weight": variant["mixed_weight"],
            "weight_decay": variant["weight_decay"],
            "checkpoint_directory": str(run_folder),
        }
        summaries.append(summary)
        target_root.mkdir(parents=True, exist_ok=True)
        write_csv(summaries, target_root / "stage2_results.csv")
        choices = [{"name": "starting_checkpoint", "validation_macro_f1": ref_macro,
                    "validation_accuracy": ref_accuracy, "mixed_macro_f1": ref_mixed,
                    "checkpoint_directory": str(checkpoint)}] + summaries
        # Select by Validation Macro-F1 only; use accuracy to break exact ties.
        winner = max(choices, key=lambda m: (float(m["validation_macro_f1"]),
                                              float(m["validation_accuracy"])))
        (target_root / "stage2_comparison.json").write_text(json.dumps({
            "reference": choices[0], "candidates": summaries,
            "validation_winner": winner,
            "note": "Development-only comparison. No Testing was read or run. "
                    "Do not overwrite the previously frozen final selection. "
                    "Earlier Test results are not an independent assessment of these new experiments.",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Overall Macro-F1={summary['validation_macro_f1']:.4f} "
              f"| mixed={summary['mixed_macro_f1']} "
              f"| best so far={winner['name']}: {winner['validation_macro_f1']:.4f}", flush=True)
    print("\nDry run complete." if args.dry_run else "\nStage-2 tuning finished.", flush=True)


if __name__ == "__main__":
    execute(args_parser())
