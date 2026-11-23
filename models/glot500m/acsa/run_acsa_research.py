"""Sequential V1-compatible ACSA research trials (Training/Validation only).

Does not touch original V1 code, previously selected models, or Testing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_acsa_research.py"
WEIGHTS = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
TRIALS = [
    {"name": "mixed_1p5", "mixed_sentiment_weight": 1.5, "real_weight": 1.0, "weight_decay": 0.01},
    {"name": "mixed_2p0", "mixed_sentiment_weight": 2.0, "real_weight": 1.0, "weight_decay": 0.01},
    {"name": "decay_0p05", "mixed_sentiment_weight": 1.0, "real_weight": 1.0, "weight_decay": 0.05},
    {"name": "real_2p0", "mixed_sentiment_weight": 1.0, "real_weight": 2.0, "weight_decay": 0.01},
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_checkpoint(path: Path) -> dict:
    if not path.is_dir() or not (path / "config.json").is_file() or not any((path / x).is_file() for x in WEIGHTS):
        raise FileNotFoundError(f"Full ACSA checkpoint is missing model/config files: {path}")
    results = path / "acsa_results.json"
    if not results.is_file():
        raise FileNotFoundError(f"ACSA results missing: {results}")
    info = json.loads(results.read_text(encoding="utf-8"))
    if info.get("training_mode") != "full_fine_tuning_no_lora":
        raise ValueError("Expected full fine-tuning checkpoint, not LoRA")
    if info.get("input_format") != "tokenizer(text=model_text, text_pair='Aspect: CODE — NAME')":
        raise ValueError("Checkpoint is not compatible with the V1 comment-first input format")
    if info.get("label_order") != ["Negative", "Neutral", "Positive"]:
        raise ValueError("Checkpoint sentiment class order does not match")
    return info


def parser():
    p = argparse.ArgumentParser(description="Separate V1-style ACSA research experiments")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "models/glot500m/artifacts/acsa/full_ft_v1")
    p.add_argument("--reference", type=Path,
                   default=ROOT / "models/glot500m/artifacts/acsa/v1_short_search/trial_02_lr1e5_sqrt",
                   help="Existing 0.7719 Validation reference for comparison only")
    p.add_argument("--workbook", type=Path,
                   default=ROOT / "data/splits/citizen_feedback_split.xlsx")
    p.add_argument("--source-map", type=Path, default=None,
                   help="Optional verified comment_id,source provenance CSV")
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "models/glot500m/artifacts/acsa/research_weights_v1")
    p.add_argument("--trials", type=int, choices=range(1, 5), metavar="{1,2,3,4}", default=3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--validation-batch-size", type=int, default=2)
    p.add_argument("--optimizer", default="adafactor", choices=["adafactor", "adamw", "adamw8bit"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--on-conflict", default="drop", choices=["error", "drop"])
    p.add_argument("--cache-dir", type=Path,
                   default=ROOT / "models/glot500m/artifacts/acsa/token_cache")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    a = parser()
    if a.epochs <= 0 or a.learning_rate <= 0 or a.batch_size <= 0 or a.gradient_accumulation <= 0:
        raise ValueError("Epochs, LR and batch sizes must be positive")
    if a.log_every < 0:
        raise ValueError("--log-every must be nonnegative")
    checkpoint = a.checkpoint.resolve()
    workbook = a.workbook.resolve()
    output = a.output_root.resolve()
    reference = a.reference.resolve()
    if not workbook.is_file():
        raise FileNotFoundError(f"Split workbook not found: {workbook}")
    if not TRAINER.is_file():
        raise FileNotFoundError(f"Place run_acsa_research.py beside {TRAINER.name}")
    if checkpoint == output or checkpoint in output.parents or output in checkpoint.parents:
        raise ValueError("Output-root must not overlap the starting checkpoint")
    if reference == output or reference in output.parents or output in reference.parents:
        raise ValueError("Output-root must not overlap the reference checkpoint")
    source_info = check_checkpoint(checkpoint)
    data_hash = digest(workbook)
    if source_info.get("data_fingerprint") != data_hash:
        raise ValueError("Workbook differs from the checkpoint's training dataset. "
                         "Do not warm-start and compare across changed partitions without review.")
    ref_info = check_checkpoint(reference) if reference.is_dir() else None
    if ref_info is not None and ref_info.get("data_fingerprint") != data_hash:
        raise ValueError("Reference checkpoint was trained with a different workbook")
    source_map = a.source_map.resolve() if a.source_map else None
    if source_map and not source_map.is_file():
        raise FileNotFoundError(f"Source map not found: {source_map}")
    map_hash = digest(source_map) if source_map else "none"
    trainer_hash = digest(TRAINER)
    print("Starting checkpoint:", checkpoint, flush=True)
    print("Validation reference:", reference if ref_info else "not available", flush=True)
    print("NO TEST SET IS READ. Existing final selection is not modified.", flush=True)
    summary = []
    for i, spec in enumerate(TRIALS[:a.trials], 1):
        target = output / f"trial_{i:02d}_{spec['name']}"
        config = {
            "checkpoint": str(checkpoint), "workbook_sha256": data_hash,
            "source_map_sha256": map_hash, "trainer_sha256": trainer_hash,
            "trial": spec, "epochs": a.epochs, "learning_rate": a.learning_rate,
            "batch_size": a.batch_size, "gradient_accumulation": a.gradient_accumulation,
            "validation_batch_size": a.validation_batch_size, "optimizer": a.optimizer,
            "seed": a.seed, "on_conflict": a.on_conflict,
            "max_length": source_info["training_parameters"].get("max_length", 128),
            "class_weighting": "sqrt", "class_weight_cap": 3.0,
        }
        commands = [
            sys.executable, str(TRAINER), "--checkpoint", str(checkpoint),
            "--workbook", str(workbook), "--output", str(target),
            "--epochs", str(a.epochs), "--patience", "2",
            "--learning-rate", str(a.learning_rate), "--head-lr-multiplier", "1",
            "--batch-size", str(a.batch_size),
            "--gradient-accumulation", str(a.gradient_accumulation),
            "--validation-batch-size", str(a.validation_batch_size),
            "--max-length", str(config["max_length"]),
            "--optimizer", a.optimizer, "--class-weighting", "sqrt",
            "--class-weight-cap", "3", "--weight-decay", str(spec["weight_decay"]),
            "--mixed-sentiment-weight", str(spec["mixed_sentiment_weight"]),
            "--real-weight", str(spec["real_weight"]),
            "--cache-dir", str(a.cache_dir.resolve()),
            "--on-conflict", a.on_conflict, "--seed", str(a.seed),
            "--log-every", str(a.log_every), "--gradient-checkpointing",
        ]
        if source_map:
            commands.extend(["--source-map", str(source_map)])
        print(f"\nExperiment {i}/{a.trials}: {spec['name']} ({a.epochs} added epochs)", flush=True)
        if a.dry_run:
            print(subprocess.list2cmdline(commands), flush=True)
            continue
        result_file = target / "acsa_results.json"
        signature_file = target / "_research_config.json"
        if result_file.is_file() and signature_file.is_file():
            existing = json.loads(signature_file.read_text(encoding="utf-8"))
            if existing != config:
                raise ValueError(f"Existing completed trial settings do not match: {target}")
            info = check_checkpoint(target)
            print("  Reusing existing completed trial", flush=True)
        else:
            if target.exists() and any(target.iterdir()):
                raise FileExistsError(f"Incomplete or unrelated trial folder: {target}. "
                                      "Move it aside or select a new output-root")
            subprocess.run(commands, check=True)
            info = check_checkpoint(target)
            signature_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        m = info.get("validation_mixed_sentiment", {})
        row = {
            "trial": i, "name": spec["name"],
            "validation_macro_f1": float(info["validation"]["macro_f1"]),
            "validation_accuracy": float(info["validation"]["accuracy"]),
            "mixed_sentiment_macro_f1": m.get("macro_f1"),
            "mixed_contrast_detection_rate": m.get("contrast_detection_rate"),
            "best_epoch": info.get("best_epoch"),
            "mixed_sentiment_weight": spec["mixed_sentiment_weight"],
            "real_weight": spec["real_weight"],
            "weight_decay": spec["weight_decay"],
            "checkpoint_directory": str(target),
        }
        summary.append(row)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "research_results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        best = max(summary, key=lambda r: (r["validation_macro_f1"], r["validation_accuracy"]))
        (output / "development_comparison.json").write_text(json.dumps({
            "best_research_trial": best,
            "reference_checkpoint": str(reference) if ref_info else None,
            "reference_validation_macro_f1": ref_info["validation"]["macro_f1"] if ref_info else None,
            "note": "Validation-only exploration. Do not change previously frozen final selection "
                    "or reuse prior Testing results to pick a new checkpoint.",
        }, indent=2), encoding="utf-8")
        print(f"  Validation Macro-F1={row['validation_macro_f1']:.4f}; "
              f"mixed={row['mixed_sentiment_macro_f1']}", flush=True)
    print("\nDry run complete." if a.dry_run else "\nResearch trials completed.", flush=True)


if __name__ == "__main__":
    main()
