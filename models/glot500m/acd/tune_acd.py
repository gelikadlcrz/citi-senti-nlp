"""Targeted Glot500-m LoRA ACD experiments using train_acd.py (fixed Validation).

Default strategy is FOCUSED ablation, not broad random searching:
  1 baseline query/value, ordinary BCE
  2 query/value + weighted BCE
  3 all-linear + ordinary BCE
  4 all-linear + weighted BCE
  5 query/value + real-comment weighting
  6 query/value + real-comment weighting + weighted BCE
  7 all-linear + real-comment weighting
  8 all-linear + real-comment weighting + weighted BCE
All experiments start from original Glot500-m, not continued adapter weights.
Do not compare these validation-tuned scores with independent Test metrics.
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
TRAIN_SCRIPT = PROJECT_ROOT / "models" / "glot500m" / "train_acd.py"
DEFAULT_WORKBOOK = PROJECT_ROOT / "data" / "splits" / "citizen_feedback_split.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "glot500m" / "artifacts" / "acd" / "acd_v3_focused"

BASELINE = {
    "learning_rate": 1e-4,
    "effective_batch_size": 16,
    "epochs": 5,
    "max_length": 128,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_targets": "qv",
    "real_weight": 1.0,
    "synthetic_weight": 1.0,
    "weighted_bce": False,
}
FIELDS = list(BASELINE)
FOCUSED_UPDATES = [
    ("baseline_qv", {}),
    ("positive_bce_qv", {"weighted_bce": True}),
    ("all_linear", {"lora_targets": "all-linear"}),
    ("all_linear_positive_bce", {"lora_targets": "all-linear", "weighted_bce": True}),
    ("real_upweight_qv", {"real_weight": 8.0}),
    ("real_upweight_qv_positive_bce", {"real_weight": 8.0, "weighted_bce": True}),
    ("all_linear_real_upweight", {"lora_targets": "all-linear", "real_weight": 8.0}),
    ("all_linear_real_upweight_positive_bce", {"lora_targets": "all-linear", "real_weight": 8.0,
                                                 "weighted_bce": True}),
]

GRID_SPACE = {
    "learning_rate": [5e-5, 1e-4, 2e-4],
    "effective_batch_size": [8, 16],
    "epochs": [3, 5],
    "weight_decay": [0.0, 0.01],
    "warmup_ratio": [0.0, 0.10],
    "lora_rank": [8, 16],
}


def create_candidates(seed: int, strategy: str = "focused") -> list[dict]:
    if strategy == "focused":
        return [dict(BASELINE, **updates, experiment=name) for name, updates in FOCUSED_UPDATES]
    if strategy != "grid":
        raise ValueError(f"Unknown strategy: {strategy}")
    keys = list(GRID_SPACE)
    configs = [dict(BASELINE, **dict(zip(keys, selection)))
               for selection in itertools.product(*(GRID_SPACE[k] for k in keys))]
    random.Random(seed).shuffle(configs)
    return [dict(cfg, experiment=f"grid_{index:03d}") for index, cfg in enumerate(configs, 1)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_signature(args, config: dict, workbook_hash: str, trainer_hash: str) -> str:
    payload = {
        "trainer_sha256": trainer_hash,
        "workbook_sha256": workbook_hash,
        "checkpoint": args.checkpoint,
        "strategy": args.strategy,
        "seed": args.seed,
        "patience": args.patience,
        "threshold_mode": args.threshold_mode,
        "threshold_grid": "0.05..0.50 by 0.01",
        "microbatch_size": args.microbatch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_checkpointing": args.gradient_checkpointing,
        "trainable_fp32": args.trainable_fp32,
        "precision": args.precision,
        "lora_dropout": args.lora_dropout,
        "max_pos_weight": args.max_pos_weight,
        "config": config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_command(args, trial_dir: Path, config: dict, sig: str) -> list[str]:
    batch = args.microbatch_size
    if config["effective_batch_size"] % batch:
        raise ValueError("effective_batch_size must be divisible by microbatch_size")
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--workbook", str(args.workbook),
        "--output", str(trial_dir),
        "--checkpoint", args.checkpoint,
        "--batch-size", str(batch),
        "--eval-batch-size", str(args.eval_batch_size),
        "--gradient-accumulation", str(config["effective_batch_size"] // batch),
        "--epochs", str(config["epochs"]),
        "--learning-rate", str(config["learning_rate"]),
        "--max-length", str(config["max_length"]),
        "--weight-decay", str(config["weight_decay"]),
        "--warmup-ratio", str(config["warmup_ratio"]),
        "--lora-rank", str(config["lora_rank"]),
        "--lora-alpha", str(config["lora_alpha"]),
        "--lora-targets", str(config["lora_targets"]),
        "--lora-dropout", str(args.lora_dropout),
        "--real-weight", str(config["real_weight"]),
        "--synthetic-weight", str(config["synthetic_weight"]),
        "--max-pos-weight", str(args.max_pos_weight),
        "--threshold-mode", args.threshold_mode,
        "--seed", str(args.seed),
        "--patience", str(args.patience),
        "--precision", args.precision,
        "--log-every", str(args.log_every),
        "--run-signature", sig,
        "--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing",
        "--trainable-fp32" if args.trainable_fp32 else "--no-trainable-fp32",
    ]
    if config["weighted_bce"]:
        command.append("--weighted-bce")
    return command


def read_result(path: Path, signature: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("training_mode") != "lora" or result.get("run_signature") != signature:
        raise ValueError(f"Incompatible completed run at {path}. Use a new --output-root.")
    weights = path.parent / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Model metrics exist but adapter weights are missing: {weights}")
    return result


def save_summary(rows: list[dict], output_root: Path, args) -> dict:
    ranked = sorted(rows, key=lambda r: (-r["macro_f1"], r["trial"]))
    cols = ["rank", "trial", "experiment", "macro_f1", "micro_f1", "exact_set_match",
            "real_macro_f1", "real_comments", "synthetic_macro_f1", "synthetic_comments",
            "best_epoch", "global_threshold", *FIELDS, "checkpoint_directory"]
    with (output_root / "search_results.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=cols)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({k: (rank if k == "rank" else row.get(k, "")) for k in cols})
    best = ranked[0]
    (output_root / "best_configuration.json").write_text(json.dumps({
        "strategy": args.strategy,
        "trial": best["trial"],
        "experiment": best["experiment"],
        "seed": args.seed,
        "validation_macro_f1": best["macro_f1"],
        "real_validation_macro_f1": best["real_macro_f1"],
        "checkpoint_directory": best["checkpoint_directory"],
        "hyperparameters": {k: best[k] for k in FIELDS},
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Selected on Validation, including synthetic records if present. "
                   "No independent Test or nested group cross-validation; synthetic leakage unverified.",
    }, indent=2), encoding="utf-8")
    return best


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CITI-Senti Glot500 ACD targeted LoRA comparisons")
    p.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    p.add_argument("--strategy", choices=["focused", "grid"], default="focused")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--threshold-mode", choices=["global", "eligible-category"], default="global")
    p.add_argument("--microbatch-size", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trainable-fp32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--lora-dropout", type=float, default=0.10)
    p.add_argument("--max-pos-weight", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    candidates = create_candidates(args.seed, args.strategy)
    if not 1 <= args.trials <= len(candidates):
        raise ValueError(f"--trials must be 1..{len(candidates)} for {args.strategy}")
    if args.eval_batch_size < 1 or args.patience < 1:
        raise ValueError("Evaluation batch size and patience must be >=1")
    if any(c["effective_batch_size"] % args.microbatch_size for c in candidates[:args.trials]):
        raise ValueError("effective_batch_size must divide by microbatch size")
    print(f"Selected {args.trials} of {len(candidates)} {args.strategy} LoRA experiments (seed {args.seed})")
    for i, cfg in enumerate(candidates[:args.trials], 1):
        print(f"  {i:02d} {cfg['experiment']}: " + ", ".join(f"{k}={cfg[k]}" for k in
             ("lora_targets", "lora_rank", "real_weight", "weighted_bce", "epochs")))
    if args.dry_run:
        print("Dry run: no training and no files written.")
        return
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Expected companion train_acd.py at: {TRAIN_SCRIPT}")
    if not args.workbook.is_file():
        raise FileNotFoundError(f"Missing workbook: {args.workbook}")
    source_hash = sha256_file(TRAIN_SCRIPT)
    workbook_hash = sha256_file(args.workbook)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, config in enumerate(candidates[:args.trials], 1):
        trial_dir = args.output_root / f"trial_{i:02d}_{config['experiment']}"
        result_file = trial_dir / "acd_results.json"
        sig = run_signature(args, config, workbook_hash, source_hash)
        if result_file.is_file():
            print(f"\n[trial {i}/{args.trials}] Resuming verified experiment: {trial_dir}", flush=True)
            result = read_result(result_file, sig)
        else:
            if trial_dir.exists() and any(trial_dir.iterdir()):
                raise FileExistsError(f"Incomplete trial output exists: {trial_dir}. "
                                      "Use a new output root or back up/remove this trial directory.")
            print(f"\n[trial {i}/{args.trials}] Training {config['experiment']}", flush=True)
            subprocess.run(build_command(args, trial_dir, config, sig), check=True, cwd=PROJECT_ROOT)
            if not result_file.is_file():
                raise RuntimeError(f"Missing results: {result_file}")
            result = read_result(result_file, sig)
        metrics = result["validation_after_threshold_tuning"]
        by_source = result.get("validation_by_source", {})
        real = by_source.get("Real", {})
        synth = by_source.get("Synthetic", {})
        row = {
            "trial": i,
            "experiment": config["experiment"],
            "macro_f1": float(metrics["macro_f1"]),
            "micro_f1": float(metrics["micro_f1"]),
            "exact_set_match": float(metrics["exact_set_match"]),
            "real_macro_f1": real.get("macro_f1"),
            "real_comments": real.get("comments", 0),
            "synthetic_macro_f1": synth.get("macro_f1"),
            "synthetic_comments": synth.get("comments", 0),
            "best_epoch": int(result["best_epoch"]),
            "global_threshold": float(result["global_threshold"]),
            "checkpoint_directory": str(trial_dir.resolve()),
            **{key: config[key] for key in FIELDS},
        }
        rows.append(row)
        best = save_summary(rows, args.output_root, args)
        print(f"Trial {i}: Macro-F1={row['macro_f1']:.4f} "
              f"Real subgroup Macro-F1={row['real_macro_f1']}", flush=True)
    print(f"\nBest overall Validation Macro-F1: {best['macro_f1']:.4f} ({best['experiment']})")
    print(f"Best adapter: {best['checkpoint_directory']}")
    print(f"Ranked results: {args.output_root / 'search_results.csv'}")
    print("NOTE: Model/threshold selection uses Validation, not independent Test.")


if __name__ == "__main__":
    main()
