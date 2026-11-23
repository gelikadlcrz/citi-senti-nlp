"""CITI-Senti GLOT500-m multilabel aspect detection — optimized development trainer.

Inputs: existing citizen_feedback_split.xlsx (Training and Validation only).
This script NEVER reads Testing. The Validation scores are development results,
not independent test estimates or the final nested-grouped-CV protocol.

Performance changes compared with the original:
- train-only positive-class weighting and a separate LR for new classifier head
- larger inference-only validation batches
- no MPS -> CPU synchronization on *every* training batch
- memory-safe checkpoint reload on a 16-GB Apple Silicon machine
- optional tokenization cache for repeated hyperparameter trials
- comprehensive prediction-count, AP, and class-level diagnostics

Use --check-data before training. Save each run to a new --output directory.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data import TRAIN_SHEET, VAL_SHEET, make_aspect_task, read_annotation_sheet
from shared.labels import ASPECT_CODES, ASPECT_NAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(workbook: Path):
    if not workbook.is_file():
        raise FileNotFoundError(f"Missing split workbook: {workbook}")
    train = make_aspect_task(read_annotation_sheet(workbook, TRAIN_SHEET))
    val = make_aspect_task(read_annotation_sheet(workbook, VAL_SHEET))
    train_ids = [str(row["comment_id"]) for row in train.row_metadata]
    val_ids = [str(row["comment_id"]) for row in val.row_metadata]
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("Each comment must appear only once within a task split")
    if set(train_ids).intersection(val_ids):
        raise ValueError("Training and Validation share comment IDs")
    for name, part in (("Training", train), ("Validation", val)):
        if part.labels.shape != (len(part.texts), len(ASPECT_CODES)):
            raise ValueError(f"{name}: labels must be N x {len(ASPECT_CODES)}")
        if not np.isin(part.labels, [0, 1]).all():
            raise ValueError(f"{name}: aspect labels must be binary")
        if any(not str(text).strip() for text in part.texts):
            raise ValueError(f"{name}: empty model_text present")
    return train, val


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_positive_weights(labels: np.ndarray, mode: str, cap: float) -> np.ndarray:
    """Compute weights from Training labels only; no Validation/Test usage."""
    positives = labels.astype(np.float64).sum(axis=0)
    negatives = len(labels) - positives
    weights = np.ones(labels.shape[1], dtype=np.float64)
    present = positives > 0
    if mode == "sqrt":
        weights[present] = np.sqrt(negatives[present] / positives[present])
    elif mode == "full":
        weights[present] = negatives[present] / positives[present]
    elif mode != "none":
        raise ValueError(f"Unknown weighting mode: {mode}")
    return np.clip(weights, 1.0, cap).astype(np.float32)


class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, cache_dir=None, checkpoint=""):
        self.labels = labels.astype(np.float32)
        # Hash actual text to prevent accidental reuse after dataset updates.
        fingerprint = hashlib.sha256()
        fingerprint.update(str((checkpoint, max_length, tokenizer.vocab_size)).encode())
        for text in texts:
            fingerprint.update(str(text).encode("utf-8"))
            fingerprint.update(b"\x00")
        cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"acd_tokens_{fingerprint.hexdigest()[:24]}.json.gz"

        if cache_path is not None and cache_path.is_file():
            with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
                self.features = json.load(stream)
            print(f"Tokenization cache hit: {cache_path.name}", flush=True)
        else:
            encoded = tokenizer(
                list(texts), max_length=max_length, truncation=True,
                padding=False,
            )
            self.features = [
                {name: values[i] for name, values in encoded.items()}
                for i in range(len(texts))
            ]
            if cache_path is not None:
                with gzip.open(cache_path, "wt", encoding="utf-8") as stream:
                    json.dump(self.features, stream)
                print(f"Tokenization cache saved: {cache_path.name}", flush=True)
        if len(self.features) != len(texts):
            raise ValueError("Cached tokenizer output count differs from dataset")
        self.lengths = np.asarray([len(row["input_ids"]) for row in self.features])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**self.features[index], "labels": self.labels[index]}


def collate_factory(tokenizer):
    def collate(items):
        features = [{k: v for k, v in row.items() if k != "labels"} for row in items]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.from_numpy(np.stack([row["labels"] for row in items]))
        return batch
    return collate


def scores(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = (probs >= thresholds.reshape(1, -1)).astype(np.int8)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "exact_set_match": float(np.mean(np.all(pred == y, axis=1))),
        "predicted_positive_labels": int(pred.sum()),
        "mean_predicted_labels_per_comment": float(pred.sum(axis=1).mean()),
    }


def macro_average_precision(y: np.ndarray, probs: np.ndarray) -> float:
    # Undefined for categories with no positives OR no negatives. Ignore those.
    eligible = [i for i in range(y.shape[1]) if 0 < int(y[:, i].sum()) < len(y)]
    if not eligible:
        return 0.0
    return float(np.mean([average_precision_score(y[:, i], probs[:, i]) for i in eligible]))


def tune_thresholds(y: np.ndarray, probs: np.ndarray, mode: str,
                    max_mean_labels: float | None = None):
    """Development-only threshold tuning on Validation (not held-out test)."""
    grid = np.round(np.arange(0.05, 0.951, 0.025), 3)
    candidates = []
    for t in grid:
        predictions = (probs >= t).astype(np.int8)
        mean_labels = predictions.sum(axis=1).mean()
        if max_mean_labels is not None and mean_labels > max_mean_labels:
            continue
        macro = float(f1_score(y, predictions, average="macro", zero_division=0))
        # Prefer simpler/sparser predictions when performance ties.
        candidates.append((macro, -float(mean_labels), -abs(float(t) - 0.5), float(t)))
    if not candidates:
        # If a hard label-count constraint is infeasible, avoid a silent failure.
        t = float(grid[np.argmin([abs((probs >= x).sum(axis=1).mean() - max_mean_labels)
                                     for x in grid])])
        print("Warning: no threshold satisfied --max-label-multiplier; "
              f"using closest threshold {t:.3f}.", flush=True)
    else:
        t = max(candidates)[-1]
    thresholds = np.full(y.shape[1], t, dtype=np.float64)
    eligible = []
    if mode == "eligible-category":
        # Avoid 11 freely fitted thresholds for very rare/unstable categories.
        for i, code in enumerate(ASPECT_CODES):
            pos = int(y[:, i].sum())
            if pos < 25 or len(y) - pos < 25:
                continue
            eligible.append(code)
            options = []
            for candidate in grid:
                pred = probs[:, i] >= candidate
                f1 = float(f1_score(y[:, i], pred, zero_division=0))
                options.append((f1, -int(pred.sum()), -abs(float(candidate) - t), float(candidate)))
            thresholds[i] = max(options)[-1]
    elif mode != "global":
        raise ValueError("--threshold-mode must be global or eligible-category")
    return thresholds, t, eligible


@torch.inference_mode()
def infer_probabilities(model, loader: DataLoader, device: torch.device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        labels.append(batch["labels"].numpy().astype(np.int8))
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs).logits
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def category_diagnostics(y, probs, thresholds):
    pred = (probs >= thresholds.reshape(1, -1)).astype(np.int8)
    result = {}
    for i, code in enumerate(ASPECT_CODES):
        yi, pi = y[:, i], pred[:, i]
        tp = int(((yi == 1) & (pi == 1)).sum())
        fp = int(((yi == 0) & (pi == 1)).sum())
        fn = int(((yi == 1) & (pi == 0)).sum())
        positive_count = int(yi.sum())
        result[code] = {
            "support": positive_count,
            "predicted_positives": int(pi.sum()),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "average_precision": (float(average_precision_score(yi, probs[:, i]))
                                  if 0 < positive_count < len(y) else None),
            "mean_probability": float(probs[:, i].mean()),
        }
    return result


def create_optimizer(model, args):
    """Use a higher LR for the randomly initialized classifier head."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = []
    for head in (False, True):
        for decay in (False, True):
            params = [
                p for name, p in model.named_parameters()
                if p.requires_grad
                and name.startswith("classifier.") == head
                and (not any(term in name for term in no_decay)) == decay
            ]
            if params:
                groups.append({
                    "params": params,
                    "lr": args.learning_rate * (args.head_lr_multiplier if head else 1.0),
                    "weight_decay": args.weight_decay if decay else 0.0,
                })
    # foreach=False avoids extra full-model temporary allocations on 16 GB RAM.
    return torch.optim.AdamW(groups, foreach=False)


def train(args):
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.epochs < 1:
        raise ValueError("batch-size, gradient-accumulation, and epochs must be positive")
    if args.validation_batch_size < 1 or args.log_every < 0:
        raise ValueError("validation-batch-size must be positive and log-every nonnegative")
    if args.head_lr_multiplier <= 0 or args.max_pos_weight < 1:
        raise ValueError("head-lr-multiplier must be positive and max-pos-weight >= 1")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup-ratio must be between 0 and 1")
    if args.max_label_multiplier < 0:
        raise ValueError("max-label-multiplier cannot be negative")
    train_task, val_task = load_data(args.workbook)
    train_ids = [str(row["comment_id"]) for row in train_task.row_metadata]
    val_ids = [str(row["comment_id"]) for row in val_task.row_metadata]
    print(f"ACD records: train={len(train_task.texts)}, validation={len(val_task.texts)}", flush=True)
    print("Label support (train / validation):")
    for i, code in enumerate(ASPECT_CODES):
        print(f"  {code}: {int(train_task.labels[:, i].sum())} / "
              f"{int(val_task.labels[:, i].sum())}")
    if args.check_data:
        print("Data check passed. Training and Validation comment IDs are disjoint.")
        print("Note: this check does not independently validate source/near-duplicate grouping.")
        return
    if (args.output / "acd_results.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed run exists in {args.output}; use a NEW --output "
                              "or explicitly supply --overwrite")

    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    set_seed(args.seed)
    device = choose_device()
    print(f"Device: {device} | model: {args.checkpoint}", flush=True)
    if device.type == "cpu":
        print("Warning: full GLOT500-m fine-tuning is very slow without a GPU")
    weights = compute_positive_weights(train_task.labels, args.loss_weighting,
                                       args.max_pos_weight)
    print(f"Loss weighting: {args.loss_weighting}; head LR multiplier: "
          f"{args.head_lr_multiplier:g}; cap: {args.max_pos_weight:g}")
    print("Positive class weights: " + ", ".join(
        f"{code}={weight:.2f}" for code, weight in zip(ASPECT_CODES, weights)
    ), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    # Pre-tokenize once; dynamic pad to the batch's actual maximum length.
    train_ds = AspectDataset(train_task.texts, train_task.labels, tokenizer,
                             args.max_length, args.cache_dir, args.checkpoint)
    val_ds = AspectDataset(val_task.texts, val_task.labels, tokenizer,
                           args.max_length, args.cache_dir, args.checkpoint)
    for name, ds in (("Training", train_ds), ("Validation", val_ds)):
        print(f"{name} tokens: median={np.median(ds.lengths):.0f}, "
              f"p95={np.percentile(ds.lengths, 95):.0f}, "
              f"at max length={int((ds.lengths == args.max_length).sum())}", flush=True)
    collate = collate_factory(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.validation_batch_size, shuffle=False, num_workers=0,
        collate_fn=collate, pin_memory=device.type == "cuda"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=len(ASPECT_CODES),
        problem_type="multi_label_classification",
        id2label={i: code for i, code in enumerate(ASPECT_CODES)},
        label2id={code: i for i, code in enumerate(ASPECT_CODES)},
    )
    if args.gradient_checkpointing:
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)
    optimizer = create_optimizer(model, args)
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(total_steps * args.warmup_ratio), max(1, total_steps)
    )
    pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {model_trainable:,}; "
          f"optimizer updates/epoch: {math.ceil(len(train_loader) / args.gradient_accumulation)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    best_value = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    label_mean = float(train_task.labels.sum(axis=1).mean())
    max_mean_labels = (label_mean * args.max_label_multiplier
                       if args.max_label_multiplier > 0 else None)
    fixed_threshold = np.full(len(ASPECT_CODES), 0.5)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        log_loss = torch.zeros((), device=device)
        log_start = 0
        for index, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            group_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    logits = model(**inputs).logits
                    loss = F.binary_cross_entropy_with_logits(
                        logits.float(), labels.float(), pos_weight=pos_weight
                    )
                scaler.scale(loss / group_size).backward()
            else:
                logits = model(**inputs).logits
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), labels.float(), pos_weight=pos_weight
                )
                (loss / group_size).backward()
            # Accumulate loss *on device*: no per-batch .item() synchronization.
            loss_total += loss.detach()
            log_loss += loss.detach()
            is_update = ((index + 1) % args.gradient_accumulation == 0
                         or index + 1 == len(train_loader))
            if is_update:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if args.log_every and (index + 1) % args.log_every == 0:
                window_size = index + 1 - log_start
                print(f"  Epoch {epoch}: batch {index+1}/{len(train_loader)} "
                      f"| avg_loss={(log_loss/window_size).item():.4f}", flush=True)
                log_start = index + 1
                log_loss.zero_()
        average_loss = float((loss_total / len(train_loader)).item())
        y, probabilities = infer_probabilities(model, val_loader, device)
        fixed = scores(y, probabilities, fixed_threshold)
        thresholds, global_t, _ = tune_thresholds(y, probabilities, args.threshold_mode,
                                                  max_mean_labels=max_mean_labels)
        tuned = scores(y, probabilities, thresholds)
        ap = macro_average_precision(y, probabilities)
        history.append({
            "epoch": epoch,
            "loss": round(average_loss, 6),
            **fixed,
            "macro_f1_threshold_tuned": tuned["macro_f1"],
            "micro_f1_threshold_tuned": tuned["micro_f1"],
            "global_threshold": global_t,
            "macro_average_precision": ap,
            "predicted_positives_at_tuned": tuned["predicted_positive_labels"],
            "mean_predicted_labels_at_tuned": tuned["mean_predicted_labels_per_comment"],
        })
        print(f"Epoch {epoch}/{args.epochs}: loss={average_loss:.4f}, "
              f"val_macro_f1@0.50={fixed['macro_f1']:.4f}, "
              f"val_macro_f1@tuned={tuned['macro_f1']:.4f}, "
              f"val_macro_AP={ap:.4f}, threshold={global_t:.3f}, "
              f"positive_predictions={tuned['predicted_positive_labels']}", flush=True)
        value = ap if args.selection_metric == "macro_ap" else tuned["macro_f1"]
        if value > best_value + 1e-8:
            best_value = value
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(args.output, safe_serialization=True)
            tokenizer.save_pretrained(args.output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    # IMPORTANT: optimizer states are large on Apple unified memory. Release them
    # BEFORE loading another full copy of the model checkpoint.
    del model, optimizer, scheduler, scaler, loss_total, log_loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(args.output).to(device)
    y, probabilities = infer_probabilities(model, val_loader, device)
    thresholds, global_t, eligible = tune_thresholds(
        y, probabilities, args.threshold_mode, max_mean_labels=max_mean_labels
    )
    tuned_metrics = scores(y, probabilities, thresholds)
    fixed_metrics = scores(y, probabilities, fixed_threshold)
    final_ap = macro_average_precision(y, probabilities)
    np.savez_compressed(
        args.output / "validation_probabilities.npz",
        gold=y, probabilities=probabilities,
        comment_ids=np.asarray(val_ids), aspect_codes=np.asarray(ASPECT_CODES)
    )
    results = {
        "checkpoint": args.checkpoint,
        "best_epoch": best_epoch,
        "best_epoch_macro_f1_at_0_5": history[best_epoch-1]["macro_f1"],
        "best_epoch_macro_f1_after_threshold_tuning": history[best_epoch-1]["macro_f1_threshold_tuned"],
        "validation_after_threshold_tuning": tuned_metrics,
        "validation_at_0_5": fixed_metrics,
        "validation_macro_average_precision": final_ap,
        "global_threshold": global_t,
        "threshold_mode": args.threshold_mode,
        "category_thresholds": {code: float(thresholds[i]) for i, code in enumerate(ASPECT_CODES)},
        "thresholds_eligible_categories": eligible,
        "validation_per_category": category_diagnostics(y, probabilities, thresholds),
        "label_names": {code: ASPECT_NAMES[code] for code in ASPECT_CODES},
        "positive_class_weights": {code: float(weights[i]) for i, code in enumerate(ASPECT_CODES)},
        "training_parameters": {
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "epochs_max": args.epochs,
            "learning_rate": args.learning_rate,
            "head_lr_multiplier": args.head_lr_multiplier,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "seed": args.seed,
            "gradient_checkpointing": args.gradient_checkpointing,
            "validation_batch_size": args.validation_batch_size,
            "loss_weighting": args.loss_weighting,
            "max_pos_weight": args.max_pos_weight,
            "max_label_multiplier": args.max_label_multiplier,
            "selection_metric": args.selection_metric,
        },
        "data_source": str(args.workbook),
        "data_fingerprint": sha256_file(args.workbook),
        "trainer_fingerprint": sha256_file(Path(__file__)),
        "train_comment_count": len(train_ids),
        "validation_comment_count": len(val_ids),
        "history": history,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Exploratory fixed-Validation development run, NOT grouped nested CV. "
                "Validation reused for checkpoint/threshold selection; do not treat this as "
                "independent test performance. Testing sheet was not opened.",
    }
    (args.output / "acd_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Best epoch: {best_epoch}; selection metric: {args.selection_metric}")
    print("Validation Macro-F1 after threshold tuning:", tuned_metrics["macro_f1"])
    print("Validation Macro-AP:", final_ap)
    print("Saved ACD model to:", args.output.resolve())


def build_parser():
    parser = argparse.ArgumentParser(description="Optimized GLOT500-m ACD development trainer")
    parser.add_argument("--workbook", type=Path,
                        default=PROJECT_ROOT / "data/splits/citizen_feedback_split.xlsx")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models/glot500m/artifacts/acd/optimized_v1")
    parser.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    parser.add_argument("--max-length", type=int, choices=(128, 256), default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-lr-multiplier", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-mode", choices=("global", "eligible-category"), default="global")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--loss-weighting", choices=("none", "sqrt", "full"), default="sqrt")
    parser.add_argument("--max-pos-weight", type=float, default=5.0)
    parser.add_argument("--max-label-multiplier", type=float, default=0.0,
                        help="Optional upper bound on mean predicted labels vs Training; 0 disables")
    parser.add_argument("--selection-metric", choices=("macro_f1", "macro_ap"), default="macro_f1")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
