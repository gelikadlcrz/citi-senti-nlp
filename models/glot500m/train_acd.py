from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data import TRAIN_SHEET, VAL_SHEET, make_aspect_task, read_annotation_sheet
from shared.labels import ASPECT_CODES, ASPECT_NAMES


class AspectDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.features = [
            {key: encoded[key][index] for key in encoded}
            for index in range(len(texts))
        ]
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {**self.features[idx], "labels": self.labels[idx]}


def collate_factory(tokenizer):
    def collate(items: list[dict]):
        inputs = [{key: value for key, value in item.items() if key != "labels"} for item in items]
        batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(
            np.stack([item["labels"] for item in items]), dtype=torch.float32
        )
        return batch

    return collate


def load_data(workbook: Path):
    train = make_aspect_task(read_annotation_sheet(workbook, TRAIN_SHEET))
    val = make_aspect_task(read_annotation_sheet(workbook, VAL_SHEET))
    train_ids = [record["comment_id"] for record in train.row_metadata]
    val_ids = [record["comment_id"] for record in val.row_metadata]
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("One or more comment IDs appear twice within a split")
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and Validation share comment IDs")
    if train.labels.shape[1] != len(ASPECT_CODES) or val.labels.shape[1] != len(ASPECT_CODES):
        raise ValueError("Expected 11 aspect-category indicators")
    return train, val


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scores(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = (probs >= thresholds.reshape(1, -1)).astype(np.int8)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "exact_set_match": float(np.mean(np.all(pred == y, axis=1))),
    }


def tune_thresholds(y: np.ndarray, probs: np.ndarray, mode: str):
    grid = np.round(np.arange(0.20, 0.801, 0.05), 2)
    # Choose the global threshold; tie-break toward 0.5 for stability.
    candidates = [
        (scores(y, probs, np.full(len(ASPECT_CODES), t))["macro_f1"],
         -abs(float(t) - 0.5), float(t))
        for t in grid
    ]
    best_global = max(candidates)[2]
    thresholds = np.full(len(ASPECT_CODES), best_global, dtype=float)
    eligible = []
    if mode == "eligible-category":
        for i, code in enumerate(ASPECT_CODES):
            positives = int(y[:, i].sum())
            negatives = int(len(y) - positives)
            if positives < 20 or negatives < 20:
                continue
            eligible.append(code)
            # Per-category tuning is allowed only for sufficient validation support.
            category_options = []
            for t in grid:
                f1 = f1_score(y[:, i], probs[:, i] >= t, zero_division=0)
                category_options.append((float(f1), -abs(float(t) - best_global), float(t)))
            thresholds[i] = max(category_options)[2]
    return thresholds, best_global, eligible


@torch.inference_mode()
def infer_probabilities(model, loader: DataLoader, device: torch.device):
    model.eval()
    outputs, labels = [], []
    for batch in loader:
        labels.append(batch["labels"].numpy().astype(np.int8))
        features = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        logits = model(**features).logits
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(labels), np.concatenate(outputs)


def train(args: argparse.Namespace):
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.epochs < 1:
        raise ValueError("Batch size, gradient accumulation and epochs must be positive")
    train_task, val_task = load_data(args.workbook)
    print(f"ACD records: train={len(train_task.texts)}, validation={len(val_task.texts)}")
    print("Label support (train / validation):")
    for i, code in enumerate(ASPECT_CODES):
        print(f"  {code}: {int(train_task.labels[:, i].sum())} / {int(val_task.labels[:, i].sum())}")
    if args.check_data:
        print("Data check passed. Training and Validation are disjoint.")
        return

    # Delay heavy model imports until after data validation.
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | model: {args.checkpoint}")
    if device.type == "cpu":
        print("Warning: full Glot500-m fine-tuning on CPU can be very slow.")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=len(ASPECT_CODES),
        problem_type="multi_label_classification",
        id2label={index: code for index, code in enumerate(ASPECT_CODES)},
        label2id={code: index for index, code in enumerate(ASPECT_CODES)},
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    train_ds = AspectDataset(train_task.texts, train_task.labels, tokenizer, args.max_length)
    val_ds = AspectDataset(val_task.texts, val_task.labels, tokenizer, args.max_length)
    collate = collate_factory(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # safe on Windows
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(x in n for x in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    fixed_threshold = np.full(len(ASPECT_CODES), 0.5)

    args.output.mkdir(parents=True, exist_ok=True)
    best_macro = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for index, batch in enumerate(train_loader):
            features = {key: value.to(device) for key, value in batch.items()}
            group_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            current_group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = model(**features).loss
            epoch_loss += float(loss.detach().item())
            scaler.scale(loss / current_group_size).backward()
            is_step = (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(train_loader)
            if is_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        gold, probabilities = infer_probabilities(model, val_loader, device)
        metric = scores(gold, probabilities, fixed_threshold)
        epoch_thresholds, epoch_global_threshold, _ = tune_thresholds(
            gold, probabilities, args.threshold_mode
        )
        tuned_metric = scores(gold, probabilities, epoch_thresholds)
        history.append({
            "epoch": epoch,
            "loss": round(epoch_loss / len(train_loader), 5),
            **metric,
            "macro_f1_threshold_tuned": tuned_metric["macro_f1"],
            "micro_f1_threshold_tuned": tuned_metric["micro_f1"],
            "global_threshold": epoch_global_threshold,
        })
        print(
            f"Epoch {epoch}/{args.epochs}: loss={history[-1]['loss']:.4f}, "
            f"val_macro_f1@0.50={metric['macro_f1']:.4f}, "
            f"val_macro_f1@tuned={tuned_metric['macro_f1']:.4f}, "
            f"threshold={epoch_global_threshold:.2f}",
            flush=True,
        )
        # Select the checkpoint using the same threshold-aware ACD metric
        # that the search script uses to rank final trial results.
        if tuned_metric["macro_f1"] > best_macro + 1e-8:
            best_macro = tuned_metric["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(args.output, safe_serialization=True)
            tokenizer.save_pretrained(args.output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    # Reload the best epoch and tune thresholds only on Validation predictions.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(args.output).to(device)
    gold, probabilities = infer_probabilities(model, val_loader, device)
    thresholds, global_threshold, eligible = tune_thresholds(
        gold, probabilities, args.threshold_mode
    )
    # Per-label diagnostics reveal whether rare or frequent categories are missed.
    category_predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)
    category_diagnostics = {}
    for i, code in enumerate(ASPECT_CODES):
        tp = int(((gold[:, i] == 1) & (category_predictions[:, i] == 1)).sum())
        fp = int(((gold[:, i] == 0) & (category_predictions[:, i] == 1)).sum())
        fn = int(((gold[:, i] == 1) & (category_predictions[:, i] == 0)).sum())
        category_diagnostics[code] = {
            "support": int(gold[:, i].sum()),
            "predicted_positives": int(category_predictions[:, i].sum()),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    results = {
        "checkpoint": args.checkpoint,
        "best_epoch": best_epoch,
        "best_epoch_macro_f1_at_0_5": history[best_epoch - 1]["macro_f1"],
        "best_epoch_macro_f1_after_threshold_tuning": best_macro,
        "validation_after_threshold_tuning": scores(gold, probabilities, thresholds),
        "global_threshold": global_threshold,
        "threshold_mode": args.threshold_mode,
        "category_thresholds": {code: float(thresholds[i]) for i, code in enumerate(ASPECT_CODES)},
        "thresholds_eligible_categories": eligible,
        "validation_per_category": category_diagnostics,
        "label_names": {code: ASPECT_NAMES[code] for code in ASPECT_CODES},
        "training_parameters": {
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "epochs_max": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "seed": args.seed,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "history": history,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(args.workbook),
        "note": "Initial ACD development run; not nested CV or final test evaluation.",
    }
    (args.output / "acd_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Best epoch:", best_epoch)
    print("Validation Macro-F1 after threshold tuning:", results["validation_after_threshold_tuning"]["macro_f1"])
    print("Saved ACD model to:", args.output.resolve())


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune Glot500-m on ACD only")
    parser.add_argument("--workbook", type=Path,
                        default=PROJECT_ROOT / "data" / "splits" / "citizen_feedback_split.xlsx")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models" / "glot500m" / "artifacts" / "acd" / "seed42")
    parser.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    parser.add_argument("--max-length", type=int, choices=(128, 256), default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-mode", choices=("global", "eligible-category"), default="global")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="Inspect ACD data without loading Glot500-m")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
