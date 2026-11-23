"""Glot500-m multilabel ACD: targeted LoRA and source-balanced experiments.

Compatibility: 11 categories, shared.data Training/Validation, RTX 4050 6 GB.
New options are controlled experiments, NOT guaranteed accuracy improvements.
No Test labels are loaded; no split changes are made by this trainer.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data import TRAIN_SHEET, VAL_SHEET, make_aspect_task, read_annotation_sheet
from shared.labels import ASPECT_CODES, ASPECT_NAMES


class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, sample_weights=None):
        encoded = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
        self.features = [{k: encoded[k][i] for k in encoded} for i in range(len(texts))]
        self.labels = np.asarray(labels, dtype=np.float32)
        if sample_weights is None:
            sample_weights = np.ones(len(self.labels), dtype=np.float32)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if len(self.sample_weights) != len(self.labels):
            raise ValueError("Each Training record needs exactly one sample weight")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**self.features[index], "labels": self.labels[index],
                "sample_weight": self.sample_weights[index]}


def collate_factory(tokenizer):
    def collate(items):
        input_items = [{k: v for k, v in item.items() if k not in ("labels", "sample_weight")}
                       for item in items]
        batch = tokenizer.pad(input_items, padding=True, return_tensors="pt")
        batch["labels"] = torch.from_numpy(np.stack([i["labels"] for i in items])).float()
        batch["sample_weight"] = torch.tensor([float(i["sample_weight"]) for i in items],
                                              dtype=torch.float32)
        return batch
    return collate


def load_data(workbook):
    train = make_aspect_task(read_annotation_sheet(workbook, TRAIN_SHEET))
    val = make_aspect_task(read_annotation_sheet(workbook, VAL_SHEET))
    train_ids = [str(r["comment_id"]) for r in train.row_metadata]
    val_ids = [str(r["comment_id"]) for r in val.row_metadata]
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("Duplicate comment_id within a partition")
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and Validation share comment IDs")
    if train.labels.shape[1] != len(ASPECT_CODES) or val.labels.shape[1] != len(ASPECT_CODES):
        raise ValueError("Expected 11 ACD category indicators")
    return train, val



def read_sources_from_sheet(workbook: Path, sheet_name: str) -> dict[str, str]:
    """Look up source_type by comment ID, avoiding assumptions about task metadata.

    The workbook contains one row per aspect; labels are reconciled by comment_id.
    Missing source column yields {} so unweighted training still works.
    """
    from openpyxl import load_workbook
    wb = load_workbook(workbook, read_only=True, data_only=True)
    try:
        if sheet_name not in wb:
            return {}
        rows = wb[sheet_name].iter_rows(values_only=True)
        header = next(rows, ())
        names = {str(v).strip().lower(): i for i, v in enumerate(header) if v is not None}
        source_col = next((names[n] for n in ("source_type", "data_origin", "origin")
                           if n in names), None)
        if "comment_id" not in names or source_col is None:
            return {}
        id_col = names["comment_id"]
        sources = {}
        for row in rows:
            if len(row) <= max(id_col, source_col) or row[id_col] is None:
                continue
            comment_id = str(row[id_col]).strip()
            value = str(row[source_col]).strip() if row[source_col] is not None else ""
            if not value:
                continue
            if comment_id in sources and sources[comment_id].casefold() != value.casefold():
                raise ValueError(f"Conflicting source types for {comment_id} in {sheet_name}")
            sources[comment_id] = value
        return sources
    finally:
        wb.close()


def source_group(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"synthetic", "generated", "ai-generated", "ai generated"}:
        return "Synthetic"
    if v in {"facebook", "tiktok", "reddit", "threads", "real", "genuine", "human"}:
        return "Real"
    return "Unknown"


def source_information(task, lookup: dict[str, str]) -> list[str]:
    result = []
    for row in task.row_metadata:
        source = lookup.get(str(row["comment_id"]).strip())
        if source is None:
            source = row.get("source_type", row.get("origin", ""))
        result.append(source_group(source))
    return result


def normalized_source_weights(groups, real_weight=1.0, synthetic_weight=1.0):
    """Normalize once across TRAINING, never independently per microbatch.

    This matters when microbatch=1: dividing by batch-weight sum would cancel
    any example weighting and make the feature ineffective.
    """
    if real_weight <= 0 or synthetic_weight <= 0:
        raise ValueError("Real and synthetic weights must be positive")
    weights = np.array([real_weight if g == "Real" else synthetic_weight if g == "Synthetic"
                        else 1.0 for g in groups], dtype=np.float32)
    if weights.size == 0 or float(weights.mean()) <= 0:
        raise ValueError("Training weights cannot be empty")
    return weights / weights.mean()


def by_source_metrics(gold, probs, thresholds, groups):
    result = {}
    for group in ("Real", "Synthetic", "Unknown"):
        indices = np.asarray([g == group for g in groups], dtype=bool)
        if indices.any():
            subset_gold, subset_prob = gold[indices], probs[indices]
            result[group] = {"comments": int(indices.sum()),
                             "gold_category_instances": int(subset_gold.sum()),
                             **scores(subset_gold, subset_prob, thresholds)}
    return result

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scores(y, probs, thresholds):
    pred = (probs >= np.asarray(thresholds).reshape(1, -1)).astype(np.int8)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "exact_set_match": float(np.mean(np.all(pred == y, axis=1))),
    }


THRESHOLD_GRID = np.round(np.arange(0.05, 0.501, 0.01), 2)


def tune_thresholds(y, probs, mode):
    """Optimize validation Macro-F1 in [0.05, 0.50] at 0.01 resolution."""
    grid = THRESHOLD_GRID
    options = [
        (scores(y, probs, np.full(len(ASPECT_CODES), t))["macro_f1"],
         -abs(float(t) - 0.5), float(t))
        for t in grid
    ]
    best_global = max(options)[2]
    thresholds = np.full(len(ASPECT_CODES), best_global, dtype=float)
    eligible = []
    if mode == "eligible-category":
        for i, code in enumerate(ASPECT_CODES):
            positives = int(y[:, i].sum())
            negatives = int(len(y) - positives)
            if positives < 20 or negatives < 20:
                continue
            eligible.append(code)
            choices = [
                (float(f1_score(y[:, i], probs[:, i] >= t, zero_division=0)),
                 -abs(float(t) - best_global), float(t))
                for t in grid
            ]
            thresholds[i] = max(choices)[2]
    return thresholds, best_global, eligible


def validation_diagnostics(y, probs):
    """Descriptive diagnostics, reported independently from threshold optimization."""
    y = np.asarray(y, dtype=np.int8)
    probs = np.asarray(probs, dtype=np.float64)
    pos = probs[y == 1]
    neg = probs[y == 0]
    diagnostic_rows = []
    for t in (0.10, 0.20, 0.30, 0.50):
        pred = probs >= t
        tp = int(np.logical_and(pred, y == 1).sum())
        predicted = int(pred.sum())
        actual = int(y.sum())
        diagnostic_rows.append({
            "threshold": t,
            "predicted_positives": predicted,
            "true_positives": tp,
            "false_positives": predicted - tp,
            "false_negatives": actual - tp,
            "micro_precision": tp / predicted if predicted else 0.0,
            "micro_recall": tp / actual if actual else 0.0,
            "macro_f1": scores(y, probs, np.full(probs.shape[1], t))["macro_f1"],
        })
    return {
        "actual_positives": int(y.sum()),
        "mean_probability_gold_positive": float(pos.mean()) if pos.size else None,
        "mean_probability_gold_negative": float(neg.mean()) if neg.size else None,
        "by_threshold": diagnostic_rows,
    }


def print_validation_diagnostics(diag):
    print("  Validation prediction diagnostics (0.10 is also eligible for selection):", flush=True)
    for row in diag["by_threshold"]:
        print(
            f"    t={row['threshold']:.2f} | predicted={row['predicted_positives']} | "
            f"TP={row['true_positives']} | FP={row['false_positives']} | "
            f"precision={row['micro_precision']:.3f} | recall={row['micro_recall']:.3f} | "
            f"Macro-F1={row['macro_f1']:.4f}", flush=True,
        )
    p, n = diag["mean_probability_gold_positive"], diag["mean_probability_gold_negative"]
    print(f"    mean score (present): {p:.4f}" if p is not None else "    mean score (present): n/a", flush=True)
    print(f"    mean score (absent):  {n:.4f}" if n is not None else "    mean score (absent):  n/a", flush=True)


def per_category_metrics(y, probs, thresholds):
    pred = (probs >= thresholds.reshape(1, -1)).astype(np.int8)
    result = {}
    for i, code in enumerate(ASPECT_CODES):
        tp = int(((y[:, i] == 1) & (pred[:, i] == 1)).sum())
        fp = int(((y[:, i] == 0) & (pred[:, i] == 1)).sum())
        fn = int(((y[:, i] == 1) & (pred[:, i] == 0)).sum())
        result[code] = {
            "support": int(y[:, i].sum()),
            "predicted_positives": int(pred[:, i].sum()),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "average_precision": (float(average_precision_score(y[:, i], probs[:, i]))
                                  if 0 < int(y[:, i].sum()) < len(y) else None),
        }
    return result


def export_validation_files(output, task, gold, probs, metrics, source_groups):
    """Keep gold labels and probabilities to inspect decisions without retraining."""
    with (output / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["comment_id", "source_group", *[f"gold_{c}" for c in ASPECT_CODES],
                         *[f"prob_{c}" for c in ASPECT_CODES]])
        for index, record in enumerate(task.row_metadata):
            writer.writerow([record["comment_id"], source_groups[index],
                             *gold[index].astype(int).tolist(),
                             *[f"{x:.8f}" for x in probs[index]]])
    with (output / "validation_per_category.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = ["category", "category_name", "support", "predicted_positives",
                  "true_positives", "false_positives", "false_negatives",
                  "precision", "recall", "f1", "average_precision"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for code in ASPECT_CODES:
            writer.writerow({"category": code, "category_name": ASPECT_NAMES[code], **metrics[code]})


def export_source_metrics(output, results):
    with (output / "validation_by_source.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = ["group", "comments", "gold_category_instances", "macro_f1", "micro_f1",
                  "exact_set_match"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for group, metrics in results.items():
            writer.writerow({"group": group, **metrics})


def precision_policy(device, requested):
    if device.type != "cuda":
        return torch.float32, False, False, "fp32"
    if requested in {"auto", "bf16"} and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False, "bf16"
    if requested == "bf16":
        raise RuntimeError("GPU does not support BF16; use --precision fp16")
    if requested in {"auto", "fp16"}:
        return torch.float16, True, True, "fp16"
    return torch.float32, False, False, "fp32"


@torch.inference_mode()
def infer_probabilities(model, loader, device, use_amp, amp_dtype):
    model.eval()
    outputs, labels = [], []
    for batch in loader:
        labels.append(batch["labels"].numpy().astype(np.int8))
        features = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                    if k not in ("labels", "sample_weight")}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(**features).logits
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(labels), np.concatenate(outputs)


def make_base_model(checkpoint, dtype):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        num_labels=len(ASPECT_CODES),
        problem_type="multi_label_classification",
        id2label={i: code for i, code in enumerate(ASPECT_CODES)},
        label2id={code: i for i, code in enumerate(ASPECT_CODES)},
    )
    # Preserve FP32 classifier weights while keeping the huge frozen encoder
    # low precision. This applies also when RELOADING saved classifier weights.
    if hasattr(model, "classifier"):
        model.classifier.float()
    return model


def train(args):
    if min(args.batch_size, args.gradient_accumulation, args.epochs, args.log_every) < 1:
        raise ValueError("Batch size, accumulation, epochs and log interval must be positive")
    train_task, val_task = load_data(args.workbook)
    train_sources = source_information(
        train_task, read_sources_from_sheet(args.workbook, TRAIN_SHEET))
    val_sources = source_information(
        val_task, read_sources_from_sheet(args.workbook, VAL_SHEET))
    from collections import Counter
    print(f"ACD records: train={len(train_task.texts)}, validation={len(val_task.texts)}", flush=True)
    print(f"Source groups (Training): {dict(Counter(train_sources))}", flush=True)
    print(f"Source groups (Validation): {dict(Counter(val_sources))}", flush=True)
    if (args.real_weight != 1.0 or args.synthetic_weight != 1.0):
        if "Unknown" in train_sources:
            raise ValueError("Source weighting requested but Training has unknown/missing "
                             "source_type values. Fix the workbook or set weights to 1.")
        if not ("Real" in train_sources and "Synthetic" in train_sources):
            raise ValueError("Source weighting requires both Real and Synthetic Training records")
    sample_weights = normalized_source_weights(train_sources,
                                                args.real_weight, args.synthetic_weight)
    print("Label support (train / validation):")
    for i, code in enumerate(ASPECT_CODES):
        print(f"  {code}: {int(train_task.labels[:, i].sum())} / {int(val_task.labels[:, i].sum())}")
    if args.check_data:
        print("Data check passed. Training and Validation are disjoint by comment ID.")
        return

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install PEFT: python -m pip install 'peft>=0.13,<1'") from exc
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype, use_amp, use_scaler, precision_name = precision_policy(device, args.precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    print(f"Device: {device} | model: {args.checkpoint} | LoRA | precision: {precision_name}", flush=True)
    if device.type == "cpu":
        print("Warning: Glot500-m training on CPU will be slow.")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    model = make_base_model(args.checkpoint, dtype)
    if args.gradient_checkpointing:
        # Non-reentrant mode works better with frozen base-model parameters.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=(["query", "value"] if args.lora_targets == "qv" else "all-linear"),
        modules_to_save=["classifier"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    if args.trainable_fp32:
        # FP32 master weights for LoRA + classifier prevent BF16 optimizer updates.
        # The frozen 395M-parameter backbone stays memory efficient.
        for param in model.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()
    model.print_trainable_parameters()
    print(f"LoRA targets: {args.lora_targets}; trainable weights FP32: {args.trainable_fp32}", flush=True)

    train_ds = AspectDataset(train_task.texts, train_task.labels, tokenizer,
                             args.max_length, sample_weights=sample_weights)
    val_ds = AspectDataset(val_task.texts, val_task.labels, tokenizer, args.max_length)
    collate = collate_factory(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
        collate_fn=collate, pin_memory=device.type == "cuda",
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(total_steps * args.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    pos_weight = None
    if args.weighted_bce:
        pos = np.asarray(train_task.labels.sum(axis=0), dtype=float)
        neg = len(train_task.labels) - pos
        weight = np.sqrt(neg / np.maximum(pos, 1.0))
        weight = np.clip(weight, 1.0, args.max_pos_weight)
        weight[pos == 0] = 1.0
        pos_weight = torch.tensor(weight, dtype=torch.float32, device=device)
        print("Weighted BCE positive weights:", dict(zip(ASPECT_CODES, np.round(weight, 2))))

    fixed_threshold = np.full(len(ASPECT_CODES), 0.5)
    if args.output.exists() and (args.output / "acd_results.json").exists():
        raise FileExistsError(
            f"Completed run exists in {args.output}. Use a new --output to preserve it."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    best_macro, best_epoch, stale_epochs = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = torch.zeros((), device=device)
        t0 = time.perf_counter()
        for index, batch in enumerate(train_loader):
            labels = batch["labels"].to(device, non_blocking=True)
            batch_weights = batch["sample_weight"].to(device, non_blocking=True)
            features = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                        if k not in ("labels", "sample_weight")}
            group_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            current_group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                logits = model(**features).logits
                # Keep BCE computation in float32 for numerical stability.
                per_label_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), labels, pos_weight=pos_weight, reduction="none"
                )
                # Per-example weights preserve impact even with microbatch=1.
                # Global train-set normalization maintains typical gradient scale.
                loss = (per_label_loss.mean(dim=1) * batch_weights).mean()
            epoch_loss += loss.detach()
            scaler.scale(loss / current_group_size).backward()
            is_step = ((index + 1) % args.gradient_accumulation == 0 or
                       index + 1 == len(train_loader))
            if is_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % args.log_every == 0 or index + 1 == len(train_loader):
                elapsed = max(0.001, time.perf_counter() - t0)
                print(f"  epoch {epoch}/{args.epochs}: {index+1}/{len(train_loader)} batches "
                      f"({(index+1)/elapsed:.2f} batches/s)", flush=True)

        gold, probs = infer_probabilities(model, val_loader, device, use_amp, dtype)
        fixed_metric = scores(gold, probs, fixed_threshold)
        thresholds, best_global, _ = tune_thresholds(gold, probs, args.threshold_mode)
        tuned_metric = scores(gold, probs, thresholds)
        diag = validation_diagnostics(gold, probs)
        history.append({
            "epoch": epoch,
            "loss": round((epoch_loss / len(train_loader)).item(), 5),
            **fixed_metric,
            "macro_f1_threshold_tuned": tuned_metric["macro_f1"],
            "micro_f1_threshold_tuned": tuned_metric["micro_f1"],
            "global_threshold": best_global,
            "epoch_seconds": round(time.perf_counter() - t0, 1),
            "prediction_diagnostics": diag,
        })
        print(f"Epoch {epoch}/{args.epochs}: loss={history[-1]['loss']:.4f}, "
              f"val_macro_f1@0.50={fixed_metric['macro_f1']:.4f}, "
              f"val_macro_f1@tuned={tuned_metric['macro_f1']:.4f}, "
              f"threshold={best_global:.2f}, "
              f"seconds={history[-1]['epoch_seconds']}", flush=True)
        print_validation_diagnostics(diag)
        if tuned_metric["macro_f1"] > best_macro + 1e-8:
            best_macro, best_epoch, stale_epochs = tuned_metric["macro_f1"], epoch, 0
            # PEFT saves only the adapters and trainable classification head.
            model.save_pretrained(str(args.output), safe_serialization=True)
            tokenizer.save_pretrained(str(args.output))
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    peak_vram_gib = (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                     if device.type == "cuda" else None)
    # IMPORTANT: AdamW holds references to model weights; release it before reloading.
    del optimizer, scheduler, scaler, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base = make_base_model(args.checkpoint, dtype)
    model = PeftModel.from_pretrained(base, str(args.output)).to(device)
    gold, probs = infer_probabilities(model, val_loader, device, use_amp, dtype)
    thresholds, global_threshold, eligible = tune_thresholds(gold, probs, args.threshold_mode)
    # Catch an adapter/classifier reload problem instead of accepting a corrupt checkpoint.
    reloaded_metric = scores(gold, probs, thresholds)
    if abs(reloaded_metric["macro_f1"] - best_macro) > 2e-4:
        raise RuntimeError(
            "Reloaded PEFT adapter does not reproduce checkpoint-selection Macro-F1: "
            f"before={best_macro:.6f}, after={reloaded_metric['macro_f1']:.6f}. "
            "Check that the saved adapter includes the classifier head."
        )
    per_category = per_category_metrics(gold, probs, thresholds)
    best_diag = validation_diagnostics(gold, probs)
    print("Best-checkpoint diagnostics:", flush=True)
    print_validation_diagnostics(best_diag)
    export_validation_files(args.output, val_task, gold, probs, per_category, val_sources)
    source_results = by_source_metrics(gold, probs, thresholds, val_sources)
    export_source_metrics(args.output, source_results)
    print("Validation Macro-F1 by source:", source_results, flush=True)
    results = {
        "checkpoint": args.checkpoint,
        "training_mode": "lora",
        "checkpoint_format": "peft_adapter_plus_classifier",
        "best_epoch": best_epoch,
        "best_epoch_macro_f1_at_0_5": history[best_epoch - 1]["macro_f1"],
        "best_epoch_macro_f1_after_threshold_tuning": best_macro,
        "validation_after_threshold_tuning": scores(gold, probs, thresholds),
        "global_threshold": global_threshold,
        "threshold_mode": args.threshold_mode,
        "threshold_grid": [float(x) for x in THRESHOLD_GRID],
        "validation_prediction_diagnostics": best_diag,
        "run_signature": args.run_signature,
        "category_thresholds": {c: float(thresholds[i]) for i, c in enumerate(ASPECT_CODES)},
        "thresholds_eligible_categories": eligible,
        "validation_per_category": per_category,
        "validation_by_source": source_results,
        "training_source_counts": dict(Counter(train_sources)),
        "validation_source_counts": dict(Counter(val_sources)),
        "label_names": {c: ASPECT_NAMES[c] for c in ASPECT_CODES},
        "training_parameters": {
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "epochs_max": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length, "seed": args.seed,
            "gradient_checkpointing": args.gradient_checkpointing,
            "precision": precision_name,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
            "trainable_fp32": args.trainable_fp32,
            "real_weight": args.real_weight,
            "synthetic_weight": args.synthetic_weight,
            "weighted_bce": args.weighted_bce,
            "max_pos_weight": args.max_pos_weight if args.weighted_bce else None,
        },
        "peak_gpu_memory_allocated_gib": peak_vram_gib,
        "history": history,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(args.workbook),
        "note": "LoRA ACD fixed-split development. Source weighting changes loss only; not "
                "data partitions. Validation is used for checkpoint and threshold selection; "
                "not independent test performance. Synthetic parent leakage not checked.",
    }
    (args.output / "acd_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Best epoch:", best_epoch)
    print("Validation Macro-F1 after threshold tuning:", results["validation_after_threshold_tuning"]["macro_f1"])
    print("Peak allocated GPU memory (GiB):", peak_vram_gib)
    print("Saved LoRA ACD model to:", args.output.resolve())


def build_parser():
    p = argparse.ArgumentParser(description="Memory-efficient Glot500-m LoRA ACD fine-tuning")
    p.add_argument("--workbook", type=Path, default=PROJECT_ROOT / "data" / "splits" / "citizen_feedback_split.xlsx")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "glot500m" / "artifacts" / "acd" / "lora_rtx4050")
    p.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    p.add_argument("--max-length", type=int, choices=(96, 128, 256), default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-mode", choices=("global", "eligible-category"), default="global")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--lora-rank", type=int, choices=(4, 8, 16, 32), default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.10)
    p.add_argument("--lora-targets", choices=("qv", "all-linear"), default="qv",
                   help="LoRA on query/value only or all linear layers (stronger, more VRAM)")
    p.add_argument("--trainable-fp32", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep adapter/classifier trainable weights in FP32")
    p.add_argument("--real-weight", type=float, default=1.0,
                   help="Loss weight of real Training comments; 1.0 disables upweighting")
    p.add_argument("--synthetic-weight", type=float, default=1.0,
                   help="Loss weight of synthetic Training comments")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weighted-bce", action="store_true", help="Optional class-imbalance experiment, not enabled by default")
    p.add_argument("--max-pos-weight", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--run-signature", default=None, help=argparse.SUPPRESS)
    p.add_argument("--check-data", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
