"""CITI-Senti: full-fine-tuning, category-conditioned Glot500-m ACSA.

The training target is sentiment toward a *gold-present* category. A comment
with two annotated aspects becomes two independent training pairs. Training and
Validation only; Testing is never opened here.

Run from the repository root:
  python models/glot500m/train_acsa.py --check-data
  python models/glot500m/train_acsa.py --gradient-checkpointing \
      --output models/glot500m/artifacts/acsa/full_ft_v1

Dependencies: torch, transformers, sentencepiece, pandas, openpyxl, sklearn,
numpy, safetensors. Uses the existing shared.labels for category definitions.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.labels import ASPECT_CODES, ASPECT_NAMES  # noqa: E402

SENTIMENTS = ("Negative", "Neutral", "Positive")
SENTIMENT_TO_ID = {name.lower(): index for index, name in enumerate(SENTIMENTS)}
CANONICAL_CODES = tuple(ASPECT_CODES)


@dataclass(frozen=True)
class Annotation:
    comment_id: str
    text: str
    aspect_code: str
    label: int
    source: str
    text_group_id: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_aspect(row: dict) -> str:
    code = normalized(row.get("aspect_code")).upper()
    if not code:
        name = normalized(row.get("aspect_category"))
        found = re.match(r"^(A\d{2})\b", name, flags=re.IGNORECASE)
        code = found.group(1).upper() if found else ""
    return code


def get_source(row: dict) -> str:
    # Do not guess provenance from platforms (e.g., Facebook might be copied
    # into a synthetic derivative). Mark unverified provenance Unknown.
    for field in ("origin", "source_origin", "data_origin", "source_type"):
        value = normalized(row.get(field)).lower()
        if value in {"real", "organic", "collected"}:
            return "Real"
        if value in {"synthetic", "generated", "ai-generated", "ai_generated"}:
            return "Synthetic"
    return "Unknown"


def read_split(workbook: Path, sheet: str, on_missing: str,
               on_conflict: str) -> tuple[list[Annotation], dict]:
    """Read annotation rows, deduplicate pairs, never fabricate sentiments."""
    frame = pd.read_excel(workbook, sheet_name=sheet, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"comment_id", "model_text", "sentiment"}
    missing_columns = required.difference(frame.columns)
    if missing_columns or ("aspect_code" not in frame and "aspect_category" not in frame):
        raise ValueError(f"{sheet}: missing required annotation columns "
                         f"{sorted(missing_columns)}; also require aspect_code or aspect_category")

    by_pair: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
    text_by_id: dict[str, str] = {}
    group_by_id: dict[str, str] = {}
    counts = Counter()
    invalid_examples = []
    for number, raw_row in enumerate(frame.to_dict(orient="records"), start=2):
        row = {str(key): value for key, value in raw_row.items()}
        comment_id = normalized(row.get("comment_id"))
        text = normalized(row.get("model_text"))
        code = get_aspect(row)
        sentiment = normalized(row.get("sentiment"))
        if not comment_id or not text or code not in CANONICAL_CODES:
            invalid_examples.append(f"row {number}: missing ID/text or invalid category ({code!r})")
            counts["bad_structural_rows"] += 1
            continue
        if comment_id in text_by_id and text_by_id[comment_id] != text:
            raise ValueError(f"{sheet}: comment {comment_id} has inconsistent model_text")
        text_by_id[comment_id] = text
        group = normalized(row.get("text_group_id"))
        if comment_id in group_by_id and group_by_id[comment_id] and group and group_by_id[comment_id] != group:
            raise ValueError(f"{sheet}: comment {comment_id} has conflicting text_group_id")
        group_by_id[comment_id] = group or group_by_id.get(comment_id, "")

        sentiment_key = sentiment.lower()
        if sentiment_key not in SENTIMENT_TO_ID:
            counts["missing_or_invalid_sentiment"] += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"row {number}: {comment_id}/{code}, sentiment={sentiment!r}")
            continue
        annotation = Annotation(comment_id, text, code, SENTIMENT_TO_ID[sentiment_key],
                                get_source(row), group)
        by_pair[(comment_id, code)].append(annotation)

    if counts["bad_structural_rows"]:
        raise ValueError(f"{sheet}: {counts['bad_structural_rows']} structurally invalid rows; "
                         f"examples: {invalid_examples[:6]}")
    if counts["missing_or_invalid_sentiment"] and on_missing == "error":
        raise ValueError(f"{sheet}: {counts['missing_or_invalid_sentiment']} rows have absent/invalid "
                         f"sentiments. Examples: {invalid_examples[:8]}. "
                         "Correct annotations or pass --on-missing drop with an audit.")

    output: list[Annotation] = []
    conflicts = []
    for pair, annotations in by_pair.items():
        labels = {a.label for a in annotations}
        if len(labels) != 1:
            conflicts.append((pair, [SENTIMENTS[a.label] for a in annotations]))
            continue
        first = annotations[0]
        resolved_group = group_by_id.get(first.comment_id, first.text_group_id)
        output.append(Annotation(first.comment_id, first.text, first.aspect_code,
                                 first.label, first.source, resolved_group))
        counts["duplicate_same_label_rows"] += len(annotations) - 1
    if conflicts and on_conflict == "error":
        raise ValueError(f"{sheet}: {len(conflicts)} comment-category pairs have conflicting "
                         f"sentiments. Examples: {conflicts[:5]}. Resolve them or "
                         "pass --on-conflict drop to omit ALL conflicting pairs.")
    counts["conflicting_pairs_dropped"] = len(conflicts)
    counts["original_rows"] = len(frame)
    counts["valid_pairs"] = len(output)
    counts["unique_comments_with_pairs"] = len({item.comment_id for item in output})
    counts["sentiment_counts"] = dict(Counter(SENTIMENTS[item.label] for item in output))
    counts["aspect_counts"] = dict(Counter(item.aspect_code for item in output))
    counts["discarded_examples"] = invalid_examples[:8]
    return output, dict(counts)


def load_data(args):
    workbook = args.workbook
    if not workbook.is_file():
        raise FileNotFoundError(f"Missing split workbook: {workbook}")
    train, train_report = read_split(workbook, args.train_sheet,
                                     args.on_missing, args.on_conflict)
    val, val_report = read_split(workbook, args.val_sheet,
                                 args.on_missing, args.on_conflict)
    if not train or not val:
        raise ValueError("ACSA needs at least one valid pair in Training and Validation")
    train_ids = {r.comment_id for r in train}
    val_ids = {r.comment_id for r in val}
    shared_ids = train_ids & val_ids
    if shared_ids:
        raise ValueError(f"Comment-ID train/val leakage: {len(shared_ids)} shared IDs; "
                         f"examples={sorted(shared_ids)[:5]}")
    train_groups = {r.text_group_id for r in train if r.text_group_id}
    val_groups = {r.text_group_id for r in val if r.text_group_id}
    group_overlap = train_groups & val_groups
    if group_overlap:
        message = (f"Training/Validation share {len(group_overlap)} text_group_id values. "
                   "This may indicate text/parent leakage; inspect before publication.")
        if args.strict_groups:
            raise ValueError(message)
        print("WARNING:", message, flush=True)
    print("ACSA annotations (one row per gold-present comment/category pair):", flush=True)
    for name, rows, report in (("Training", train, train_report), ("Validation", val, val_report)):
        print(f"  {name}: {len(rows)} pairs from {report['unique_comments_with_pairs']} comments; "
              f"sentiment counts={report['sentiment_counts']}", flush=True)
        print(f"    Missing/invalid sentiment rows={report.get('missing_or_invalid_sentiment', 0)}; "
              f"identical duplicates={report.get('duplicate_same_label_rows', 0)}; "
              f"conflicting pairs dropped={report.get('conflicting_pairs_dropped', 0)}", flush=True)
    return train, val, {"Training": train_report, "Validation": val_report,
                        "text_group_id_overlap_count": len(group_overlap)}


def aspect_prompt(code: str) -> str:
    if code not in CANONICAL_CODES:
        raise ValueError(f"Invalid aspect category {code!r}; expected one of {CANONICAL_CODES}")
    return f"Aspect: {code} — {ASPECT_NAMES[code]}"


class ACSADataset(Dataset):
    def __init__(self, annotations: list[Annotation], tokenizer, max_length: int,
                 cache_dir: Path | None, checkpoint: str):
        self.rows = annotations
        digest = hashlib.sha256()
        digest.update(json.dumps([checkpoint, max_length, tokenizer.vocab_size,
                                  "paired-text-v1"], ensure_ascii=False).encode())
        for row in annotations:
            digest.update(f"{row.comment_id}\0{row.text}\0{row.aspect_code}\0".encode())
        filename = f"acsa_pairs_{digest.hexdigest()[:24]}.json.gz"
        cache_path = cache_dir / filename if cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
                self.features = json.load(stream)
            print(f"Cache hit: {cache_path}", flush=True)
        else:
            texts = [a.text for a in annotations]
            prompts = [aspect_prompt(a.aspect_code) for a in annotations]
            # 'only_first' prevents a long citizen comment from truncating the
            # category name; tokenizer constructs the appropriate pair markers.
            tokens = tokenizer(texts, text_pair=prompts, max_length=max_length,
                               truncation="only_first", padding=False)
            self.features = [
                {key: values[i] for key, values in tokens.items()}
                for i in range(len(annotations))
            ]
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(cache_path, "wt", encoding="utf-8") as stream:
                    json.dump(self.features, stream)
                print(f"Saved token cache: {cache_path}", flush=True)
        if len(self.features) != len(annotations):
            raise RuntimeError("Token cache length mismatch; remove cache and retry")
        self.lengths = np.array([len(x["input_ids"]) for x in self.features])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {**self.features[index], "labels": self.rows[index].label}


def make_collator(tokenizer):
    def collate(batch):
        labels = torch.tensor([row["labels"] for row in batch], dtype=torch.long)
        features = [{key: value for key, value in row.items() if key != "labels"}
                    for row in batch]
        padded = tokenizer.pad(features, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return collate


def class_weights(train: list[Annotation], mode: str, cap: float) -> torch.Tensor:
    totals = np.bincount([item.label for item in train], minlength=3).astype(np.float64)
    if np.any(totals == 0):
        print("WARNING: a sentiment class is missing from Training; the classifier "
              "cannot learn it reliably.", flush=True)
    weights = np.ones(3, dtype=np.float64)
    present = totals > 0
    if mode == "sqrt":
        weights[present] = np.sqrt(len(train) / (3.0 * totals[present]))
    elif mode == "inverse":
        weights[present] = len(train) / (3.0 * totals[present])
    elif mode != "none":
        raise ValueError(f"Unrecognized class weighting {mode}")
    # Relative balancing; classes commoner than average may receive < 1 weight.
    if mode != "none":
        weights = np.clip(weights, 1.0 / cap, cap)
    # Normalize the expected *training-dataset* weight to 1, rather than
    # normalizing within each microbatch. With batch_size=1, PyTorch's default
    # weighted CE reduction="mean" cancels the class weight completely!
    expected_weight = float(np.dot(weights, totals) / max(1.0, totals.sum()))
    weights = weights / expected_weight
    return torch.tensor(weights, dtype=torch.float32)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_precision(mode: str, device: torch.device):
    if device.type != "cuda":
        if mode not in {"auto", "fp32"}:
            raise ValueError("bf16/fp16 autocast in this script requires CUDA")
        return None
    if mode == "fp32":
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_optimizer(model, args):
    """Factorized Adafactor uses far less optimizer memory than AdamW."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = []
    for head in (False, True):
        for decay in (False, True):
            params = [
                p for name, p in model.named_parameters()
                if p.requires_grad
                and name.startswith("classifier.") == head
                and (not any(tag in name for tag in no_decay)) == decay
            ]
            if params:
                groups.append({"params": params,
                               "lr": args.learning_rate * (args.head_lr_multiplier if head else 1),
                               "weight_decay": args.weight_decay if decay else 0.0})
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(groups, lr=args.learning_rate,
                         relative_step=False, scale_parameter=False,
                         warmup_init=False, clip_threshold=1.0)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.learning_rate, foreach=False)
    if args.optimizer == "adamw8bit":
        if not torch.cuda.is_available():
            raise ValueError("adamw8bit requires a CUDA GPU")
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("Install compatible bitsandbytes to use adamw8bit") from exc
        return bnb.optim.AdamW8bit(groups, lr=args.learning_rate)
    raise ValueError(f"Unknown optimizer {args.optimizer}")


def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(round(total_steps * warmup_ratio))
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def calculate_metrics(gold: np.ndarray, predictions: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predictions, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(accuracy_score(gold, predictions)),
        "micro_f1": float(f1_score_micro_from_confusion(gold, predictions)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(SENTIMENTS)
        },
        "confusion_matrix": confusion_matrix(gold, predictions, labels=[0, 1, 2]).tolist(),
    }


def f1_score_micro_from_confusion(gold, predictions) -> float:
    # For a single-label 3-class task, micro-F1 equals accuracy.
    return float(np.mean(gold == predictions))


@torch.inference_mode()
def validate(model, loader, device, weights, precision):
    model.eval()
    gold, probabilities = [], []
    total_loss, total_count = 0.0, 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {name: value.to(device) for name, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=precision,
                            enabled=(precision is not None)):
            logits = model(**inputs).logits
            per_example_loss = F.cross_entropy(logits.float(), labels, reduction="none")
            loss = (per_example_loss * weights[labels]).mean()
        gold.append(labels.cpu().numpy())
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        total_loss += float(loss.item()) * len(labels)
        total_count += len(labels)
    y = np.concatenate(gold)
    probs = np.concatenate(probabilities)
    predictions = np.argmax(probs, axis=1)
    metrics = calculate_metrics(y, predictions)
    metrics["loss"] = total_loss / max(1, total_count)
    return metrics, y, probs


def record_details(rows: list[Annotation], y: np.ndarray, probs: np.ndarray) -> dict:
    predicted = probs.argmax(axis=1)
    per_aspect = {}
    for code in CANONICAL_CODES:
        indices = np.array([i for i, row in enumerate(rows) if row.aspect_code == code], dtype=int)
        if len(indices) == 0:
            per_aspect[code] = {"support_pairs": 0, "macro_f1": None, "accuracy": None}
        else:
            vals = calculate_metrics(y[indices], predicted[indices])
            per_aspect[code] = {"support_pairs": int(len(indices)),
                                "macro_f1": vals["macro_f1"], "accuracy": vals["accuracy"]}
    sources = {}
    for source in ("Real", "Synthetic", "Unknown"):
        indices = np.array([i for i, row in enumerate(rows) if row.source == source], dtype=int)
        if len(indices):
            vals = calculate_metrics(y[indices], predicted[indices])
            sources[source] = {"pairs": len(indices), "macro_f1": vals["macro_f1"],
                               "accuracy": vals["accuracy"]}
    return {"per_aspect": per_aspect, "by_source": sources}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args):
    if not 0 < args.warmup_ratio < 1 or args.max_length < 48:
        raise ValueError("warmup-ratio must be between 0 and 1; max-length >= 48")
    if min(args.epochs, args.batch_size, args.gradient_accumulation,
           args.validation_batch_size, args.patience) < 1:
        raise ValueError("epochs, batches, accumulation, and patience must be positive")
    if args.class_weight_cap < 1 or args.head_lr_multiplier <= 0:
        raise ValueError("class-weight-cap >= 1; head-lr-multiplier > 0 required")
    train_rows, val_rows, data_report = load_data(args)
    if args.check_data:
        print("Data check finished. Review any dropped-label counts and duplicate-group warnings.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "acsa_results.json").is_file() and not args.overwrite:
        raise FileExistsError(f"ACSA results already exist in {args.output}; use another --output")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(args.seed)
    device = select_device()
    precision = choose_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Full fine-tuning: model={args.checkpoint}; device={device}; "
          f"precision={str(precision)}; optimizer={args.optimizer}", flush=True)
    print("No LoRA, no frozen encoder layers.", flush=True)
    if args.optimizer == "adamw" and device.type == "cuda":
        print("WARNING: full AdamW may exceed 6 GiB VRAM; use --optimizer adafactor "
              "or optional adamw8bit.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    train_ds = ACSADataset(train_rows, tokenizer, args.max_length, args.cache_dir, args.checkpoint)
    val_ds = ACSADataset(val_rows, tokenizer, args.max_length, args.cache_dir, args.checkpoint)
    print(f"Token lengths: train median={np.median(train_ds.lengths):.0f}, "
          f"p95={np.percentile(train_ds.lengths, 95):.0f}, "
          f"at_limit={(train_ds.lengths == args.max_length).sum()}")
    collate = make_collator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.validation_batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate,
                            pin_memory=(device.type == "cuda"))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=3, problem_type="single_label_classification",
        id2label={i: name for i, name in enumerate(SENTIMENTS)},
        label2id={name: i for i, name in enumerate(SENTIMENTS)},
    )
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == trainable, "Full fine-tuning unexpectedly froze model parameters"
    print(f"Trainable parameters: {trainable:,}/{total:,}", flush=True)
    optimizer = make_optimizer(model, args)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    scheduler = make_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_ratio)
    weights = class_weights(train_rows, args.class_weighting, args.class_weight_cap).to(device)
    print("Sentiment class weights:", {name: round(float(w), 3)
                                        for name, w in zip(SENTIMENTS, weights.cpu().tolist())})
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == torch.float16))
    history = []
    best_macro = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        cumulative_loss, batches = 0.0, 0
        for index, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            group_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            with torch.autocast(device_type="cuda", dtype=precision,
                                enabled=(precision is not None)):
                logits = model(**inputs).logits
                per_example_loss = F.cross_entropy(logits.float(), labels, reduction="none")
                # Fixed train-derived weights: effective with batch-size 1.
                loss = (per_example_loss * weights[labels]).mean()
            scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
            # Avoid a CPU/GPU sync every microbatch. Log occasionally instead.
            if args.log_every and (index + 1) % args.log_every == 0:
                print(f"  Epoch {epoch}: batches {index+1}/{len(train_loader)} "
                      f"| loss={loss.item():.4f}", flush=True)
            if ((index + 1) % args.gradient_accumulation == 0
                    or index + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batches += 1
        metrics, _, _ = validate(model, val_loader, device, weights, precision)
        history.append({"epoch": epoch, "validation_macro_f1": metrics["macro_f1"],
                        "validation_accuracy": metrics["accuracy"],
                        "validation_loss": metrics["loss"]})
        print(f"Epoch {epoch}/{args.epochs}: val_macro_f1={metrics['macro_f1']:.4f} "
              f"accuracy={metrics['accuracy']:.4f} val_loss={metrics['loss']:.4f}", flush=True)
        if (metrics["macro_f1"] > best_macro + 1e-6 or
            (abs(metrics["macro_f1"] - best_macro) <= 1e-6 and
             metrics["loss"] < best_loss - 1e-6)):
            best_macro = metrics["macro_f1"]
            best_loss = metrics["loss"]
            best_epoch = epoch
            stale = 0
            model.save_pretrained(args.output, safe_serialization=True)
            tokenizer.save_pretrained(args.output)
            print(f"  Saved new best checkpoint: {args.output}", flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no improvement in {args.patience} epochs.")
                break

    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output).to(device)
    final, y, probs = validate(best_model, val_loader, device, weights, precision)
    details = record_details(val_rows, y, probs)

    prediction_path = args.output / "validation_predictions.csv"
    pd.DataFrame({
        "comment_id": [a.comment_id for a in val_rows],
        "aspect_code": [a.aspect_code for a in val_rows],
        "source": [a.source for a in val_rows],
        "gold_sentiment": [SENTIMENTS[int(k)] for k in y],
        "predicted_sentiment": [SENTIMENTS[int(k)] for k in probs.argmax(axis=1)],
        **{f"p_{name.lower()}": probs[:, i] for i, name in enumerate(SENTIMENTS)},
    }).to_csv(prediction_path, index=False)
    pd.DataFrame([
        {"aspect_code": code, "aspect_name": ASPECT_NAMES[code], **stats}
        for code, stats in details["per_aspect"].items()
    ]).to_csv(args.output / "validation_per_category.csv", index=False)
    pd.DataFrame([
        {"sentiment": label, **stats}
        for label, stats in final["per_class"].items()
    ]).to_csv(args.output / "validation_per_sentiment.csv", index=False)
    cm = final["confusion_matrix"]
    pd.DataFrame(cm, index=[f"gold_{name}" for name in SENTIMENTS],
                 columns=[f"pred_{name}" for name in SENTIMENTS]).to_csv(
                     args.output / "validation_confusion_matrix.csv")
    report = {
        "task": "Conditional ACSA on gold-present comment-category pairs",
        "training_mode": "full_fine_tuning_no_lora",
        "checkpoint": args.checkpoint,
        "input_format": "tokenizer(text=model_text, text_pair='Aspect: CODE — NAME')",
        "label_order": list(SENTIMENTS),
        "aspect_codes": list(CANONICAL_CODES),
        "aspect_names": {code: ASPECT_NAMES[code] for code in CANONICAL_CODES},
        "best_epoch": best_epoch,
        "validation": final,
        "validation_per_aspect": details["per_aspect"],
        "validation_by_source": details["by_source"],
        "data_audit": data_report,
        "history": history,
        "training_parameters": {
            "learning_rate": args.learning_rate,
            "head_lr_multiplier": args.head_lr_multiplier,
            "optimizer": args.optimizer,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "max_length": args.max_length,
            "precision": args.precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "class_weight_cap": args.class_weight_cap,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
        },
        "data_source": str(args.workbook),
        "data_fingerprint": sha256_file(args.workbook),
        "trainer_fingerprint": sha256_file(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Development Validation is used for checkpoint selection; not independent "
                 "Test performance. Synthetic-parent leakage cannot be verified by IDs alone. "
                 "Source-specific metrics only reflect explicit provenance metadata."),
    }
    (args.output / "acsa_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Best epoch={best_epoch}; best val Macro-F1={final['macro_f1']:.4f}; "
          f"accuracy={final['accuracy']:.4f}")
    print(f"Saved checkpoint, metrics and CSV outputs to: {args.output.resolve()}")
    if device.type == "cuda":
        print(f"Peak allocated CUDA memory: "
              f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")


def build_parser():
    parser = argparse.ArgumentParser(description="Glot500-m full fine-tuning for conditional ACSA")
    parser.add_argument("--workbook", type=Path,
                        default=PROJECT_ROOT / "data/splits/citizen_feedback_split.xlsx")
    parser.add_argument("--train-sheet", default="Training")
    parser.add_argument("--val-sheet", default="Validation")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models/glot500m/artifacts/acsa/full_ft_v1")
    parser.add_argument("--checkpoint", default="cis-lmu/glot500-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-lr-multiplier", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, choices=[128, 192, 256], default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--optimizer", choices=["adafactor", "adamw", "adamw8bit"], default="adafactor")
    parser.add_argument("--class-weighting", choices=["none", "sqrt", "inverse"], default="sqrt")
    parser.add_argument("--class-weight-cap", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--on-missing", choices=["drop", "error"], default="drop")
    parser.add_argument("--on-conflict", choices=["drop", "error"], default="error")
    parser.add_argument("--strict-groups", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run_training(build_parser().parse_args())
