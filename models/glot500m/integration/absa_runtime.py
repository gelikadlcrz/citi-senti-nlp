"""Shared, read-only CITI-Senti ACD -> ACSA inference runtime.

Supports either a full-fine-tuned ACD checkpoint or a PEFT LoRA ACD adapter,
and the V1 full-fine-tuned, comment-first ACSA checkpoint. Does not train,
calibrate thresholds, or write files.

Place alongside predict_absa_pipeline.py and evaluate_absa_pipeline.py, e.g.
models/glot500m/integration/. Commands may be run from the repository root.
"""
from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def locate_project_root() -> Path:
    for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (parent / "shared" / "labels.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise FileNotFoundError(
        "Cannot find project root with shared/labels.py. Install scripts inside "
        "your citi-senti-nlp repository, e.g. models/glot500m/integration/."
    )


PROJECT_ROOT = locate_project_root()
from shared.labels import ASPECT_CODES, ASPECT_NAMES  # noqa: E402

CODES = tuple(ASPECT_CODES)
SENTIMENTS = ("Negative", "Neutral", "Positive")
DEFAULT_WORKBOOK = PROJECT_ROOT / "data" / "splits" / "citizen_feedback_split.xlsx"
WEIGHT_FILENAMES = (
    "model.safetensors", "pytorch_model.bin",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
)
TOKENIZER_FILENAMES = (
    "tokenizer.json", "tokenizer.model", "spiece.model",
    "sentencepiece.bpe.model", "vocab.json",
)


def require_dir(path: Path, name: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    return path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def has_model_weights(path: Path) -> bool:
    return any((path / fname).is_file() for fname in WEIGHT_FILENAMES)


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    chosen = torch.device(name)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if chosen.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("--device mps requested but MPS is unavailable")
    return chosen


def release_handle(handle: ACDHandle | ACSAHandle) -> None:
    """Release model weights before loading the next stage onto a small GPU."""
    device = handle.device
    handle.model = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def read_acd_report(model_dir: Path) -> dict:
    for name in ("acd_results.json", ".acd_results.json"):
        candidate = model_dir / name
        if candidate.is_file():
            return load_json(candidate)
    raise FileNotFoundError(
        f"No ACD results/threshold metadata found in {model_dir}. "
        "Thresholds must be selected on Validation, never on Testing."
    )


def get_model_label_order(config: Any, expected: tuple[str, ...], *,
                          allow_generic: bool = False) -> tuple[str, ...]:
    mapping = config.id2label
    order = tuple(str(mapping.get(i, mapping.get(str(i), ""))) for i in range(len(expected)))
    if allow_generic and all(x.startswith("LABEL_") for x in order):
        print("WARNING: ACD checkpoint has generic label names; using shared.labels order.", file=sys.stderr, flush=True)
        order = expected
    if len(order) != len(expected) or set(order) != set(expected):
        raise ValueError(f"Model output order {order} differs from expected {expected}")
    return order


@dataclass
class ACDHandle:
    model: Any
    tokenizer: Any
    order: tuple[str, ...]
    thresholds: np.ndarray
    max_length: int
    device: torch.device
    kind: str


@dataclass
class ACSAHandle:
    model: Any
    tokenizer: Any
    max_length: int
    device: torch.device


def load_acd(model_dir: Path, device: torch.device) -> ACDHandle:
    """Load full FT or LoRA ACD, with saved Validation-tuned thresholds."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = require_dir(model_dir, "ACD model")
    report = read_acd_report(model_dir)
    is_lora = (model_dir / "adapter_config.json").is_file()
    if is_lora:
        from peft import PeftModel

        adapter = load_json(model_dir / "adapter_config.json")
        base_source = report.get("checkpoint") or adapter.get("base_model_name_or_path")
        if not base_source:
            raise ValueError("LoRA ACD adapter has no base model reference")
        # Loading the frozen base in reduced precision reduces memory use for LoRA.
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        if not any((model_dir / f).is_file() for f in (
            "adapter_model.safetensors", "adapter_model.bin"
        )):
            raise FileNotFoundError(f"Missing ACD LoRA adapter weights in {model_dir}")
        if "classifier" not in (adapter.get("modules_to_save") or []):
            raise ValueError("ACD LoRA adapter does not declare a saved classifier head; avoid random-head predictions")
        base = AutoModelForSequenceClassification.from_pretrained(
            base_source, num_labels=len(CODES), problem_type="multi_label_classification",
            id2label={i: c for i, c in enumerate(CODES)},
            label2id={c: i for i, c in enumerate(CODES)},
            torch_dtype=dtype,
        )
        if hasattr(base, "classifier"):
            base.classifier.float()
        model = PeftModel.from_pretrained(base, str(model_dir), is_trainable=False)
        order = CODES
        kind = "LoRA adapter"
    else:
        if not (model_dir / "config.json").is_file() or not has_model_weights(model_dir):
            raise FileNotFoundError(f"ACD full checkpoint requires config.json and model weights: {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True
        )
        order = get_model_label_order(model.config, CODES, allow_generic=True)
        base_source = report.get("checkpoint", "cis-lmu/glot500-base")
        kind = "full fine-tuning"

    saved_codes = report.get("label_names", {})
    if saved_codes and set(saved_codes) != set(CODES):
        raise ValueError("ACD results label_names does not match shared.labels")

    chosen = report.get("category_thresholds")
    if isinstance(chosen, dict) and all(code in chosen for code in order):
        thresholds = np.asarray([float(chosen[code]) for code in order])
    elif report.get("global_threshold") is not None:
        thresholds = np.full(len(order), float(report["global_threshold"]))
    else:
        raise ValueError("ACD results have no saved Validation-selected thresholds")
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("ACD thresholds must be finite and between 0 and 1")

    tok_source = str(model_dir) if any((model_dir / f).is_file() for f in TOKENIZER_FILENAMES) else base_source
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)
    max_len = int(report.get("training_parameters", {}).get("max_length", 128))
    if max_len <= 0:
        raise ValueError("Invalid saved ACD max_length")
    model = model.to(device).eval()
    print(f"ACD: {kind}; max_length={max_len}; thresholds from Validation", file=sys.stderr, flush=True)
    return ACDHandle(model, tokenizer, order, thresholds, max_len, device, kind)


def load_acsa(model_dir: Path, device: torch.device) -> ACSAHandle:
    """Load the user's full-fine-tuned, V1 comment-first ACSA checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = require_dir(model_dir, "ACSA model")
    if not (model_dir / "config.json").is_file() or not has_model_weights(model_dir):
        raise FileNotFoundError(f"ACSA full checkpoint requires config.json and model weights: {model_dir}")
    report = load_json(model_dir / "acsa_results.json")
    expected_format = "tokenizer(text=model_text, text_pair='Aspect: CODE — NAME')"
    if report.get("input_format") not in (None, expected_format):
        raise ValueError(
            f"Unsupported ACSA input format: {report['input_format']}. "
            "This runtime implements the original V1 comment-first paired input."
        )
    if report.get("label_order") not in (None, list(SENTIMENTS)):
        raise ValueError("ACSA report sentiment order differs from Negative, Neutral, Positive")
    if report.get("aspect_codes") not in (None, list(CODES)):
        raise ValueError("ACSA report aspect codes differ from shared.labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    order = get_model_label_order(model.config, SENTIMENTS)
    if order != SENTIMENTS:
        raise ValueError(f"ACSA output order must be {SENTIMENTS}, got {order}")
    tok_source = str(model_dir) if any((model_dir / f).is_file() for f in TOKENIZER_FILENAMES) else report.get("checkpoint")
    if not tok_source:
        raise FileNotFoundError("No tokenizer files or base tokenizer reference for ACSA")
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)
    max_len = int(report.get("training_parameters", {}).get("max_length", 128))
    if max_len <= 0:
        raise ValueError("Invalid saved ACSA max_length")
    model = model.to(device).eval()
    print(f"ACSA: full fine-tuning, V1 paired input; max_length={max_len}", file=sys.stderr, flush=True)
    return ACSAHandle(model, tokenizer, max_len, device)


@torch.inference_mode()
def infer_acd(handle: ACDHandle, texts: list[str], batch_size: int = 2) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(handle.order)), dtype=np.float32)
    outputs = []
    for start in range(0, len(texts), batch_size):
        features = handle.tokenizer(
            texts[start:start + batch_size], max_length=handle.max_length,
            truncation=True, padding=True, return_tensors="pt"
        )
        features = {name: value.to(handle.device) for name, value in features.items()}
        if handle.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                logits = handle.model(**features).logits
        else:
            logits = handle.model(**features).logits
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def aspect_prompt(code: str) -> str:
    if code not in CODES:
        raise ValueError(f"Unexpected aspect category {code!r}")
    return f"Aspect: {code} — {ASPECT_NAMES[code]}"


@torch.inference_mode()
def infer_acsa(handle: ACSAHandle, pairs: list[tuple[str, str]],
               batch_size: int = 2) -> np.ndarray:
    if not pairs:
        return np.zeros((0, len(SENTIMENTS)), dtype=np.float32)
    outputs = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        texts = [t for t, _ in batch]
        prompts = [aspect_prompt(code) for _, code in batch]
        features = handle.tokenizer(
            texts, text_pair=prompts, truncation="only_first",
            max_length=handle.max_length, padding=True, return_tensors="pt"
        )
        features = {name: value.to(handle.device) for name, value in features.items()}
        if handle.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                logits = handle.model(**features).logits
        else:
            logits = handle.model(**features).logits
        outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_comments(texts: list[str], *, acd_dir: Path, acsa_dir: Path,
                     acd_batch_size: int = 2, acsa_batch_size: int = 2,
                     device_name: str = "auto") -> list[dict]:
    """Run ACD -> ACSA sequentially; returns JSON-ready aspect/sentiment records."""
    if acd_batch_size < 1 or acsa_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    texts = [str(t).strip() for t in texts]
    if any(not t for t in texts):
        raise ValueError("Comments must not be empty")
    if not texts:
        return []
    device = select_device(device_name)
    acd = load_acd(acd_dir, device)
    order, thresholds = acd.order, acd.thresholds.copy()
    acd_probs = infer_acd(acd, texts, acd_batch_size)
    release_handle(acd)
    del acd

    hits = acd_probs >= thresholds[None, :]
    assignments = [(i, str(code)) for i in range(len(texts))
                   for j, code in enumerate(order) if hits[i, j]]
    responses = [dict(comment=text, aspects=[]) for text in texts]
    if not assignments:
        return responses

    acsa = load_acsa(acsa_dir, device)
    probs = infer_acsa(acsa, [(texts[i], code) for i, code in assignments], acsa_batch_size)
    release_handle(acsa)
    del acsa
    index_for_code = {code: j for j, code in enumerate(order)}
    for (i, code), distribution in zip(assignments, probs):
        label_index = int(distribution.argmax())
        responses[i]["aspects"].append({
            "aspect_code": code,
            "aspect_name": ASPECT_NAMES[code],
            "acd_probability": round(float(acd_probs[i, index_for_code[code]]), 6),
            "sentiment": SENTIMENTS[label_index],
            "sentiment_confidence": round(float(distribution[label_index]), 6),
            "sentiment_probabilities": {
                label: round(float(distribution[k]), 6)
                for k, label in enumerate(SENTIMENTS)
            },
        })
    return responses
