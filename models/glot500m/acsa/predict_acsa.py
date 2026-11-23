"""Standalone ACSA inference; accepts gold or ACD-predicted aspect codes.

This script DOES NOT run ACD. In production, pass ACD's detected categories to
predict_pairs(), then return one sentiment per detected category.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_acsa import SENTIMENTS, aspect_prompt


@torch.inference_mode()
def predict_pairs(model, tokenizer, pairs: list[tuple[str, str]], device,
                  max_length: int = 128, batch_size: int = 2) -> list[dict]:
    """pairs=[(comment_text, aspect_code), ...] in input order."""
    output = []
    model.eval()
    use_amp = device.type == "cuda"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp else None
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        texts = [text for text, _ in chunk]
        prompts = [aspect_prompt(code) for _, code in chunk]
        batch = tokenizer(texts, text_pair=prompts, max_length=max_length,
                          truncation="only_first", padding=True, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in batch.items()}
        if use_amp:
            with torch.autocast("cuda", dtype=dtype):
                logits = model(**inputs).logits
        else:
            logits = model(**inputs).logits
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for (_, code), probability in zip(chunk, probabilities):
            result = {
                "aspect_code": code,
                "sentiment": SENTIMENTS[int(np.argmax(probability))],
                "sentiment_probabilities": {
                    label: round(float(probability[i]), 6)
                    for i, label in enumerate(SENTIMENTS)
                },
            }
            output.append(result)
    return output


def load_acsa(model_dir: Path, device: torch.device | None = None):
    """Load the full ACSA checkpoint once and reuse it for many predictions."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = device or (torch.device("cuda") if torch.cuda.is_available() else
                        torch.device("mps") if hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available() else torch.device("cpu"))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tuple(model.config.id2label[i] for i in range(3)) != SENTIMENTS:
        raise ValueError("Checkpoint sentiment label order differs from Negative, Neutral, Positive")
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment conditional on aspect category")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--aspects", nargs="+", required=True,
                        help="One or more A01..A11 aspect codes, e.g. A02 A10")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    report = args.model_dir / "acsa_results.json"
    config = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    length = args.max_length or config.get("training_parameters", {}).get("max_length", 128)
    model, tokenizer, device = load_acsa(args.model_dir)
    results = predict_pairs(model, tokenizer, [(args.text, code) for code in args.aspects],
                            device, max_length=length, batch_size=args.batch_size)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
