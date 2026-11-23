"""CITI-Senti ACD -> ACSA inference; prints JSON for a new citizen comment.

No training or gold labels are used. Uses Validation-selected ACD thresholds,
then predicts a Negative/Neutral/Positive sentiment per detected aspect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.glot500m.integration.absa_runtime import predict_comments


def main():
    parser = argparse.ArgumentParser(description="Predict aspect-category sentiments for citizen feedback")
    parser.add_argument("--acd-dir", required=True, type=Path,
                        help="Full-fine-tuned ACD checkpoint OR LoRA adapter directory")
    parser.add_argument("--acsa-dir", required=True, type=Path,
                        help="Full-fine-tuned ACSA V1 checkpoint")
    parser.add_argument("--text", required=True, help="Citizen feedback comment")
    parser.add_argument("--acd-batch-size", type=int, default=2)
    parser.add_argument("--acsa-batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    result = predict_comments(
        [args.text], acd_dir=args.acd_dir, acsa_dir=args.acsa_dir,
        acd_batch_size=args.acd_batch_size, acsa_batch_size=args.acsa_batch_size,
        device_name=args.device,
    )[0]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
