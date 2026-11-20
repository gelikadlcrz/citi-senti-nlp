"""Train CITI-Senti Logistic Regression on the shared TF-IDF features.

Settings come from models/logistic_regression/config.yaml; any flag given on
the command line overrides the config. Run from the repository root:
    python -m models.logistic_regression.train
    python -m models.logistic_regression.train --C 0.3 --class-weight balanced

For the grouped research protocol (once shared inputs are ready):
    python -m models.logistic_regression.train --grouped-cv --manifest data/splits/group_manifest.csv
"""
from __future__ import annotations

import argparse
import sys

from models.logistic_regression.config import load_config
from shared.train_common import CLASS_WEIGHT_CHOICES, run_training


def main(argv: list[str] | None = None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--grouped-cv" in argv or "--audit" in argv:
        from models.logistic_regression.pipeline import main as run_grouped
        return run_grouped([item for item in argv if item != "--grouped-cv"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config (default: config.yaml beside this script)")
    parser.add_argument("--features")
    parser.add_argument("--output")
    parser.add_argument("--C", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--class-weight", choices=CLASS_WEIGHT_CHOICES)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    training = cfg.get("training", {})
    try:
        return run_training(
            "logreg",
            features=args.features or cfg["features"],
            output=args.output or cfg["output"],
            c=args.C if args.C is not None else float(training.get("C", 1.0)),
            max_iter=args.max_iter if args.max_iter is not None else int(training.get("max_iter", 5000)),
            class_weight=args.class_weight if args.class_weight is not None else training.get("class_weight"),
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()