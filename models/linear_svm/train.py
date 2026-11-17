"""Initial, untuned Linear SVM run on shared precomputed TF-IDF.

python -m models.linear_svm.train
python -m models.linear_svm.train --C 0.3 --class-weight balanced
"""
from __future__ import annotations
import argparse
from models.linear_svm.config import load_config
from shared.train_common import run_training, CLASS_WEIGHT_CHOICES

def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--features')
    p.add_argument('--output')
    p.add_argument('--C', type=float)
    p.add_argument('--max-iter', type=int)
    p.add_argument('--class-weight', choices=CLASS_WEIGHT_CHOICES)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    t = cfg.get('training', {})
    return run_training(
        'linear_svm',
        features=a.features or cfg['features'],
        output=a.output or cfg['output'],
        c=a.C if a.C is not None else float(t.get('C', 1.0)),
        max_iter=a.max_iter if a.max_iter is not None else int(t.get('max_iter', 5000)),
        class_weight=a.class_weight if a.class_weight is not None else t.get('class_weight', 'none'),
    )

if __name__ == '__main__':
    main()
