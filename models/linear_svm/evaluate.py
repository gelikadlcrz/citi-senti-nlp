"""Evaluate SAVED SVM models with shared features; Testing explicitly gated.

python -m models.linear_svm.evaluate --split val
# Phase 2 only, after finalizing every configuration:
# python -m models.linear_svm.evaluate --split test --allow-test
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
from models.linear_svm.thresholds import aspect_margins, apply_thresholds, load_thresholds
from shared.features import load_features
from shared.train_common import _metrics

def evaluate(*, artifacts='models/linear_svm/artifacts/tuned', features='models/tfidf',
             split='val', allow_test=False):
    if split not in ('val','test'):
        raise ValueError('Only val or test supported')
    if split == 'test' and not allow_test:
        raise ValueError('Held-out test use is locked: explicitly pass --allow-test only after model selection')
    root = Path(artifacts)
    cutoffs = load_thresholds(root)
    report = {'model_type':'linear_svm', 'split':split,
              'aspect_thresholds':'training_oof_margins' if cutoffs else 'native_margin_0', 'tasks':{}}
    for task in ('aspect','sentiment'):
        model = joblib.load(root / f'{task}_model.joblib')  # Trusted local team artifacts only
        x, y = load_features(features, task=task, split=split)
        if model.n_features_in_ != x.shape[1]:
            raise ValueError(f'{task}: saved model and {split} TF-IDF width disagree')
        pred = apply_thresholds(aspect_margins(model,x),cutoffs) if task=='aspect' else model.predict(x)
        report['tasks'][task] = _metrics(task,y,pred)
    return report

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', default='models/linear_svm/artifacts/tuned')
    p.add_argument('--features', default='models/tfidf')
    p.add_argument('--split', choices=['val','test'],default='val')
    p.add_argument('--allow-test',action='store_true')
    p.add_argument('--output',type=Path)
    args = p.parse_args(argv)
    report = evaluate(artifacts=args.artifacts,features=args.features,split=args.split,allow_test=args.allow_test)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return report
if __name__=='__main__':
    main()
