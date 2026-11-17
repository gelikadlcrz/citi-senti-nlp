"""TRAINING-only grouped cross-validation tuning of both LinearSVC tasks.

python -m models.linear_svm.tune
python -m models.linear_svm.tune --cv-folds 3 --C-grid 0.1 1 --class-weight-grid none

No Testing features are loaded. Precomputed shared Training TF-IDF stays frozen.
Validation reports are descriptive; selection uses Training CV only.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from models.linear_svm.config import load_config
from models.linear_svm.thresholds import aspect_margins, apply_thresholds, tune_thresholds, save_thresholds
from shared.features import load_features
from shared.train_common import _metrics, build_model, parse_class_weight

FIELDS = ['task', 'C', 'class_weight', 'cv_macro_f1', 'cv_macro_f1_std', 'val_macro_f1', 'selected']

def load_groups(features: str | Path, task: str, expected: int) -> np.ndarray:
    path = Path(features) / f'{task}_train_rows.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path}: grouped CV requires text_group_id manifest')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    groups = np.asarray([r.get('text_group_id') or r['comment_id'] for r in rows], dtype=object)
    if len(groups) != expected:
        raise ValueError(f'{path}: expected {expected} rows, got {len(groups)}')
    return groups

def oof_aspect_margins(x, y, groups, cv_folds, c, weight, max_iter):
    folds = GroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    output = np.zeros(y.shape, dtype=float)
    for tr, ho in folds.split(x, groups=groups):
        model = build_model('linear_svm', 'aspect', c, max_iter, weight)
        model.fit(x[tr], y[tr])
        output[ho] = aspect_margins(model, x[ho])
    return output

def sentiment_fold_scores(x, y, groups, cv_folds, c, weight, max_iter):
    folds = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    for tr, ho in folds.split(x, y, groups=groups):
        model = build_model('linear_svm', 'sentiment', c, max_iter, weight)
        model.fit(x[tr], y[tr])
        scores.append(float(f1_score(y[ho], model.predict(x[ho]), labels=[0, 1, 2], average='macro', zero_division=0)))
    return np.array(scores)

def tune(*, features, output, c_grid, class_weight_grid, cv_folds=5, max_iter=5000,
         min_positives=10, max_positive_rate=0.5, tune_aspect_thresholds=True) -> dict:
    if cv_folds < 2 or max_iter < 1:
        raise ValueError('cv_folds must be >= 2, max_iter >= 1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = [(float(c), parse_class_weight(w)) for c in c_grid for w in class_weight_grid]
    if not configs or any(c <= 0 for c, _ in configs):
        raise ValueError('Supply positive C values and at least one class weight')
    report = {
        'model_type': 'linear_svm', 'sklearn_version': sklearn.__version__,
        'selection_partition': 'Training (grouped CV)', 'evaluation_partition': 'Validation (report only)',
        'note': 'Uses precomputed shared word TF-IDF fitted on all Training before CV; not fold-fitted word+character TF-IDF or nested CV.',
        'n_configs_per_task': len(configs), 'tasks': {}, 'selected': {},
    }
    trials = []
    for task in ('aspect', 'sentiment'):
        x_tr, y_tr = load_features(features, task=task, split='train')
        x_val, y_val = load_features(features, task=task, split='val')
        groups = load_groups(features, task, len(y_tr))
        candidates = []
        for c, cw in configs:
            model = build_model('linear_svm', task, c, max_iter, cw)
            if task == 'aspect':
                scores = oof_aspect_margins(x_tr, y_tr, groups, cv_folds, c, cw, max_iter)
                thresholds = tune_thresholds(y_tr, scores, min_positives=min_positives,
                                             max_positive_rate=max_positive_rate) if tune_aspect_thresholds else None
                cv_f1 = float(f1_score(y_tr, apply_thresholds(scores, thresholds), average='macro', zero_division=0))
                cv_std = None  # cutoffs optimized on pooled OOF margins, not fold-specific.
            else:
                folds = sentiment_fold_scores(x_tr, y_tr, groups, cv_folds, c, cw, max_iter)
                cv_f1, cv_std, thresholds = float(folds.mean()), float(folds.std()), None
            model.fit(x_tr, y_tr)
            if task == 'aspect':
                pred = apply_thresholds(aspect_margins(model, x_val), thresholds)
            else:
                pred = model.predict(x_val)
            val_f1 = float(_metrics(task, y_val, pred)['macro_f1'])
            row = dict(task=task, C=c, class_weight=cw or 'none', cv_macro_f1=cv_f1,
                       cv_macro_f1_std=cv_std, val_macro_f1=val_f1, selected=False)
            trials.append(row)
            candidates.append((row, model, thresholds))
            print(f'{task:9s} C={c:<6g} weight={cw or "none":8s}: CV={cv_f1:.4f}, validation={val_f1:.4f}', flush=True)
        # Do NOT rank on validation. Prefer lower C, then unweighted on equal CV.
        candidates.sort(key=lambda item: (-item[0]['cv_macro_f1'], item[0]['C'], item[0]['class_weight']!='none'))
        chosen, model, thresholds = candidates[0]
        chosen['selected'] = True
        joblib.dump(model, output / f'{task}_model.joblib')
        report['selected'][task] = {k: chosen[k] for k in ('C', 'class_weight', 'cv_macro_f1')}
        if task == 'aspect':
            margins = aspect_margins(model, x_val)
            if thresholds is not None:
                save_thresholds(output, thresholds)
            report['tasks'][task] = {
                'native_margin_cutoff_0': _metrics(task, y_val, apply_thresholds(margins, None)),
                'selected_cutoffs': _metrics(task, y_val, apply_thresholds(margins, thresholds)),
                'thresholds': thresholds,
            }
        else:
            report['tasks'][task] = _metrics(task, y_val, model.predict(x_val))
    with (output / 'tuning_results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(trials)
    (output / 'selected_config.json').write_text(json.dumps({
        'max_iter': max_iter, 'cv_folds': cv_folds, 'C_grid': c_grid,
        'class_weight_grid': class_weight_grid, 'selected': report['selected'],
    }, indent=2)+'\n', encoding='utf-8')
    (output / 'validation_metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'\nSaved final TRAINING-selected SVM models and validation reports to {output}', flush=True)
    return report

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--features')
    p.add_argument('--output')
    p.add_argument('--C-grid', type=float, nargs='+')
    p.add_argument('--class-weight-grid', nargs='+', choices=['none', 'balanced'])
    p.add_argument('--cv-folds', type=int)
    p.add_argument('--max-iter', type=int)
    p.add_argument('--no-thresholds', action='store_true')
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    t = cfg['tuning']
    return tune(features=args.features or cfg['features'], output=args.output or t['output'],
                c_grid=args.C_grid if args.C_grid is not None else t['C_grid'],
                class_weight_grid=args.class_weight_grid or t['class_weight_grid'],
                cv_folds=args.cv_folds or t['cv_folds'], max_iter=args.max_iter or t['max_iter'],
                min_positives=t['min_positives'], max_positive_rate=t['max_positive_rate'],
                tune_aspect_thresholds=t['tune_aspect_thresholds'] and not args.no_thresholds)

if __name__ == '__main__':
    main()
