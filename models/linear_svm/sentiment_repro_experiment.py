"""Reproducible, training-selected sentiment ablation for CITI-Senti Approach A.

Keep the accepted ACD and ACSA bundles untouched. Never read Testing.
Run as: python -m models.linear_svm.sentiment_repro_experiment --help
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from shared.data import read_annotation_sheet
from shared.labels import ASPECT_CODES, SENTIMENT_TO_INDEX
from models.linear_svm.experiment_target_absa import (
    SEED, extract, features as baseline_features, new_vectorizer, subset
)

# Experimental variants. 'baseline' exactly uses the Approach A feature construction.
VARIANTS = ('baseline', 'local_context', 'context_target_by_category')


def target_window(text: str, target: str, radius: int = 90) -> str:
    """Use a target-centered local character window; no access to annotation labels.

    For non-literal or abbreviated target spans, use the supplied target alone.
    Keep negation and punctuation to retain multilingual sentiment context.
    """
    text, target = str(text), str(target).strip()
    if not target:
        return ''
    match = re.search(re.escape(target), text, flags=re.IGNORECASE)
    if match is None:
        return target
    return text[max(0, match.start() - radius): min(len(text), match.end() + radius)]


def build_features(vectorizer, data: dict, variant: str, radius: int = 90):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    base = baseline_features(vectorizer, data, 'sentiment')
    if variant == 'baseline':
        return base
    local = [target_window(t, a, radius) for t, a in zip(data['text'], data['target'])]
    context_x = vectorizer.transform(local).tocsr()
    if variant == 'local_context':
        return sparse.hstack([base, context_x], format='csr')
    # Category-target interaction: target lexical features activate only the relevant aspect block.
    target_x = vectorizer.transform(data['target']).tocoo()
    cats = np.asarray(data['category'], dtype=int)
    width = target_x.shape[1]
    by_category = sparse.coo_matrix(
        (target_x.data, (target_x.row, target_x.col + cats[target_x.row] * width)),
        shape=(len(cats), len(ASPECT_CODES) * width),
    ).tocsr()
    return sparse.hstack([base, context_x, by_category], format='csr')


def fit_sentiment(train: dict, config: tuple, max_iter: int, radius: int):
    c, weight, variant = config
    vec = new_vectorizer(use_char=True, min_df=1)
    vec.fit(train['text'])  # train-fold text only; never Validation or Testing
    x = build_features(vec, train, variant, radius)
    y = np.asarray(train['y'])
    if len(np.unique(y)) < 2:
        raise ValueError('Sentiment train fold must have at least two classes')
    svc = LinearSVC(C=float(c), class_weight=weight, max_iter=max_iter, random_state=SEED)
    svc.fit(x, y)
    return {'vectorizer': vec, 'model': svc, 'task': 'sentiment',
            'config': {'C': float(c), 'class_weight': weight, 'variant': variant,
                       'word_ngrams': [1, 2], 'char_ngrams': [3, 5],
                       'min_df': 1, 'seed': SEED, 'radius_chars': radius,
                       'target_conditioned': True}}


def infer(bundle, data: dict):
    x = build_features(bundle['vectorizer'], data, bundle['config']['variant'],
                       bundle['config']['radius_chars'])
    return bundle['model'].predict(x)


def folds_for(data, n_splits: int):
    groups, y = np.asarray(data['groups']), np.asarray(data['y'])
    if len(np.unique(groups)) < n_splits:
        raise ValueError('Not enough independent groups for CV')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for tr, ho in folds:
        if set(groups[tr]) & set(groups[ho]):
            raise AssertionError('Group leakage')
    return folds


def score_configs(train, configs, folds, max_iter, radius):
    records = []
    for config in configs:
        preds = np.full(len(train['y']), -1, dtype=int)
        for tr, ho in folds:
            model = fit_sentiment(subset(train, tr), config, max_iter, radius)
            preds[ho] = infer(model, subset(train, ho))
        if (preds < 0).any():
            raise AssertionError('Missing OOF prediction')
        score = float(f1_score(train['y'], preds, labels=[0, 1, 2], average='macro', zero_division=0))
        records.append({'config': list(config), 'oof_macro_f1': score})
    # Deterministic tie-break: simpler variant, lower C, no weighting.
    ranks = {v: i for i, v in enumerate(VARIANTS)}
    records.sort(key=lambda r: (-r['oof_macro_f1'], ranks[r['config'][2]],
                                r['config'][0], r['config'][1] is not None))
    return records[0], records


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), default=str).encode('utf-8')).hexdigest()


def data_digest(data):
    # Record training inputs as a hash only; don't export private text.
    return digest_json([{'text': t, 'category': int(c), 'target': a,
                         'label': int(y), 'group': str(g)}
                        for t, c, a, y, g in zip(data['text'], data['category'],
                                                data['target'], data['y'], data['groups'])])


def versions():
    names = ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'openpyxl')
    found = {}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return {'python': sys.version.split()[0], 'platform': platform.platform(),
            'libraries': found, 'seed': SEED}


def metrics(y, pred):
    return {'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, pred)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            'classification_report': classification_report(y, pred, labels=[0, 1, 2],
                                                           target_names=['Negative', 'Neutral', 'Positive'],
                                                           zero_division=0, output_dict=True)}


def run(workbook, output, outer_folds=5, inner_folds=3, final_folds=3,
        c_values=(1., 3., 10.), weights=('balanced', None),
        variants=('baseline', 'local_context', 'context_target_by_category'),
        max_iter=10000, radius=90):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite experiment outputs: {output}')
    output.mkdir(parents=True, exist_ok=True)
    # Explicitly read only Training and Validation; never read Testing.
    train = extract(read_annotation_sheet(workbook, 'Training'), 'sentiment')
    val = extract(read_annotation_sheet(workbook, 'Validation'), 'sentiment')
    if set(train['groups']) & set(val['groups']):
        raise ValueError('Training and Validation groups overlap')
    configs = [(float(c), weight, variant) for variant in variants
               for c in c_values for weight in weights]
    outer = folds_for(train, outer_folds)
    report = {'protocol': 'Approach A; train-fold TF-IDF; nested stratified-group CV',
              'test_used': False, 'selected_by': 'Training-only nested/grouped CV',
              'baseline_artifacts_modified': False,
              'data': {'training_rows': len(train['y']), 'validation_rows': len(val['y']),
                       'training_groups': len(set(train['groups'])),
                       'validation_groups': len(set(val['groups'])),
                       'training_sha256': data_digest(train), 'validation_sha256': data_digest(val)},
              'settings': {'outer_folds': outer_folds, 'inner_folds': inner_folds,
                           'final_folds': final_folds, 'C_grid': list(c_values),
                           'weights': list(weights), 'variants': list(variants),
                           'max_iter': max_iter, 'radius_chars': radius},
              'environment': versions(), 'nested_outer': []}
    outer_preds = np.full(len(train['y']), -1, dtype=int)
    fold_manifest = []
    for number, (tr, ho) in enumerate(outer, start=1):
        inner_train = subset(train, tr)
        selected, _ = score_configs(inner_train, configs,
                                    folds_for(inner_train, inner_folds), max_iter, radius)
        fitted = fit_sentiment(inner_train, tuple(selected['config']), max_iter, radius)
        pred = infer(fitted, subset(train, ho))
        outer_preds[ho] = pred
        report['nested_outer'].append({'fold': number, 'selected': selected,
                                       'metrics': metrics(np.asarray(train['y'])[ho], pred)})
        for idx in ho:
            fold_manifest.append({'training_row_index': int(idx), 'group_id': str(train['groups'][idx]),
                                  'fold': number})
        print(f'outer {number}/{outer_folds}: Macro-F1={report["nested_outer"][-1]["metrics"]["macro_f1"]:.4f}', flush=True)
    report['nested_oof'] = metrics(train['y'], outer_preds)
    report['outer_macro_f1_mean'] = float(np.mean([f['metrics']['macro_f1'] for f in report['nested_outer']]))
    report['outer_macro_f1_std'] = float(np.std([f['metrics']['macro_f1'] for f in report['nested_outer']]))
    selected, grid = score_configs(train, configs, folds_for(train, final_folds), max_iter, radius)
    fitted = fit_sentiment(train, tuple(selected['config']), max_iter, radius)
    val_pred = infer(fitted, val)
    report['final_train_selected'] = selected
    report['final_train_grid'] = grid
    report['validation'] = metrics(val['y'], val_pred)
    report['scope_note'] = 'Gold/supplied target evaluation; NOT fully automatic end-to-end ABSA. Validation not used to select settings.'
    pd.DataFrame(fold_manifest).sort_values('training_row_index').to_csv(output / 'outer_fold_manifest.csv', index=False)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    joblib.dump(fitted, output / 'sentiment_bundle.joblib')
    print(f'Nested mean={report["outer_macro_f1_mean"]:.4f}; Training selected={selected["config"]}; '
          f'Validation={report["validation"]["macro_f1"]:.4f}', flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook', default='data/splits/citizen_feedback_split.xlsx')
    p.add_argument('--output', default='models/linear_svm/artifacts/sentiment_repro_v1')
    p.add_argument('--outer-folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--final-folds', type=int, default=3)
    p.add_argument('--C-grid', nargs='+', type=float, default=[1., 3., 10.])
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--max-iter', type=int, default=10000)
    p.add_argument('--radius', type=int, default=90)
    args = p.parse_args()
    if any(c <= 0 for c in args.C_grid):
        p.error('C must be positive')
    if args.radius < 0 or min(args.outer_folds, args.inner_folds, args.final_folds) < 2:
        p.error('radius must be nonnegative and folds at least 2')
    run(args.workbook, args.output, args.outer_folds, args.inner_folds, args.final_folds,
        args.C_grid, variants=args.variants, max_iter=args.max_iter, radius=args.radius)


if __name__ == '__main__':
    main()
