"""CITI-Senti Approach A: target-conditioned ABSA, fold-fitted word+char TF-IDF, nested grouped CV.

Run from the citi-senti-nlp repository root:
  python -m models.linear_svm.experiment_target_absa --workbook data/splits/citizen_feedback_split.xlsx
Only Training/Validation sheets are read. This is a separate experimental pipeline.

Note: ACSA is target-conditioned to preserve differing polarity for distinct targets
in the same comment/category. It also adds category one-hot and category interactions.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss, confusion_matrix
from sklearn.svm import LinearSVC
from shared.data import read_annotation_sheet, make_aspect_task
from shared.labels import ASPECT_CODES, ASPECT_TO_INDEX, SENTIMENT_TO_INDEX

SEED = 42


def new_vectorizer(use_char=True, min_df=1):
    parts = [('word', TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', lowercase=True,
                                      sublinear_tf=True, min_df=min_df, norm='l2'))]
    if use_char:
        parts.append(('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                                              lowercase=True, sublinear_tf=True, min_df=min_df, norm='l2')))
    return FeatureUnion(parts)


def group_ids(metadata):
    return np.asarray([str(r.get('text_group_id') or r['comment_id']) for r in metadata])


def extract(frame, task):
    if task == 'aspect':
        data = make_aspect_task(frame)
        return {'text': list(data.texts), 'y': np.asarray(data.labels), 'groups': group_ids(data.row_metadata),
                'category': None, 'target': None, 'metadata': data.row_metadata}
    texts, labels, cats, targets, groups, metadata = [], [], [], [], [], []
    for _, row in frame.iterrows():
        if row['sentiment'] not in SENTIMENT_TO_INDEX:
            raise ValueError(f"Invalid sentiment for {row['comment_id']}: {row['sentiment']}")
        if row['aspect_code'] not in ASPECT_TO_INDEX:
            raise ValueError(f"Invalid aspect code for {row['comment_id']}: {row['aspect_code']}")
        if __import__('pandas').isna(row['aspect_text']) or not str(row['aspect_text']).strip():
            raise ValueError(f"Missing target expression for {row['comment_id']} / {row['aspect_code']}")
        texts.append(str(row['model_text']))
        labels.append(SENTIMENT_TO_INDEX[str(row['sentiment'])])
        cats.append(ASPECT_TO_INDEX[str(row['aspect_code'])])
        targets.append(str(row['aspect_text']))
        groups.append(str(row['text_group_id'] or row['comment_id']))
        metadata.append({'comment_id': str(row['comment_id']), 'aspect_code': str(row['aspect_code'])})
    return {'text': texts, 'y': np.asarray(labels), 'groups': np.asarray(groups),
            'category': np.asarray(cats), 'target': targets, 'metadata': metadata}


def subset(data, idx):
    return {k: (None if v is None else [v[int(i)] for i in idx] if isinstance(v, list) else v[idx])
            for k, v in data.items()}


def features(fitted_vec, data, task):
    x = fitted_vec.transform(data['text']).tocsr()
    if task == 'aspect':
        return x
    n, d = x.shape
    cats = np.asarray(data['category'], dtype=int)
    onehot = sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), cats)), shape=(n, len(ASPECT_CODES)))
    # Row i's text is copied into only the block corresponding to its category.
    interaction = sparse.coo_matrix((x.data, (x.tocoo().row, x.tocoo().col + cats[x.tocoo().row] * d)),
                                    shape=(n, d * len(ASPECT_CODES))).tocsr()
    # Target phrase is retained because the actual corpus can contain conflicting polarities
    # for two targets inside one (comment, category). Do NOT drop target without resolving that schema.
    target_x = fitted_vec.transform(data['target']).tocsr()
    return sparse.hstack([x, onehot, interaction, target_x], format='csr')


class AspectSVM:
    """11 independent binary SVMs; constant margins for single-class training folds."""
    def __init__(self, c, weight, max_iter):
        self.c, self.weight, self.max_iter = c, weight, max_iter

    def fit(self, x, y):
        self.models = []
        self.unsupported = []
        for j, code in enumerate(ASPECT_CODES):
            yy = y[:, j]
            if len(np.unique(yy)) < 2:
                constant = int(yy[0]); self.models.append(constant); self.unsupported.append(code)
            else:
                model = LinearSVC(C=self.c, class_weight=self.weight, max_iter=self.max_iter, random_state=SEED)
                model.fit(x, yy); self.models.append(model)
        return self

    def decision_function(self, x):
        return np.column_stack([np.full(x.shape[0], 1.0 if model == 1 else -1.0)
                                if isinstance(model, (int, np.integer)) else model.decision_function(x)
                                for model in self.models])


def fit_model(train, task, c, weight, use_char, min_df, max_iter):
    vec = new_vectorizer(use_char, min_df)
    x = vec.fit_transform(train['text']).tocsr()
    if task == 'sentiment':
        x = features(vec, train, task)
        model = LinearSVC(C=c, class_weight=weight, max_iter=max_iter, random_state=SEED)
        if len(np.unique(train['y'])) < 2:
            raise ValueError('ACSA fold has only one sentiment class')
        model.fit(x, train['y'])
    else:
        model = AspectSVM(c, weight, max_iter).fit(x, train['y'])
    return {'vectorizer': vec, 'model': model, 'task': task,
            'config': {'C': c, 'class_weight': weight, 'char_ngrams': use_char, 'min_df': min_df}}


def margins(bundle, data):
    return bundle['model'].decision_function(features(bundle['vectorizer'], data, bundle['task']))


def choose_thresholds(gold, scores, min_positive=10, max_rate=0.5):
    cuts = np.zeros(gold.shape[1], dtype=float)
    for j in range(gold.shape[1]):
        if gold[:, j].sum() < min_positive:
            continue
        candidate = np.unique(np.r_[0., np.percentile(scores[:, j], np.arange(10, 100, 5))])
        best_score = -1.
        for cut in sorted(candidate, key=lambda v: (abs(v), v)):
            predicted = scores[:, j] >= cut
            if predicted.mean() > max_rate:
                continue
            score = f1_score(gold[:, j], predicted, zero_division=0)
            if score > best_score + 1.e-12:
                best_score, cuts[j] = score, cut
    return cuts


def split_folds(data, task, n):
    if len(np.unique(data['groups'])) < n:
        raise ValueError('Fewer unique source groups than requested CV folds')
    if task == 'sentiment':
        cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
        return list(cv.split(np.zeros(len(data['y'])), data['y'], groups=data['groups']))
    cv = GroupKFold(n_splits=n, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(data['y'])), groups=data['groups']))


def inner_score(data, task, cfg, inner_folds, max_iter):
    folds = split_folds(data, task, inner_folds)
    if task == 'aspect':
        scores = np.zeros(data['y'].shape)
        for train_idx, hold_idx in folds:
            fitted = fit_model(subset(data, train_idx), task, *cfg, max_iter)
            scores[hold_idx] = margins(fitted, subset(data, hold_idx))
        cuts = choose_thresholds(data['y'], scores)
        score = f1_score(data['y'], scores >= cuts, average='macro', zero_division=0)
        return float(score), cuts.tolist()
    preds = np.zeros(len(data['y']), dtype=int)
    for train_idx, hold_idx in folds:
        fitted = fit_model(subset(data, train_idx), task, *cfg, max_iter)
        hold = subset(data, hold_idx)
        preds[hold_idx] = fitted['model'].predict(features(fitted['vectorizer'], hold, task))
    return float(f1_score(data['y'], preds, labels=[0, 1, 2], average='macro', zero_division=0)), None


def select(data, task, configs, inner_folds, max_iter):
    results = []
    for cfg in configs:
        score, cuts = inner_score(data, task, cfg, inner_folds, max_iter)
        results.append({'score': score, 'config': cfg, 'thresholds': cuts})
    # CV only. Prefer lower C, no weighting, and fewer feature blocks on ties.
    results.sort(key=lambda z: (-z['score'], z['config'][0], z['config'][1] is not None, z['config'][2]))
    return results[0], results


def metric(task, y, pred):
    out = {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
           'macro_precision': float(precision_score(y, pred, average='macro', zero_division=0)),
           'macro_recall': float(recall_score(y, pred, average='macro', zero_division=0))}
    if task == 'aspect':
        out.update(micro_f1=float(f1_score(y, pred, average='micro', zero_division=0)),
                   exact_match=float(accuracy_score(y, pred)), hamming_loss=float(hamming_loss(y, pred)),
                   per_aspect_f1=dict(zip(ASPECT_CODES, [float(f1_score(y[:, i], pred[:, i], zero_division=0)) for i in range(len(ASPECT_CODES))])))
    else:
        out.update(accuracy=float(accuracy_score(y, pred)),
                   confusion_matrix=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())
    return out


def run(workbook, output, outer_folds=5, inner_folds=3, final_folds=3,
        c_values=(1, 3, 15), min_dfs=(1,), max_iter=10000):
    train_frame = read_annotation_sheet(workbook, 'Training')
    val_frame = read_annotation_sheet(workbook, 'Validation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = [(float(c), weight, use_char, min_df)
               for c in c_values for weight in (None, 'balanced')
               for use_char in (False, True) for min_df in min_dfs]
    report = {'protocol': 'nested group-isolated CV, TF-IDF fitted separately within each training fold',
              'note': 'Approach A: ACSA uses gold supplied target expressions and preserves opposing target-level polarities. Not fully automatic end-to-end ABSA.',
              'test_used': False, 'tasks': {}}
    for task in ('aspect', 'sentiment'):
        train, val = extract(train_frame, task), extract(val_frame, task)
        if set(train['groups']) & set(val['groups']):
            raise ValueError(f'{task}: train/validation text_group_id overlap detected')
        fold_metrics = []
        for fold_no, (tr, ho) in enumerate(split_folds(train, task, outer_folds), 1):
            inner_train, hold = subset(train, tr), subset(train, ho)
            best, _ = select(inner_train, task, configs, inner_folds, max_iter)
            fitted = fit_model(inner_train, task, *best['config'], max_iter)
            if task == 'aspect':
                pred = (margins(fitted, hold) >= np.asarray(best['thresholds'])).astype(int)
            else:
                pred = fitted['model'].predict(features(fitted['vectorizer'], hold, task))
            m = metric(task, hold['y'], pred)
            fold_metrics.append({'fold': fold_no, 'metrics': m, 'selected_config': list(best['config'])})
            print(f'{task} outer fold {fold_no}/{outer_folds}: Macro-F1={m["macro_f1"]:.4f}', flush=True)
        best, grid = select(train, task, configs, final_folds, max_iter)
        fitted = fit_model(train, task, *best['config'], max_iter)
        fitted['thresholds'] = best['thresholds']
        joblib.dump(fitted, output / f'{task}_bundle.joblib')
        if task == 'aspect':
            val_pred = (margins(fitted, val) >= np.asarray(best['thresholds'])).astype(int)
        else:
            val_pred = fitted['model'].predict(features(fitted['vectorizer'], val, task))
        report['tasks'][task] = {
            'nested_outer_folds': fold_metrics,
            'outer_macro_f1_mean': float(np.mean([v['metrics']['macro_f1'] for v in fold_metrics])),
            'outer_macro_f1_std': float(np.std([v['metrics']['macro_f1'] for v in fold_metrics])),
            'final_train_cv_selected': {'config': list(best['config']), 'macro_f1': best['score'],
                                        'thresholds': best['thresholds']},
            'validation': metric(task, val['y'], val_pred),
            'train_cv_grid': [{'config': list(v['config']), 'macro_f1': v['score']} for v in grid],
        }
        print(f'{task} nested-CV mean={report["tasks"][task]["outer_macro_f1_mean"]:.4f}; '
              f'validation={report["tasks"][task]["validation"]["macro_f1"]:.4f}', flush=True)
    report['evaluation_scope'] = {'aspect': 'comment-level category detection', 'sentiment': 'gold/supplied target-conditioned sentiment', 'fully_automatic_end_to_end': False, 'target_extraction_available': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook', default='data/splits/citizen_feedback_split.xlsx')
    p.add_argument('--output', default='models/linear_svm/artifacts/target_absa')
    p.add_argument('--outer-folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--final-folds', type=int, default=3)
    p.add_argument('--C-grid', type=float, nargs='+', default=[1, 3, 15])
    p.add_argument('--min-df-grid', type=int, nargs='+', default=[1])
    args = p.parse_args()
    if any(c <= 0 for c in args.C_grid) or any(v < 1 for v in args.min_df_grid):
        p.error('C must be positive and min_df at least 1')
    run(args.workbook, args.output, args.outer_folds, args.inner_folds, args.final_folds, args.C_grid, args.min_df_grid)


if __name__ == '__main__':
    main()
