"""Training-only, grouped OOF diagnostics for CITI-Senti Linear SVM Approach A.

Copy to models/linear_svm/error_analysis.py. No Testing or Validation sheets
are read. This is post-selection diagnostic analysis, NOT an unbiased estimate
of final generalization performance (the fixed config was previously chosen).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from shared.data import read_annotation_sheet
from shared.labels import ASPECT_CODES
from models.linear_svm.experiment_target_absa import (
    extract, subset, split_folds, fit_model, margins, inner_score,
)


def run(workbook, report_path, output, folds=5, threshold_folds=3, max_iter=10000):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    cfg = report['tasks']['aspect']['final_train_cv_selected']['config']
    config = (float(cfg[0]), None if cfg[1] in (None, 'none') else cfg[1], bool(cfg[2]), int(cfg[3]))
    data = extract(read_annotation_sheet(workbook, 'Training'), 'aspect')
    y = np.asarray(data['y'], dtype=int)
    if y.ndim != 2 or y.shape[1] != len(ASPECT_CODES):
        raise ValueError('Unexpected aspect target dimensions')

    n, k = y.shape
    if n == 0:
        raise ValueError('No Training comments found')
    pred = np.zeros((n, k), dtype=int)
    score = np.full((n, k), np.nan, dtype=float)
    thresholds = np.full((n, k), np.nan, dtype=float)
    folds_for_row = np.full(n, -1, dtype=int)

    for fold_no, (tr, ho) in enumerate(split_folds(data, 'aspect', folds), 1):
        training = subset(data, tr)
        heldout = subset(data, ho)
        # Thresholds tuned ONLY on this outer fold's training portion.
        _, cuts = inner_score(training, 'aspect', config, threshold_folds, max_iter)
        cut = np.asarray(cuts, dtype=float)
        fitted = fit_model(training, 'aspect', *config, max_iter)
        sc = margins(fitted, heldout)
        score[ho] = sc
        thresholds[ho] = cut
        pred[ho] = (sc >= cut).astype(int)
        folds_for_row[ho] = fold_no
        print(f'Completed training-only diagnostic fold {fold_no}/{folds}', flush=True)

    if (folds_for_row < 0).any() or not np.isfinite(score).all():
        raise RuntimeError('Some records did not receive OOF predictions')

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = Counter()
    for i in range(n):
        md = data['metadata'][i]
        cid = str(md.get('comment_id', f'row_{i}'))
        actual = [ASPECT_CODES[j] for j in range(k) if y[i, j]]
        predicted = [ASPECT_CODES[j] for j in range(k) if pred[i, j]]
        false_positive = sorted(set(predicted) - set(actual))
        false_negative = sorted(set(actual) - set(predicted))
        for code in false_positive: counts[(code, 'FP')] += 1
        for code in false_negative: counts[(code, 'FN')] += 1
        rows.append({
            'comment_id': cid,
            'fold': int(folds_for_row[i]),
            'actual_aspects': '|'.join(actual),
            'predicted_aspects': '|'.join(predicted),
            'false_positives': '|'.join(false_positive),
            'false_negatives': '|'.join(false_negative),
            'exact_match': int(not (false_positive or false_negative)),
        })

    def write_csv(path, records, fields):
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    write_csv(out / 'oof_comment_errors.csv', rows, list(rows[0].keys()))

    detailed = []
    for i in range(n):
        cid = rows[i]['comment_id']
        for j, code in enumerate(ASPECT_CODES):
            gold, predicted = int(y[i, j]), int(pred[i, j])
            if gold == predicted:
                continue
            detailed.append({
                'comment_id': cid,
                'fold': int(folds_for_row[i]),
                'aspect_code': code,
                'error': 'FN' if gold else 'FP',
                'gold': gold,
                'predicted': predicted,
                'margin': float(score[i, j]),
                'threshold': float(thresholds[i, j]),
                'distance_from_threshold': float(score[i, j] - thresholds[i, j]),
            })
    detailed.sort(key=lambda z: (z['aspect_code'], z['error'], abs(z['distance_from_threshold'])))
    write_csv(out / 'oof_aspect_errors.csv', detailed,
              ['comment_id', 'fold', 'aspect_code', 'error', 'gold', 'predicted', 'margin', 'threshold', 'distance_from_threshold'])

    p, r, f1, support = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    summary = []
    for j, code in enumerate(ASPECT_CODES):
        summary.append({
            'aspect_code': code,
            'support': int(support[j]),
            'precision': float(p[j]),
            'recall': float(r[j]),
            'f1': float(f1[j]),
            'false_positives': int(counts[(code, 'FP')]),
            'false_negatives': int(counts[(code, 'FN')]),
        })
    write_csv(out / 'oof_aspect_summary.csv', summary,
              ['aspect_code', 'support', 'precision', 'recall', 'f1', 'false_positives', 'false_negatives'])
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss
    metrics = {
        'scope': 'Training grouped out-of-fold diagnostics; not nested model-selection evaluation',
        'comments': int(n),
        'groups': int(len(set(map(str, data['groups'])))),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y, pred, average='micro', zero_division=0)),
        'exact_match_accuracy': float(accuracy_score(y, pred)),
        'hamming_loss': float(hamming_loss(y, pred)),
        'total_false_positives': int(sum(v['false_positives'] for v in summary)),
        'total_false_negatives': int(sum(v['false_negatives'] for v in summary)),
        'config': {'C': config[0], 'class_weight': config[1], 'char_ngrams': config[2], 'min_df': config[3]},
    }
    (out / 'oof_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    (out / 'README.txt').write_text(
        'TRAINING-ONLY out-of-fold diagnostic exports. No raw comment text included.\n'
        'Fixed config selected in a prior experiment, so OOF metrics are diagnostic\n'
        'and should NOT be called unbiased nested-CV estimates.\n'
        'All thresholds were selected within the corresponding outer training fold.\n'
        'Keep comment IDs and files private; do not publish identifying examples.\n', encoding='utf-8')
    print('\nAspect   Support    F1     FP   FN')
    for x in summary:
        print(f"{x['aspect_code']:6} {x['support']:7}  {x['f1']:.3f}  {x['false_positives']:4} {x['false_negatives']:4}")
    print('\nTRAINING-ONLY DIAGNOSTIC METRICS')
    print(json.dumps(metrics, indent=2))
    print(f'\nSaved: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', default='data/splits/citizen_feedback_split.xlsx')
    parser.add_argument('--report', default='models/linear_svm/artifacts/target_absa/report.json')
    parser.add_argument('--output', default='models/linear_svm/artifacts/target_aspect_error_analysis')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--threshold-folds', type=int, default=3)
    args = parser.parse_args()
    run(args.workbook, args.report, args.output, args.folds, args.threshold_folds)


if __name__ == '__main__':
    main()
