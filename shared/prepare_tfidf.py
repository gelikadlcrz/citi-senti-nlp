"""Fit CITI-Senti shared TF-IDF on TRAINING ONLY; transform VALIDATION.

Run from the repository root:
    python -m shared.prepare_tfidf --workbook data/splits/citizen_feedback_split.xlsx

Neither the original workbook nor its split assignments are modified.
The Testing worksheet is not opened by this command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse

from shared.data import (
    WORKBOOK_DEFAULT, TRAIN_SHEET, VAL_SHEET,
    TaskData, make_aspect_task, make_sentiment_task, read_annotation_sheet,
)
from shared.labels import ASPECT_CODES, ASPECT_NAMES, SENTIMENT_LABELS
from shared.tfidf import (
    ASPECT_VECTORIZER_FILE, SENTIMENT_VECTORIZER_FILE, METADATA_FILE,
    create_vectorizer, save_vectorizers,
)


def _hash_texts(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode('utf-8')).hexdigest()


def save_split(output: Path, task: str, split: str, dataset: TaskData, vectorizer) -> None:
    x = vectorizer.transform(dataset.texts)
    sparse.save_npz(output / f'{task}_{split}_X.npz', x)
    np.save(output / f'{task}_{split}_y.npy', dataset.labels, allow_pickle=False)
    rows_path = output / f'{task}_{split}_rows.csv'
    if not dataset.row_metadata:
        raise ValueError(f'{task}/{split}: no examples')
    with rows_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset.row_metadata[0]))
        writer.writeheader()
        writer.writerows(dataset.row_metadata)
    print(f'  {task:9s} / {split:5s}  X: {x.shape}; labels: {dataset.labels.shape}')


def prepare(*, workbook: str | Path, output: str | Path, max_features: int | None = None, force: bool = False) -> dict:
    path = Path(workbook)
    output = Path(output)
    existing = [output / file for file in (ASPECT_VECTORIZER_FILE, SENTIMENT_VECTORIZER_FILE, METADATA_FILE)]
    if any(p.exists() for p in existing) and not force:
        raise FileExistsError(
            f'Found an existing fitted TF-IDF in {output}. Reuse it for both baselines. '
            'To intentionally regenerate, specify --force and retrain BOTH baseline models.'
        )
    if max_features is not None and max_features < 1:
        raise ValueError('max_features must be a positive integer')

    train = read_annotation_sheet(path, TRAIN_SHEET)
    val = read_annotation_sheet(path, VAL_SHEET)
    datasets = {
        'train': (make_aspect_task(train), make_sentiment_task(train)),
        'val': (make_aspect_task(val), make_sentiment_task(val)),
    }
    aspect_train, sentiment_train = datasets['train']
    aspect_vec = create_vectorizer(max_features=max_features)
    sentiment_vec = create_vectorizer(max_features=max_features)
    # The only two .fit calls; Validation and Testing are never fitted.
    aspect_vec.fit(aspect_train.texts)
    sentiment_vec.fit(sentiment_train.texts)

    metadata = {
        'schema_version': 2,
        'source_workbook': path.name,
        'source_sheets': {'train': TRAIN_SHEET, 'val': VAL_SHEET, 'test': 'Testing'},
        'source_columns': {
            'comment_id': 'comment_id', 'model_text': 'model_text',
            'aspect_id': 'aspect_id', 'aspect_code': 'aspect_code',
            'aspect_text': 'aspect_text', 'sentiment': 'sentiment',
        },
        'aspect_codes': list(ASPECT_CODES),
        'aspect_names': ASPECT_NAMES,
        'sentiment_to_index': {label: i for i, label in enumerate(SENTIMENT_LABELS)},
        'feature_inputs': {
            'aspect': 'one model_text per comment_id',
            'sentiment': 'model_text + aspect_code (category name) + aspect_text per annotation row',
        },
        'important_limit': 'Sentiment prediction requires an aspect target; category-only aspect detection does not provide that target.',
        'vectorizer': {
            'analyzer': 'word', 'ngram_range': [1, 2],
            'token_pattern': r'(?u)\b\w+\b', 'lowercase': True,
            'sublinear_tf': True, 'min_df': 1, 'norm': 'l2',
            'max_features': max_features,
        },
        'aspect_vocabulary_size': len(aspect_vec.vocabulary_),
        'sentiment_vocabulary_size': len(sentiment_vec.vocabulary_),
        'train_comment_count': len(aspect_train.texts),
        'train_sentiment_example_count': len(sentiment_train.texts),
        'validation_comment_count': len(datasets['val'][0].texts),
        'validation_sentiment_example_count': len(datasets['val'][1].texts),
        'train_aspect_input_sha256': _hash_texts(aspect_train.texts),
        'train_sentiment_input_sha256': _hash_texts(sentiment_train.texts),
        'environment': {
            'python': sys.version.split()[0], 'scikit-learn': sklearn.__version__,
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'pandas': pd.__version__, 'joblib': joblib.__version__,
        },
        'sklearn_version': sklearn.__version__,
    }

    output.mkdir(parents=True, exist_ok=True)
    if force:
        # Avoid leaving stale matrices (especially a previously transformed test).
        for suffix in ('*_X.npz', '*_y.npy', '*_rows.csv'):
            for file in output.glob(suffix):
                file.unlink()
    save_vectorizers(output, aspect_vec, sentiment_vec, metadata)
    print(f'TF-IDF fitted ONLY on {TRAIN_SHEET} worksheet of {path.name}.')
    for split, (aspect_data, sentiment_data) in datasets.items():
        save_split(output, 'aspect', split, aspect_data, aspect_vec)
        save_split(output, 'sentiment', split, sentiment_data, sentiment_vec)
    print('Done. Both baseline teams use the SAME matrices in this directory.')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', default=str(WORKBOOK_DEFAULT))
    parser.add_argument('--output', default='models/tfidf')
    parser.add_argument('--max-features', type=int, default=None)
    parser.add_argument('--force', action='store_true', help='Re-fit, invalidate all split matrices, and require retraining')
    args = parser.parse_args()
    prepare(workbook=args.workbook, output=args.output, max_features=args.max_features, force=args.force)


if __name__ == '__main__':
    main()
