"""Transform the held-out Testing worksheet LATER using saved TF-IDF (no fitting).

Run only after models/hyperparameters are finalized:
    python -m shared.transform_split --workbook data/splits/citizen_feedback_split.xlsx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.data import WORKBOOK_DEFAULT, TEST_SHEET, make_aspect_task, make_sentiment_task, read_annotation_sheet
from shared.prepare_tfidf import save_split
from shared.tfidf import METADATA_FILE, load_aspect_vectorizer, load_sentiment_vectorizer


def transform_test(workbook: str | Path, features: str | Path, *, force: bool = False) -> None:
    directory = Path(features)
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 2:
        raise ValueError('Expected schema_version=2; this script cannot use an older shared TF-IDF package')
    files = [directory / f'{task}_test_{suffix}' for task in ('aspect', 'sentiment') for suffix in ('X.npz', 'y.npy', 'rows.csv')]
    if any(file.exists() for file in files) and not force:
        raise FileExistsError('Test feature files already exist. Use --force only to intentionally replace them.')
    annotated = read_annotation_sheet(workbook, TEST_SHEET)
    save_split(directory, 'aspect', 'test', make_aspect_task(annotated), load_aspect_vectorizer(directory))
    save_split(directory, 'sentiment', 'test', make_sentiment_task(annotated), load_sentiment_vectorizer(directory))
    print('Testing transformed using TRAIN-fitted TF-IDF. No fitting or evaluation performed.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', default=str(WORKBOOK_DEFAULT))
    parser.add_argument('--features', default='models/tfidf')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    transform_test(args.workbook, args.features, force=args.force)


if __name__ == '__main__':
    main()
