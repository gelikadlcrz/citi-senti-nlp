"""Unit tests: synthetic examples; does not depend on private research workbook."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from shared.data import (
    make_aspect_task, make_sentiment_input, make_sentiment_task,
    read_annotation_sheet,
)
from shared.features import load_features
from shared.prepare_tfidf import prepare
from shared.tfidf import load_aspect_vectorizer, load_sentiment_vectorizer
from shared.transform_split import transform_test


def sample_workbook(folder: Path) -> Path:
    path = folder / 'citizen_feedback_split.xlsx'
    rows_train = [
        ('C01','C01-A1','A02','inconsistent enforcement','Negative','Hindi maayos implementation.'),
        ('C01','C01-A2','A01','good ordinance','Positive','Hindi maayos implementation.'),
        # Same category but contrasting sentiments: retaining aspect_text is necessary.
        ('C02','C02-A1','A10','improved traffic','Positive','Traffic okay but bad parking.'),
        ('C02','C02-A2','A10','bad parking','Negative','Traffic okay but bad parking.'),
        ('C03','C03-A1','A06','policy','Neutral','What is the policy?'),
        ('C04','C04-A1','A04','accessible','Positive','Mabilis at accessible.'),
        ('C05','C05-A1','A03','fees','Negative','Mahal ang fees.'),
    ]
    rows_val = [
        ('C06','C06-A1','A07','response','Negative','unknownvalidationtoken lacks response.'),
        ('C07','C07-A1','A02','enforcement','Positive','Good enforcement.'),
    ]
    rows_test = [
        ('C08','C08-A1','A01','good','Positive','unknowntesttoken good policy.'),
    ]
    columns = ['comment_id','aspect_id','aspect_code','aspect_text','sentiment','model_text']
    with pd.ExcelWriter(path) as w:
        for name, rows in [('Training',rows_train),('Validation',rows_val),('Testing',rows_test)]:
            pd.DataFrame(rows, columns=columns).to_excel(w, sheet_name=name, index=False)
    return path


def test_preserves_same_category_opposite_sentiments(tmp_path):
    path = sample_workbook(tmp_path)
    df = read_annotation_sheet(path, 'Training')
    aspects = make_aspect_task(df)
    sentiments = make_sentiment_task(df)
    assert aspects.labels.shape == (5, 11)
    assert sentiments.labels.shape == (7,)
    assert sentiments.row_metadata[2]['aspect_id'] == 'C02-A1'
    assert sentiments.row_metadata[3]['aspect_id'] == 'C02-A2'
    assert sentiments.labels[2] != sentiments.labels[3]
    assert sentiments.texts[2] != sentiments.texts[3]
    assert 'ASPECT_TARGET' in make_sentiment_input('hello', 'A02', 'implementation')


def test_fit_shared_vectorizers_and_hold_test(tmp_path):
    path = sample_workbook(tmp_path)
    features = tmp_path / 'output'
    meta = prepare(workbook=path, output=features)
    assert meta['schema_version'] == 2
    assert meta['train_comment_count'] == 5
    assert meta['train_sentiment_example_count'] == 7
    a = load_aspect_vectorizer(features)
    s = load_sentiment_vectorizer(features)
    assert 'unknownvalidationtoken' not in a.vocabulary_
    assert 'unknownvalidationtoken' not in s.vocabulary_
    assert 'unknowntesttoken' not in a.vocabulary_
    assert 'unknowntesttoken' not in s.vocabulary_
    train_x, train_y = load_features(features, task='aspect', split='train')
    val_x, val_y = load_features(features, task='aspect', split='val')
    sx, sy = load_features(features, task='sentiment', split='train')
    assert train_x.shape[0] == train_y.shape[0] == 5
    assert val_x.shape[0] == val_y.shape[0] == 2
    assert train_x.shape[1] == val_x.shape[1]
    assert sx.shape[0] == sy.shape[0] == 7
    assert not (features / 'aspect_test_X.npz').exists()

    with pytest.raises(FileExistsError):
        prepare(workbook=path, output=features)

    transform_test(workbook=path, features=features)
    test_x, test_y = load_features(features, task='sentiment', split='test')
    assert test_x.shape[0] == test_y.shape[0] == 1
    assert 'unknowntesttoken' not in load_sentiment_vectorizer(features).vocabulary_
    with (features / 'sentiment_train_rows.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows[2]['aspect_text'] == 'improved traffic'


def test_validation_requires_existing_columns(tmp_path):
    path = sample_workbook(tmp_path)
    with pytest.raises(ValueError, match='Could not read worksheet'):
        read_annotation_sheet(path, 'Missing')
