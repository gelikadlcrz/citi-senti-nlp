"""Run with: python -m pytest -q tests/test_sentiment_repro.py"""
import numpy as np
import pytest
from scipy import sparse

from models.linear_svm.sentiment_repro_experiment import (
    VARIANTS, build_features, folds_for, target_window, fit_sentiment, infer,
)
from models.linear_svm.experiment_target_absa import new_vectorizer


def synthetic():
    # Same comment can have two different target-specific polarities; group together.
    return {'text': ['Maganda parking system pero mahal parking fee'] * 2 +
                    ['Mas mura parking fee dito', 'Napakamahal naman ng fee'],
            'target': ['parking system', 'parking fee', 'parking fee', 'fee'],
            'category': np.array([3, 2, 2, 2]),
            'y': np.array([2, 0, 2, 0]),
            'groups': np.array(['g1', 'g1', 'g2', 'g3'])}


def test_context_preserves_contrast():
    s = 'Maganda parking system pero mahal parking fee'
    assert 'mahal' in target_window(s, 'parking fee')
    assert target_window(s, 'not present') == 'not present'


def test_feature_variants_keep_target_rows():
    data = synthetic()
    vec = new_vectorizer(True, 1)
    vec.fit(data['text'])
    widths = [build_features(vec, data, name).shape for name in VARIANTS]
    assert all(shape[0] == len(data['y']) for shape in widths)
    assert widths[0][1] < widths[1][1] < widths[2][1]


def test_group_integrity_and_repeatability():
    data = synthetic()
    first = folds_for(data, 2)
    second = folds_for(data, 2)
    for (tr, ho), (tr2, ho2) in zip(first, second):
        assert np.array_equal(tr, tr2)
        assert np.array_equal(ho, ho2)
        assert not (set(data['groups'][tr]) & set(data['groups'][ho]))


def test_fit_and_predict_all_variants():
    data = synthetic()
    for variant in VARIANTS:
        bundle = fit_sentiment(data, (1., 'balanced', variant), 2000, 90)
        assert len(infer(bundle, data)) == len(data['y'])
