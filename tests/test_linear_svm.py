"""Small isolated smoke/unit tests; does not evaluate the held-out Testing worksheet."""
import numpy as np
import pytest
from models.linear_svm.thresholds import apply_thresholds, tune_thresholds
from models.linear_svm.tune import tune
from models.linear_svm.evaluate import evaluate
from shared.labels import ASPECT_CODES

def test_svm_margins_are_not_probabilities():
    y=np.zeros((20,len(ASPECT_CODES)),dtype=np.int8)
    y[:8,0]=1
    scores=np.full(y.shape,-2.0)
    scores[:8,0]=1.0
    cuts=tune_thresholds(y,scores,min_positives=5)
    assert set(cuts)==set(ASPECT_CODES)
    pred=apply_thresholds(scores,cuts)
    assert np.array_equal(pred[:,0],y[:,0])

def test_test_partition_requires_explicit_unlock():
    with pytest.raises(ValueError,match='Held-out'):
        evaluate(split='test')

from models.linear_svm.category_sentiment import select_sentiments

def test_category_head_never_returns_empty_sentiment():
    margins=np.array([[-1.,-2.,-0.1],[0.5,-0.3,0.7]])
    flags=select_sentiments(margins)
    assert flags[0].tolist()==[0,0,1]
    assert flags[1].tolist()==[1,0,1]
