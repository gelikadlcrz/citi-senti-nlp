"""Synthetic-only checks for the LR sentence-only adapter; no research test split."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from models.logistic_regression.category_sentiment import (
    MODEL_FILE, category_examples, make_category_input, select_sentiments,
    train_category_sentiment,
)
from models.logistic_regression.predict import LogisticRegressionPredictor
from shared.labels import ASPECT_CODES


def _row(cid, code, target, sentiment, comment):
    return dict(comment_id=cid, aspect_id=f"{cid}-{code}-{target}", aspect_code=code,
                aspect_text=target, sentiment=sentiment, model_text=comment)


def test_aggregation_keeps_multiple_aspects_and_mixed_sentiment():
    frame = pd.DataFrame([
        _row("C1", "A01", "ordinance", "Positive", "good ordinance but unfair penalties"),
        _row("C1", "A01", "penalties", "Negative", "good ordinance but unfair penalties"),
        _row("C1", "A03", "fees", "Negative", "good ordinance but unfair penalties"),
        _row("C2", "A03", "cost", "Neutral", "how much is the cost"),
    ])
    texts, y, manifest = category_examples(frame)
    assert len(texts) == 3
    assert [m["aspect_code"] for m in manifest] == ["A01", "A03", "A03"]
    assert y[0].tolist() == [1, 0, 1]  # Negative AND Positive retained
    assert y[1].tolist() == [1, 0, 0]
    assert y[2].tolist() == [0, 1, 0]
    assert "ASPECT_TARGET:" not in texts[0]  # never invent an extracted target


def test_category_sentiment_selects_at_least_one_and_can_return_two():
    proba = np.array([[.7, .1, .8], [.2, .3, .4]])
    pred = select_sentiments(proba)
    assert pred.tolist() == [[1, 0, 1], [0, 0, 1]]
    with pytest.raises(ValueError):
        select_sentiments(np.array([[.5, .5]]))


class _FakeAspectVectorizer:
    vocabulary_ = {"one": 0}

    def transform(self, texts):
        return sparse.csr_matrix(np.ones((len(texts), 1)))


class _FakeAspectModel:
    n_features_in_ = 1

    def predict(self, x):
        y = np.zeros((x.shape[0], len(ASPECT_CODES)), dtype=np.int8)
        y[:, 0] = 1
        y[:, 2] = 1
        return y


class _FakeCategoryModel:
    n_features_in_ = 1

    def predict_proba(self, x):
        return np.array([[.1, .1, .9], [.9, .1, .7]])[:x.shape[0]]


class _FakeTargetSentimentModel:
    n_features_in_ = 1

    def predict(self, x):
        return np.zeros(x.shape[0], dtype=int)


def _fake_predictor(tmp_path):
    pred = LogisticRegressionPredictor.__new__(LogisticRegressionPredictor)
    pred.artifacts_dir = tmp_path
    pred.aspect_model = _FakeAspectModel()
    pred.sentiment_model = _FakeTargetSentimentModel()
    pred.aspect_vectorizer = _FakeAspectVectorizer()
    pred.sentiment_vectorizer = _FakeAspectVectorizer()
    pred.aspect_thresholds = None
    pred.category_sentiment_model = _FakeCategoryModel()
    return pred


def test_sentence_only_preserves_multiple_predicted_categories(tmp_path):
    predictor = _fake_predictor(tmp_path)
    result = predictor.predict_comment("Maganda ang batas pero mahal ang multa")
    assert [x["code"] for x in result["aspects"]] == ["A01", "A03"]
    assert result["aspects"][0]["sentiments"] == ["Positive"]
    assert result["aspects"][1]["sentiments"] == ["Negative", "Positive"]
    assert all(x["target_text"] is None for x in result["aspects"])
    assert result["sentiment_scope"] == "category_level"


def test_multiple_targets_same_category_are_not_collapsed(tmp_path):
    predictor = _fake_predictor(tmp_path)
    result = predictor.predict_targets("good ordinance, expensive fees", [
        {"aspect_code": "A03", "aspect_text": "fees"},
        {"aspect_code": "A03", "aspect_text": "penalties"},
        {"aspect_code": "A01", "aspect_text": "ordinance"},
    ])
    assert len(result["target_sentiments"]) == 3
    assert [x["aspect_code"] for x in result["target_sentiments"]] == ["A03", "A03", "A01"]
    assert result["sentiment_scope"] == "target_conditioned"
    with pytest.raises(ValueError, match="nonempty"):
        predictor.predict_targets("comment", [])


def test_sentence_only_missing_category_head_errors_clearly(tmp_path):
    predictor = _fake_predictor(tmp_path)
    predictor.category_sentiment_model = None
    with pytest.raises(FileNotFoundError, match="category_sentiment"):
        predictor.predict_comment("Maganda pero mahal")


def test_category_training_is_separate_from_original_benchmark(tmp_path):
    folder = tmp_path / "features"
    folder.mkdir()
    artifacts = tmp_path / "lr"
    artifacts.mkdir()
    # Existing LR models are NOT re-fit or overwritten by category training.
    (artifacts / "aspect_model.joblib").write_bytes(b"original aspect")
    (artifacts / "sentiment_model.joblib").write_bytes(b"original sentiment")
    rows = []
    for i in range(24):
        label = ["Negative", "Neutral", "Positive"][i % 3]
        text = f"comment {i} " + {"Negative":"expensive", "Neutral":"question", "Positive":"helpful"}[label]
        rows.append(_row(f"C{i:02}", "A03" if i % 2 else "A01", f"target {i}", label, text))
    # Duplicate same comment/category with opposite sentiment must be preserved.
    rows.append(_row("C00", "A01", "other target", "Positive", rows[0]["model_text"]))
    val = [_row(f"V{i}", "A03" if i % 2 else "A01", f"val {i}",
                ["Negative", "Neutral", "Positive"][i % 3], f"new comment {i} fees")
           for i in range(6)]
    workbook = tmp_path / "split.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Training", index=False)
        pd.DataFrame(val).to_excel(writer, sheet_name="Validation", index=False)
    texts, _, _ = category_examples(pd.DataFrame(rows))
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(texts)
    joblib.dump(vec, folder / "sentiment_vectorizer.joblib")
    result = train_category_sentiment(
        artifacts=artifacts, features=folder, workbook=workbook, c=1.0, max_iter=500,
    )
    assert (artifacts / MODEL_FILE).is_file()
    assert (artifacts / "category_sentiment_validation.json").is_file()
    assert (artifacts / "aspect_model.joblib").read_bytes() == b"original aspect"
    assert (artifacts / "sentiment_model.joblib").read_bytes() == b"original sentiment"
    assert result["train_mixed_sentiment_categories"] == 1
    assert result["is_original_benchmark_metric"] is False
    assert result["val_category_rows"] == 6
    assert 0 <= result["metrics"]["macro_f1"] <= 1
    assert joblib.load(artifacts / MODEL_FILE).n_features_in_ == len(vec.vocabulary_)


def test_pair_metrics_counts_multi_aspect_and_same_category_mixed_sentiments():
    from models.logistic_regression.evaluate_sentence_only import pair_metrics

    gold = [
        {("A01", "Positive"), ("A03", "Negative")},
        {("A03", "Negative"), ("A03", "Positive")},
    ]
    pred = [
        {("A01", "Positive"), ("A10", "Negative")},
        {("A03", "Positive")},
    ]
    report = pair_metrics(gold, pred)
    assert report["gold_pairs"] == 4
    assert report["predicted_pairs"] == 3
    assert report["micro_precision"] == pytest.approx(2 / 3)
    assert report["micro_recall"] == pytest.approx(2 / 4)
    assert report["micro_f1"] == pytest.approx(4 / 7)
    assert report["exact_match_accuracy"] == 0
