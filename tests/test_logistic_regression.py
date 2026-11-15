"""Synthetic-only Logistic Regression checks; does not load research data."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from shared.labels import ASPECT_CODES, SENTIMENT_LABELS
from shared.train_common import _estimator, _metrics


def _sample_features(tmp_path):
    folder = tmp_path / "tfidf"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({
        "schema_version": 2,
        "aspect_codes": list(ASPECT_CODES),
        "sentiment_to_index": {name: i for i, name in enumerate(SENTIMENT_LABELS)},
    }))
    for split, n, start in (("train", 18, 0), ("val", 6, 100)):
        ids = [f"C{i:03}" for i in range(start, start + n)]
        x = sparse.csr_matrix(np.eye(n, 24, dtype=np.float32))
        y_aspect = np.zeros((n, len(ASPECT_CODES)), dtype=np.int8)
        for i in range(n):
            y_aspect[i, i % len(ASPECT_CODES)] = 1
        y_sentiment = np.array([i % 3 for i in range(n)], dtype=np.int8)
        for task, y in (("aspect", y_aspect), ("sentiment", y_sentiment)):
            sparse.save_npz(folder / f"{task}_{split}_X.npz", x)
            np.save(folder / f"{task}_{split}_y.npy", y)
            with (folder / f"{task}_{split}_rows.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["comment_id", "text_group_id"])
                writer.writeheader()
                writer.writerows({"comment_id": cid, "text_group_id": f"g-{cid}"} for cid in ids)
    return folder


def test_lr_estimators_fit_sparse_features():
    X = sparse.csr_matrix(np.array([
        [1, 0, 0, 0], [0.8, 0.2, 0, 0],
        [0, 1, 0, 0], [0, 0.8, 0.2, 0],
        [0, 0, 1, 0], [0, 0, 0.8, 0.2],
    ], dtype=np.float32))
    y = np.array([0, 0, 1, 1, 2, 2])
    model = _estimator("logreg", c=1.0, max_iter=1000)
    assert isinstance(model, LogisticRegression)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (6,)
    assert 0.0 <= _metrics("sentiment", y, predictions)["macro_f1"] <= 1.0

    y_aspect = np.column_stack([y == 0, y == 1]).astype(np.int8)
    aspect_model = OneVsRestClassifier(_estimator("logreg", c=1.0, max_iter=1000))
    aspect_model.fit(X, y_aspect)
    pred_aspect = aspect_model.predict(X)
    assert pred_aspect.shape == y_aspect.shape


def _save_synthetic_lr_artifacts(feature_dir, artifact_dir):
    """Train two tiny models; avoids research annotations and fitted vectors."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from models.logistic_regression.predict import LogisticRegressionPredictor
    from shared.features import load_features

    artifact_dir.mkdir(parents=True, exist_ok=True)
    x_aspect, y_aspect = load_features(feature_dir, task="aspect", split="train")
    x_sentiment, y_sentiment = load_features(feature_dir, task="sentiment", split="train")
    aspect_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    sentiment_model = LogisticRegression(max_iter=1000)
    aspect_model.fit(x_aspect, y_aspect)
    sentiment_model.fit(x_sentiment, y_sentiment)
    joblib.dump(aspect_model, artifact_dir / "aspect_model.joblib")
    joblib.dump(sentiment_model, artifact_dir / "sentiment_model.joblib")

    # Vocabulary order and size match the synthetic saved classifiers.
    vocabulary = {f"token{i}": i for i in range(x_aspect.shape[1])}
    for task in ("aspect", "sentiment"):
        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        vectorizer.fit([" ".join(vocabulary)])
        joblib.dump(vectorizer, feature_dir / f"{task}_vectorizer.joblib")


def test_saved_lr_models_support_validation_evaluation(tmp_path):
    from models.logistic_regression.evaluate import evaluate_saved_models

    features = _sample_features(tmp_path)
    artifacts = tmp_path / "lr_models"
    _save_synthetic_lr_artifacts(features, artifacts)
    report = evaluate_saved_models(artifacts=artifacts, features=features)
    assert report["split"] == "val"
    assert set(report["tasks"]) == {"aspect", "sentiment"}
    assert 0 <= report["tasks"]["aspect"]["macro_f1"] <= 1
    assert 0 <= report["tasks"]["sentiment"]["macro_f1"] <= 1


def test_evaluate_guards_the_held_out_test_partition(tmp_path):
    from models.logistic_regression.evaluate import evaluate_saved_models

    with pytest.raises(ValueError, match="held out"):
        evaluate_saved_models(artifacts=tmp_path, features=tmp_path, split="test")


def test_inference_uses_saved_models_and_requires_aspect_target(tmp_path):
    from models.logistic_regression.predict import LogisticRegressionPredictor

    features = _sample_features(tmp_path)
    artifacts = tmp_path / "lr_models"
    _save_synthetic_lr_artifacts(features, artifacts)
    predictor = LogisticRegressionPredictor(artifacts=artifacts, features=features)
    predicted_aspects = predictor.predict_aspects("token5 token6")
    assert isinstance(predicted_aspects, list)
    assert all(item["code"] in ASPECT_CODES for item in predicted_aspects)
    sentiment = predictor.predict_sentiment(
        model_text="token5 token6", aspect_code="A03", aspect_text="token5"
    )
    assert sentiment in SENTIMENT_LABELS
    with pytest.raises(ValueError, match="must not be blank"):
        predictor.predict_aspects("  ")
    with pytest.raises(ValueError, match="non-empty"):
        predictor.predict_sentiment(
            model_text="token5", aspect_code="A03", aspect_text=""
        )
