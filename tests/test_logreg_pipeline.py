"""LR-only safeguards for paper Sections 3.4–3.7; all fixtures synthetic."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from models.logistic_regression.pipeline import (
    Candidate, search_grid, category_interactions, category_records, audit_conflicts,
    select_thresholds, aspect_mask, _factory, _manifest, AspectBinaryRelevance,
    predict_text, make_folds,
)
from models.logistic_regression.metrics import metrics, group_bootstrap_ci
from shared.labels import ASPECT_CODES


def _row(cid, code, sentiment, text="comment", ref=""):
    return dict(comment_id=cid, aspect_code=code, sentiment=sentiment, model_text=text,
                source_reference=ref)


def test_search_space_exactly_24_prespecified_combinations():
    grid = search_grid()
    assert len(grid) == len(set(grid)) == 24
    assert set(c.C for c in grid) == {0.1, 1.0, 10.0}
    assert set(c.word_ngram_range for c in grid) == {(1, 1), (1, 2)}
    assert set(c.min_df for c in grid) == {1, 2}
    assert set(c.class_weight for c in grid) == {None, "balanced"}


def test_sparse_category_feature_blocks_no_target_text():
    x = sparse.csr_matrix([[1., 0., 2.], [0., 3., 0.]])
    encoded = category_interactions(x, np.array([0, 2]), n_categories=3)
    assert encoded.shape == (2, 3 + 3 + 3*3)
    assert encoded[0, :3].toarray().tolist() == [[1.,0.,2.]]
    assert encoded[0, 3:6].toarray().tolist() == [[1.,0.,0.]]
    assert encoded[0, 6:15].toarray().tolist() == [[1.,0.,2.,0.,0.,0.,0.,0.,0.]]
    assert encoded[1, 6:15].toarray().tolist() == [[0.,0.,0.,0.,0.,0.,0.,3.,0.]]
    with pytest.raises(ValueError):
        category_interactions(x, np.array([0, 5]), n_categories=3)


def test_contradictory_sentiments_require_explicit_review():
    frame = pd.DataFrame([
        _row("C1", "A01", "Positive", "great ordinance but poor enforcement"),
        _row("C1", "A01", "Negative", "great ordinance but poor enforcement"),
        _row("C1", "A03", "Neutral", "great ordinance but poor enforcement"),
    ])
    assert audit_conflicts(frame) == {"C1|A01": ["Negative", "Positive"]}
    with pytest.raises(ValueError, match="Unresolved"):
        category_records(frame)
    rows = category_records(frame, {"C1|A01":"Negative"})
    assert len(rows) == 2
    assert {r[1] for r in rows} == {"A01", "A03"}
    with pytest.raises(ValueError, match="Unused"):
        category_records(frame, {"C1|A01":"Negative", "C2|A02":"Positive"})


def test_threshold_range_global_and_min_20_class_support():
    truth = np.zeros((50, 11), dtype=np.int8)
    truth[:25, 0] = 1             # eligible
    truth[:19, 1] = 1             # ineligible
    score = np.tile(np.linspace(.1, .9, 50)[:,None], (1,11))
    chosen, policy = select_thresholds(truth, score)
    assert len(policy["grid"]) == 13
    assert policy["eligible_categories"] == ["A01"]
    assert chosen["A02"] == policy["global"]
    assert all(.2 <= value <= .8 for value in chosen.values())
    disabled, _ = select_thresholds(truth, score, unsupported={"A01"})
    assert disabled["A01"] == .5
    assert "A01" not in select_thresholds(truth, score, unsupported={"A01"})[1]["eligible_categories"]


def test_one_class_binary_fold_uses_constant_without_crashing():
    x = sparse.csr_matrix(np.eye(12))
    y = np.zeros((12,11), dtype=np.int8)
    y[:6,0] = 1
    model = AspectBinaryRelevance(C=1., class_weight=None, max_iter=100).fit(x, y)
    assert len(model.unsupported_) == 10
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (12, 11)
    assert np.all(probabilities[:,1:] == 0)


def test_grouped_cv_never_splits_same_group():
    groups = np.array([f"G{i//2}" for i in range(18)])
    for tr, val in make_folds(np.arange(18), groups, n_splits=3):
        assert not (set(groups[tr]) & set(groups[val]))


def test_manifest_requires_group_isolation_and_frozen_outer_folds(tmp_path):
    dev = pd.DataFrame([_row(f"C{i}", "A01", "Positive", ref="same" if i < 2 else f"src{i}") for i in range(10)])
    rows = [dict(comment_id=f"C{i}", group_id="X" if i < 2 else f"G{i}", split="development", outer_fold=str(i%5 if i>=2 else 0)) for i in range(10)]
    rows.append(dict(comment_id="T1", group_id="T", split="test", outer_fold=""))
    path = tmp_path/"groups.csv"
    pd.DataFrame(rows).to_csv(path,index=False)
    _manifest(path, dev)
    rows[-1]["group_id"] = "X"
    pd.DataFrame(rows).to_csv(path,index=False)
    with pytest.raises(ValueError, match="cross development and test"):
        _manifest(path,dev)


def test_shared_factory_contract_rejects_word_only_patch():
    from shared.tfidf import create_vectorizer
    with pytest.raises(RuntimeError, match="Awaiting the shared TF-IDF patch"):
        _factory(Candidate((1,2), 1, 1.0, None), create_vectorizer)


def test_shared_factory_accepts_word_and_character_design():
    def mock_shared_factory(*, word_ngram_range, char_ngram_range, min_df):
        return FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b",
                                     ngram_range=word_ngram_range, min_df=min_df)),
            ("character", TfidfVectorizer(analyzer="char", ngram_range=char_ngram_range,
                                          min_df=min_df)),
        ])
    vec = _factory(Candidate((1,2),1,1.,None), mock_shared_factory)
    X = vec.fit_transform(["ayus ang programa", "mahal ang bayad"])
    assert X.shape[0] == 2 and X.shape[1] > 3


def test_end_to_end_returns_multiple_distinct_pairs_and_no_category_neutral_fabrication():
    class AspectModel:
        def predict_proba(self, x):
            ans = np.zeros((len(x.toarray()),11))
            ans[:,0] = .9
            ans[:,2] = .8
            return ans
    class SentimentModel:
        def predict(self, x):
            return np.array([2, 0])
    class MockV:
        def transform(self, texts):
            return sparse.csr_matrix(np.ones((len(texts), 2)))
    bundle = {"aspect":{"classifier":AspectModel(), "vectorizer":MockV(),
                         "thresholds": {c:.5 for c in ASPECT_CODES}},
              "sentiment":{"classifier":SentimentModel(),"vectorizer":MockV()}}
    assert predict_text(bundle, "ordinance good but parking fees high") == [
        ("A01","Positive"),("A03","Negative")]
    bundle["aspect"]["thresholds"] = {c:1.0 for c in ASPECT_CODES}
    assert predict_text(bundle, "ordinance good but parking fees high") == []


def test_pair_metrics_fixed_33_and_group_bootstrap():
    gold = [{("A01","Positive"), ("A03","Negative")}, {("A01","Neutral")}]
    pred = [{("A01","Positive")}, {("A01","Neutral")}]
    report = metrics(gold, pred)
    assert report["supported_pairs"] == 3
    assert report["macro_f1_33_pairs"] == pytest.approx(2/33)
    assert report["supported_macro_f1"] == pytest.approx(2/3)
    ci = group_bootstrap_ci(gold, pred, ["thread1","thread2"], replicates=100, seed=42)
    assert ci["group_count"] == 2
    assert all(0 <= score <= 1 for score in ci["unadjusted_percentile_95_ci"])


def test_nested_protocol_runs_all_outer_folds_and_saves_artifacts_without_test(tmp_path, monkeypatch):
    """A one-candidate synthetic smoke test; production uses the full 24 per task."""
    from models.logistic_regression import pipeline as m
    import joblib

    rows = []
    groups = []
    for i in range(30):
        code = "A01" if i % 2 else "A03"
        sentiment = ["Negative", "Neutral", "Positive"][i % 3]
        text = f"ordinance {i} " + ["bad high fee", "how much cost", "good helpful support"][i % 3]
        rows.append(dict(comment_id=f"C{i}", aspect_id=f"C{i}-A1", aspect_code=code,
                         aspect_text=f"target{i}", sentiment=sentiment, model_text=text,
                         source_reference=f"source{i}"))
        groups.append(dict(comment_id=f"C{i}", group_id=f"group{i}",
                           split="development", outer_fold=str(i % 5)))
    workbook = tmp_path / "split.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows[:22]).to_excel(writer, sheet_name="Training", index=False)
        pd.DataFrame(rows[22:]).to_excel(writer, sheet_name="Validation", index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(groups).to_csv(manifest, index=False)

    def feature_factory(*, word_ngram_range, char_ngram_range, min_df):
        return FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b",
                                      ngram_range=word_ngram_range, min_df=min_df)),
            ("character", TfidfVectorizer(analyzer="char", ngram_range=char_ngram_range,
                                           min_df=min_df)),
        ])

    monkeypatch.setattr(m, "search_grid", lambda: [Candidate((1, 1), 1, 0.1, None)])
    report = m.nested_grouped_run(workbook=workbook, manifest=manifest,
                                   output=tmp_path / "artifacts", feature_factory=feature_factory,
                                   max_iter=60)
    assert len(report["outer_folds"]) == 5
    assert report["development_comments"] == 30
    assert report["test_used_by_this_procedure"] is False
    saved = joblib.load(tmp_path / "artifacts" / m.BUNDLE_FILE)
    assert len(m.predict_text(saved, "ordinance good fee")) >= 0
    assert (tmp_path / "artifacts" / "selection_grid_results.json").exists()
    assert len((tmp_path / "artifacts" / "outer_oof_predictions.jsonl").read_text().splitlines()) == 30
    with pytest.raises(FileExistsError):
        m.nested_grouped_run(workbook=workbook, manifest=manifest,
                             output=tmp_path / "artifacts", feature_factory=feature_factory,
                             max_iter=60)


def test_train_entrypoint_routes_to_grouped_pipeline(monkeypatch):
    from models.logistic_regression import train, pipeline

    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return {"ok": True}

    monkeypatch.setattr(pipeline, "main", fake_main)
    result = train.main(["--grouped-cv", "--manifest", "data/splits/group_manifest.csv"])
    assert result == {"ok": True}
    assert called["argv"] == ["--manifest", "data/splits/group_manifest.csv"]


def test_predict_entrypoint_uses_grouped_bundle_when_present(tmp_path, monkeypatch):
    from models.logistic_regression import predict
    from models.logistic_regression.pipeline import BUNDLE_FILE

    (tmp_path / BUNDLE_FILE).touch()
    monkeypatch.setattr(predict, "predict_frozen_comment",
                        lambda text, *, artifacts: {"pairs": [{"aspect_code": "A01", "sentiment": "Positive"}]})
    output = predict.main(["--artifacts", str(tmp_path), "--text", "Great ordinance"])
    assert output["pairs"][0]["aspect_code"] == "A01"


def test_frozen_evaluation_remains_locked_without_explicit_test_approval(tmp_path):
    from models.logistic_regression.evaluate import evaluate_frozen

    with pytest.raises(ValueError, match="held-out Testing is locked|Held-out Testing is locked"):
        evaluate_frozen(artifacts=tmp_path, allow_test=False)


def test_evaluate_entrypoint_routes_to_frozen_test(monkeypatch):
    import sys
    from models.logistic_regression import evaluate

    captured = {}

    def fake_frozen(argv):
        captured["argv"] = argv

    monkeypatch.setattr(evaluate, "_frozen_test_cli", fake_frozen)
    monkeypatch.setattr(sys, "argv", ["pytest", "--frozen-test", "--allow-test"])
    evaluate.main()
    assert captured["argv"] == ["--allow-test"]
