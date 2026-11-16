"""Synthetic-only checks for LR config loading, tuning and aspect thresholds."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from models.logistic_regression.config import load_config
from models.logistic_regression.thresholds import (
    apply_thresholds, load_thresholds, out_of_fold_probabilities, save_thresholds, tune_thresholds,
)
from shared.labels import ASPECT_CODES
from shared.train_common import parse_class_weight

N_ASPECTS = len(ASPECT_CODES)


def _write_features(folder, n_train=44, n_val=22, seed=0):
    """Learnable synthetic data: feature j fires when aspect j is present."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("val", n_val)):
        y_aspect = np.zeros((n, N_ASPECTS), dtype=np.int8)
        for i in range(n):
            y_aspect[i, i % N_ASPECTS] = 1
        x = np.hstack([y_aspect + rng.normal(0, 0.1, y_aspect.shape), rng.random((n, 5))])
        y_sentiment = np.array([i % 3 for i in range(n)], dtype=np.int8)
        x_sent = np.eye(3)[y_sentiment] + rng.normal(0, 0.1, (n, 3))
        sparse.save_npz(folder / f"aspect_{split}_X.npz", sparse.csr_matrix(np.abs(x).astype(np.float32)))
        np.save(folder / f"aspect_{split}_y.npy", y_aspect)
        sparse.save_npz(folder / f"sentiment_{split}_X.npz", sparse.csr_matrix(np.abs(x_sent).astype(np.float32)))
        np.save(folder / f"sentiment_{split}_y.npy", y_sentiment)
    return folder


def test_shipped_config_loads():
    cfg = load_config()
    assert cfg["output"].endswith("baseline_C1")
    assert all(c > 0 for c in cfg["tuning"]["C_grid"])


def test_parse_class_weight():
    assert parse_class_weight("none") is None
    assert parse_class_weight(None) is None
    assert parse_class_weight("Balanced") == "balanced"
    with pytest.raises(ValueError):
        parse_class_weight("heavy")


def test_train_uses_config_output_and_cli_override(tmp_path):
    from models.logistic_regression.train import main

    features = _write_features(tmp_path / "tfidf")
    out = tmp_path / "out"
    result = main(["--features", str(features), "--output", str(out), "--C", "0.5",
                   "--class-weight", "balanced"])
    assert result["C"] == 0.5 and result["class_weight"] == "balanced"
    assert (out / "aspect_model.joblib").is_file()
    assert (out / "sentiment_model.joblib").is_file()


def test_threshold_tuning_lowers_cutoff_for_underconfident_aspect():
    y = np.zeros((10, N_ASPECTS), dtype=int)
    y[:5, 0] = 1
    proba = np.full((10, N_ASPECTS), 0.1)
    proba[:5, 0] = 0.3          # true positives, but below 0.5
    thresholds = tune_thresholds(y, proba, min_positives=1)
    assert thresholds["A01"] <= 0.3
    assert thresholds["A02"] == 0.5   # no positives -> left at default
    assert apply_thresholds(proba, thresholds)[:5, 0].all()
    assert not apply_thresholds(proba, None)[:, 0].any()


def test_thresholds_round_trip(tmp_path):
    assert load_thresholds(tmp_path) is None
    save_thresholds(tmp_path, {"A01": 0.3})
    assert load_thresholds(tmp_path) == {"A01": 0.3}
    (tmp_path / "aspect_thresholds.json").write_text(json.dumps({"A99": 0.4}))
    with pytest.raises(ValueError, match="unknown"):
        load_thresholds(tmp_path)


def test_tune_selects_and_saves_everything(tmp_path):
    from models.logistic_regression.evaluate import evaluate_saved_models
    from models.logistic_regression.tune import tune

    features = _write_features(tmp_path / "tfidf")
    out = tmp_path / "tuned"
    report = tune(features=features, output=out, c_grid=[0.1, 10.0],
                  class_weight_grid=["none", "balanced"], max_iter=2000)
    for name in ("aspect_model.joblib", "sentiment_model.joblib", "aspect_thresholds.json",
                 "validation_metrics.json", "selected_config.json", "tuning_results.csv"):
        assert (out / name).is_file(), name
    assert set(report["selected"]) == {"aspect", "sentiment"}
    assert len((out / "tuning_results.csv").read_text().strip().splitlines()) == 1 + 2 * 2 * 2

    evaluated = evaluate_saved_models(artifacts=out, features=features)
    assert evaluated["aspect_thresholds"] == "tuned"
    assert evaluated["tasks"]["aspect"]["macro_f1"] == pytest.approx(
        report["tasks"]["aspect"]["tuned_thresholds"]["macro_f1"])


def test_tune_rejects_bad_grid(tmp_path):
    from models.logistic_regression.tune import tune

    with pytest.raises(ValueError):
        tune(features=tmp_path, output=tmp_path / "o", c_grid=[], class_weight_grid=["none"])
    with pytest.raises(ValueError):
        tune(features=tmp_path, output=tmp_path / "o", c_grid=[-1.0], class_weight_grid=["none"])


def test_aspect_with_too_few_examples_keeps_default_cutoff():
    y = np.zeros((40, N_ASPECTS), dtype=int)
    y[:3, 0] = 1                       # only 3 positives
    proba = np.full((40, N_ASPECTS), 0.1)
    proba[:3, 0] = 0.3
    assert tune_thresholds(y, proba, min_positives=10)["A01"] == 0.5


def test_cutoff_may_not_flag_almost_every_comment():
    # A useless model: positives and negatives get the same probability,
    # so the best F1 would come from saying "yes" to everyone.
    y = np.zeros((20, N_ASPECTS), dtype=int)
    y[:4, 0] = 1
    proba = np.full((20, N_ASPECTS), 0.4)
    unguarded = tune_thresholds(y, proba, min_positives=1, max_positive_rate=1.0)
    guarded = tune_thresholds(y, proba, min_positives=1, max_positive_rate=0.5)
    assert apply_thresholds(proba, unguarded)[:, 0].all()
    assert apply_thresholds(proba, guarded)[:, 0].mean() <= 0.5


def test_out_of_fold_probabilities_cover_every_training_row(tmp_path):
    from shared.features import load_features
    from shared.train_common import build_model

    features = _write_features(tmp_path / "tfidf")
    x, y = load_features(features, task="aspect", split="train")
    oof = out_of_fold_probabilities(
        lambda: build_model("logreg", "aspect", 1.0, 2000), x, y, n_splits=4)
    assert oof.shape == y.shape
    assert ((oof > 0) & (oof < 1)).all()
    with pytest.raises(ValueError):
        out_of_fold_probabilities(lambda: None, x, y, n_splits=1)


def test_tuning_csv_marks_exactly_one_selected_row_per_task(tmp_path):
    import csv
    from models.logistic_regression.tune import tune

    features = _write_features(tmp_path / "tfidf")
    out = tmp_path / "tuned"
    tune(features=features, output=out, c_grid=[0.1, 10.0],
         class_weight_grid=["none", "balanced"], max_iter=2000, cv_folds=3)
    rows = list(csv.DictReader((out / "tuning_results.csv").open()))
    for task in ("aspect", "sentiment"):
        assert sum(r["selected"] == "True" for r in rows if r["task"] == task) == 1


def test_predictor_default_points_at_tuned_artifacts():
    import inspect
    from models.logistic_regression.predict import LogisticRegressionPredictor

    default = inspect.signature(LogisticRegressionPredictor).parameters["artifacts"].default
    assert str(default).endswith("artifacts/tuned")


def test_grouped_out_of_fold_keeps_each_group_in_one_fold(tmp_path):
    from shared.features import load_features
    from shared.train_common import build_model

    features = _write_features(tmp_path / "tfidf")
    x, y = load_features(features, task="aspect", split="train")
    groups = np.repeat(np.arange(len(y) // 2), 2)   # pairs of rows share a comment
    seen_together = []

    class Spy:
        def __init__(self):
            self.model = build_model("logreg", "aspect", 1.0, 2000)
        def fit(self, xs, ys):
            seen_together.append(xs.shape[0])
            self.model.fit(xs, ys)
            return self
        def predict_proba(self, xs):
            return self.model.predict_proba(xs)

    oof = out_of_fold_probabilities(Spy, x, y, n_splits=4, groups=groups)
    assert oof.shape == y.shape
    assert all(n % 2 == 0 for n in seen_together)   # groups were never split


def test_tune_uses_grouped_cv_when_row_manifest_exists(tmp_path):
    import csv
    from models.logistic_regression.tune import tune

    features = _write_features(tmp_path / "tfidf")
    for task, n in (("aspect", 44), ("sentiment", 44)):
        with (features / f"{task}_train_rows.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["comment_id", "text_group_id"])
            writer.writerows([[i, f"g{i // 2}"] for i in range(n)])
    report = tune(features=features, output=tmp_path / "tuned", c_grid=[1.0],
                  class_weight_grid=["balanced"], max_iter=2000, cv_folds=3)
    assert report["cv_grouped_by_comment"] == {"aspect": True, "sentiment": True}


def test_compare_runs_reports_baseline_and_tuned(tmp_path):
    from models.logistic_regression.compare import compare_runs
    from models.logistic_regression.train import main as train_main
    from models.logistic_regression.tune import tune

    features = _write_features(tmp_path / "tfidf")
    train_main(["--features", str(features), "--output", str(tmp_path / "baseline_C1")])
    tune(features=features, output=tmp_path / "tuned", c_grid=[0.1, 10.0],
         class_weight_grid=["none", "balanced"], max_iter=2000)
    rows = compare_runs(["baseline_C1", "tuned"], root=tmp_path, features=features)
    assert [r["run"] for r in rows] == ["baseline_C1", "tuned"]
    assert rows[0]["aspect_thresholds"] == "default_0.5"
    assert rows[1]["aspect_thresholds"] == "tuned"
    assert 0 <= rows[1]["aspect_hamming_loss"] <= 1