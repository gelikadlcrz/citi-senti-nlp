"""Logistic Regression pipeline: grouped nested ACD -> single-label ACSA.

This is a separate research protocol, NOT the historical target-conditioned
baseline. It deliberately refuses unreviewed conflicting (comment, category)
labels and missing shared group/fold metadata. Never reads Testing during fit.

Requires the shared TF-IDF team to expose create_vectorizer with keyword args:
    word_ngram_range, char_ngram_range, min_df
and return a fitted-on-demand sklearn transformer combining word and char TF-IDF.
The old word-only create_vectorizer does not satisfy this contract.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import sklearn
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from shared.data import WORKBOOK_DEFAULT, read_annotation_sheet
from shared.labels import ASPECT_CODES, ASPECT_TO_INDEX, SENTIMENT_LABELS, SENTIMENT_TO_INDEX, normalize_aspect
from models.logistic_regression.metrics import metrics as pair_metrics, group_bootstrap_ci

BUNDLE_FILE = "model_bundle.joblib"
RESOLUTION_FORMAT = "comment_id|aspect_code"
THRESHOLD_GRID = tuple(round(float(t), 2) for t in np.arange(.20, .801, .05))


@dataclass(frozen=True)
class Candidate:
    word_ngram_range: tuple[int, int]
    min_df: int
    C: float
    class_weight: str | None


def search_grid() -> list[Candidate]:
    """Paper Table 2: 2 word ranges * 2 min_df * 3 C * 2 weights = 24."""
    return [Candidate(word, df, c, weight)
            for word in ((1, 1), (1, 2)) for df in (1, 2)
            for c in (.1, 1.0, 10.0) for weight in (None, "balanced")]


def _factory(candidate: Candidate, feature_factory: Callable | None):
    if feature_factory is None:
        from shared.tfidf import create_vectorizer
        feature_factory = create_vectorizer
    try:
        vectorizer = feature_factory(
            word_ngram_range=candidate.word_ngram_range,
            char_ngram_range=(3, 5), min_df=candidate.min_df,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Awaiting the shared TF-IDF patch: shared.tfidf.create_vectorizer "
            "must accept word_ngram_range, char_ngram_range=(3,5), min_df. "
            "Do not run the word-only vectorizer and claim methodology alignment."
        ) from exc
    if not hasattr(vectorizer, 'fit_transform') or not hasattr(vectorizer, 'transform'):
        raise TypeError("Shared feature factory must return an sklearn-style transformer")
    # Verify TF-IDF design when the transformer exposes its parameters.
    params = vectorizer.get_params(deep=True) if hasattr(vectorizer, "get_params") else {}
    analyzers = [str(v).lower() for k, v in params.items() if k.endswith("analyzer")]
    if params and not (any(v == "word" for v in analyzers) and any(v.startswith("char") for v in analyzers)):
        raise ValueError("Shared feature factory must include both word and character TF-IDF")
    # L2-normalize the combined word+character feature vector (not merely
    # each component independently). This wrapper does not fit any shared data.
    return Pipeline([("shared_tfidf", vectorizer), ("combined_l2", Normalizer(copy=False))])


def category_interactions(x, category_index: np.ndarray, n_categories: int = len(ASPECT_CODES)):
    """[x, one-hot(category), one-hot(category) tensor x], sparse throughout."""
    x = sparse.csr_matrix(x)
    cat = np.asarray(category_index, dtype=int)
    n, d = x.shape
    if cat.shape != (n,) or np.any(cat < 0) or np.any(cat >= n_categories):
        raise ValueError("Each row needs one valid category index")
    onehot = sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), cat)),
                               shape=(n, n_categories))
    coo = x.tocoo()
    interaction = sparse.csr_matrix((coo.data, (coo.row, cat[coo.row] * d + coo.col)),
                                     shape=(n, n_categories * d))
    return sparse.hstack((x, onehot, interaction), format="csr")


class ConstantClassifier:
    """Deterministic prediction for ACD labels absent/constant in a training fold."""
    def __init__(self, value: int):
        self.value = int(value)

    def predict_proba(self, x):
        return np.full(x.shape[0], self.value, dtype=float)


class AspectBinaryRelevance:
    def __init__(self, *, C: float, class_weight: str | None, max_iter: int = 5000):
        self.C, self.class_weight, self.max_iter = C, class_weight, max_iter

    def fit(self, x, y):
        arr = np.asarray(y)
        if arr.ndim != 2 or arr.shape[1] != len(ASPECT_CODES):
            raise ValueError("ACD expects 11 binary targets")
        self.models_ = []
        self.unsupported_ = []
        for j, code in enumerate(ASPECT_CODES):
            classes = np.unique(arr[:, j])
            if len(classes) == 1:
                model = ConstantClassifier(int(classes[0]))
                self.unsupported_.append(code)
            else:
                model = LogisticRegression(C=self.C, class_weight=self.class_weight,
                                           max_iter=self.max_iter, random_state=42,
                                           solver="lbfgs")
                model.fit(x, arr[:, j])
            self.models_.append(model)
        return self

    def predict_proba(self, x):
        if not hasattr(self, "models_"):
            raise RuntimeError("Aspect model has not been fitted")
        return np.column_stack([
            model.predict_proba(x) if isinstance(model, ConstantClassifier)
            else model.predict_proba(x)[:, 1] for model in self.models_
        ])


def category_records(frame: pd.DataFrame, resolutions: dict[str, str] | None = None):
    """Exactly one sentiment per (comment, category); never silently vote.

    The resolution file must explicitly map ambiguous keys to a reviewed label.
    It does not alter the source workbook, and the chosen label must be one of
    the originally annotated labels for that pair.
    """
    resolutions = resolutions or {}
    output = []
    conflicts = {}
    used = set()
    for (cid, code), group in frame.groupby(["comment_id", "aspect_code"], sort=False):
        texts = group["model_text"].unique()
        if len(texts) != 1:
            raise ValueError(f"Inconsistent model_text for {cid}")
        values = sorted(group["sentiment"].unique().tolist())
        key = f"{cid}|{code}"
        if len(values) > 1:
            conflicts[key] = values
            chosen = resolutions.get(key)
            if chosen not in values:
                continue
            used.add(key)
        else:
            chosen = values[0]
        output.append((str(cid), str(code), str(texts[0]), SENTIMENT_TO_INDEX[chosen]))
    if conflicts and set(conflicts) != used:
        missing = {key: values for key, values in conflicts.items() if key not in used}
        raise ValueError("Unresolved comment–category sentiments: " + json.dumps(missing, ensure_ascii=False)
                         + ". Provide reviewed resolutions JSON mapping comment_id|aspect_code to one existing sentiment.")
    unknown = set(resolutions) - set(conflicts)
    if unknown:
        raise ValueError(f"Unused or invalid sentiment resolutions: {sorted(unknown)}")
    return output


def audit_conflicts(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {f"{cid}|{code}": sorted(group["sentiment"].unique().tolist())
            for (cid, code), group in frame.groupby(["comment_id", "aspect_code"], sort=False)
            if group["sentiment"].nunique() > 1}


def _read_resolutions(path: str | Path | None):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Resolutions file must be a JSON object")
    if any(value not in SENTIMENT_LABELS for value in data.values()):
        raise ValueError("Resolution values must be Negative, Neutral or Positive")
    return data


def _manifest(path: str | Path, dev: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Shared group manifest contract; no partitions are generated by LR."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing group-isolated split/fold manifest: {p}. "
            "The data team must supply comment_id,group_id,split,outer_fold."
        )
    rows = pd.read_csv(p, dtype=str, keep_default_na=False)
    required = {"comment_id", "group_id", "split", "outer_fold"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Group manifest lacks columns {sorted(required - set(rows.columns))}")
    if rows.comment_id.duplicated().any() or (rows[["comment_id", "group_id", "split"]] == "").any().any():
        raise ValueError("Group manifest must have unique comment IDs and nonblank group/split values")
    if set(rows["split"]) - {"development", "test"}:
        raise ValueError("Manifest split must be development or test")
    by_id = {row.comment_id: row._asdict() for row in rows.itertuples(index=False)}
    dev_ids = set(dev["comment_id"])
    if not dev_ids.issubset(by_id):
        raise ValueError(f"Missing development comment IDs in group manifest: {sorted(dev_ids - by_id)[:6]}")
    if any(by_id[cid]["split"] != "development" for cid in dev_ids):
        raise ValueError("Development workbook contains comments marked as test")
    if set(rows.loc[rows.split == "development", "comment_id"]) != dev_ids:
        raise ValueError("Manifest development IDs differ from Training+Validation IDs")
    group_splits = rows.groupby("group_id")["split"].nunique()
    if (group_splits > 1).any():
        raise ValueError("Merged source/near-duplicate groups cross development and test")
    folds = rows[rows.split == "development"]
    if set(folds.outer_fold) != set(map(str, range(5))):
        raise ValueError("Development manifest requires outer_fold values 0–4")
    if (folds.groupby("group_id")["outer_fold"].nunique() > 1).any():
        raise ValueError("One group is distributed across multiple outer folds")
    # If source_reference exists, it must not be split across different groups.
    if "source_reference" in dev.columns:
        sr = dev[["comment_id", "source_reference"]].drop_duplicates("comment_id")
        sr = sr[sr.source_reference.astype(str).str.strip() != ""]
        per_source: dict[str, set[str]] = {}
        for row in sr.itertuples(index=False):
            per_source.setdefault(str(row.source_reference), set()).add(str(by_id[str(row.comment_id)]["group_id"]))
        if any(len(group_ids) > 1 for group_ids in per_source.values()):
            raise ValueError("Source-thread references cross manifest groups; ask the data team to rebuild groups")
    return by_id


def load_development(*, workbook: str | Path, manifest: str | Path,
                     resolutions: str | Path | None = None):
    train = read_annotation_sheet(workbook, "Training")
    val = read_annotation_sheet(workbook, "Validation")
    dev = pd.concat([train, val], ignore_index=True)
    if (dev.groupby("comment_id")["model_text"].nunique() > 1).any():
        raise ValueError("Inconsistent comment text")
    meta = _manifest(manifest, dev)
    sentiment_rows = category_records(dev, _read_resolutions(resolutions))
    comments = dev.drop_duplicates("comment_id", keep="first").reset_index(drop=True)
    ids = comments["comment_id"].astype(str).tolist()
    x_text = comments["model_text"].astype(str).tolist()
    y = np.zeros((len(ids), len(ASPECT_CODES)), dtype=np.int8)
    i_by_id = {cid: i for i, cid in enumerate(ids)}
    for row in dev[["comment_id", "aspect_code"]].drop_duplicates().itertuples(index=False):
        y[i_by_id[str(row.comment_id)], ASPECT_TO_INDEX[str(row.aspect_code)]] = 1
    group = np.asarray([meta[cid]["group_id"] for cid in ids])
    outer = np.asarray([int(meta[cid]["outer_fold"]) for cid in ids])
    return {"ids": ids, "texts": x_text, "aspects": y, "groups": group,
            "outer": outer, "categories": sentiment_rows, "resolution_count": len(_read_resolutions(resolutions)),
            "primary_language": comments["primary_language"].astype(str).tolist(),
            "code_switched": comments["code_switched"].astype(str).tolist()}


def make_folds(indices: np.ndarray, groups: np.ndarray, n_splits: int = 3):
    indices = np.asarray(indices, dtype=int)
    if len(np.unique(groups[indices])) < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct training groups")
    folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, va in folds.split(indices, groups=groups[indices]):
        yield indices[tr], indices[va]


def _fit_predict(dev, train_idx, val_idx, *, task: str, candidate: Candidate,
                 feature_factory=None, max_iter: int = 5000):
    """Fold-local feature fitting; no test/validation text enters the vocabulary."""
    vec = _factory(candidate, feature_factory)
    x_train = vec.fit_transform([dev["texts"][i] for i in train_idx])
    x_val = vec.transform([dev["texts"][i] for i in val_idx])
    if task == "aspect":
        y_tr = dev["aspects"][train_idx]
        model = AspectBinaryRelevance(C=candidate.C, class_weight=candidate.class_weight, max_iter=max_iter)
        model.fit(x_train, y_tr)
        return model.predict_proba(x_val), model.unsupported_
    # Category instances are only those whose comment belongs to the respective fold.
    id_to_pos = {dev["ids"][idx]: pos for pos, idx in enumerate(train_idx)}
    id_to_vpos = {dev["ids"][idx]: pos for pos, idx in enumerate(val_idx)}
    tr_records = [r for r in dev["categories"] if r[0] in id_to_pos]
    va_records = [r for r in dev["categories"] if r[0] in id_to_vpos]
    tr_x = category_interactions(x_train[[id_to_pos[r[0]] for r in tr_records]],
                                 np.array([ASPECT_TO_INDEX[r[1]] for r in tr_records]))
    va_x = category_interactions(x_val[[id_to_vpos[r[0]] for r in va_records]],
                                 np.array([ASPECT_TO_INDEX[r[1]] for r in va_records]))
    labels = np.array([r[3] for r in tr_records])
    if len(np.unique(labels)) != 3:
        raise ValueError("Every ACSA training fold must contain all three sentiment classes")
    model = LogisticRegression(C=candidate.C, class_weight=candidate.class_weight,
                               max_iter=max_iter, random_state=42, solver="lbfgs")
    model.fit(tr_x, labels)
    prediction = model.predict(va_x)
    return ([(r[0], r[1], r[3], int(pred)) for r, pred in zip(va_records, prediction)], [] )


def select_thresholds(y: np.ndarray, probabilities: np.ndarray,
                      *, per_category: bool = True, unsupported: set[str] | None = None):
    """Global cutoff first; per-label cutoff only at >=20 pos AND >=20 neg."""
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != y.shape or y.ndim != 2 or y.shape[1] != len(ASPECT_CODES):
        raise ValueError("Threshold arrays must both have n×11 shape")
    scores = [(float(f1_score(y, probabilities >= t, average="macro", zero_division=0)), t)
              for t in THRESHOLD_GRID]
    global_t = sorted(scores, key=lambda pair: (-pair[0], abs(pair[1] - .5), pair[1]))[0][1]
    thresholds = {code: global_t for code in ASPECT_CODES}
    eligible = []
    unsupported = unsupported or set()
    for code in unsupported:
        thresholds[code] = .5
    if per_category:
        for j, code in enumerate(ASPECT_CODES):
            if code in unsupported:
                continue
            positives = int(y[:, j].sum())
            if positives < 20 or len(y) - positives < 20:
                continue
            eligible.append(code)
            local = [(float(f1_score(y[:, j], probabilities[:, j] >= t, zero_division=0)), t)
                     for t in THRESHOLD_GRID]
            thresholds[code] = sorted(local, key=lambda pair: (-pair[0], abs(pair[1] - global_t), pair[1]))[0][1]
    return thresholds, {"global": global_t, "eligible_categories": eligible,
                        "grid": list(THRESHOLD_GRID)}


def aspect_mask(probabilities: np.ndarray, thresholds: dict[str, float]):
    return (np.asarray(probabilities) >= np.array([thresholds[c] for c in ASPECT_CODES])).astype(np.int8)


def _candidate_oof(dev, indices, task, candidate, *, feature_factory=None, max_iter=5000):
    """One pooled inner-OOF prediction for each training instance."""
    if task == "aspect":
        probs = np.full((len(dev["ids"]), len(ASPECT_CODES)), np.nan)
        unsupported = []
        for tr, va in make_folds(indices, dev["groups"], 3):
            pred, missing = _fit_predict(dev, tr, va, task="aspect", candidate=candidate,
                                         feature_factory=feature_factory, max_iter=max_iter)
            probs[va] = pred
            unsupported.append({"fold": len(unsupported), "categories": missing})
        return probs[indices], dev["aspects"][indices], unsupported
    collected = []
    for tr, va in make_folds(indices, dev["groups"], 3):
        pred, _ = _fit_predict(dev, tr, va, task="sentiment", candidate=candidate,
                               feature_factory=feature_factory, max_iter=max_iter)
        collected.extend(pred)
    return collected, None, []


def choose_task(dev, indices, task, *, feature_factory=None, max_iter=5000):
    results = []
    for candidate in search_grid():
        oof, y, unsupported = _candidate_oof(dev, indices, task, candidate,
                                               feature_factory=feature_factory, max_iter=max_iter)
        if task == "aspect":
            missing_categories = {code for row in unsupported for code in row["categories"]}
            thresholds, rule = select_thresholds(y, oof, unsupported=missing_categories)
            f1 = f1_score(y, aspect_mask(oof, thresholds), average="macro", zero_division=0)
        else:
            f1 = f1_score([row[2] for row in oof], [row[3] for row in oof],
                          labels=list(range(3)), average="macro", zero_division=0)
            thresholds, rule = None, None
        results.append({"candidate": candidate, "inner_oof_macro_f1": float(f1),
                        "thresholds": thresholds, "threshold_rule": rule,
                        "unsupported_folds": unsupported})
    # Deterministic order for ties: Table 2 enumeration order.
    selected = max(enumerate(results), key=lambda ir: (ir[1]["inner_oof_macro_f1"], -ir[0]))[1]
    return selected, results


def fit_pair(dev, train_idx, *, aspect_choice, sentiment_choice,
             feature_factory=None, max_iter=5000):
    """Fit task-specific vocabularies only on given training comments."""
    models = {}
    for task, choice in (("aspect", aspect_choice), ("sentiment", sentiment_choice)):
        candidate = choice["candidate"]
        vec = _factory(candidate, feature_factory)
        x = vec.fit_transform([dev["texts"][i] for i in train_idx])
        if task == "aspect":
            model = AspectBinaryRelevance(C=candidate.C, class_weight=candidate.class_weight,
                                          max_iter=max_iter).fit(x, dev["aspects"][train_idx])
            models[task] = {"vectorizer": vec, "classifier": model,
                            "thresholds": choice["thresholds"],
                            "unsupported_categories": model.unsupported_}
        else:
            id_to_pos = {dev["ids"][idx]: pos for pos, idx in enumerate(train_idx)}
            recs = [r for r in dev["categories"] if r[0] in id_to_pos]
            features = category_interactions(x[[id_to_pos[r[0]] for r in recs]],
                                              np.array([ASPECT_TO_INDEX[r[1]] for r in recs]))
            y = np.asarray([r[3] for r in recs])
            if len(np.unique(y)) != 3:
                raise ValueError("All three sentiment classes are required for ACSA training")
            model = LogisticRegression(C=candidate.C, class_weight=candidate.class_weight,
                                       max_iter=max_iter, random_state=42, solver="lbfgs")
            model.fit(features, y)
            models[task] = {"vectorizer": vec, "classifier": model}
    return models


def predict_text(bundle, comment: str):
    """Exactly zero or one sentiment per predicted category; never target spans."""
    text = str(comment)
    if not text.strip():
        raise ValueError("Comment must not be blank")
    aspect = bundle["aspect"]
    proba = aspect["classifier"].predict_proba(aspect["vectorizer"].transform([text]))
    flags = aspect_mask(proba, aspect["thresholds"])[0]
    codes = [code for code, present in zip(ASPECT_CODES, flags) if present]
    if not codes:
        return []
    sentiment = bundle["sentiment"]
    x = sentiment["vectorizer"].transform([text] * len(codes))
    x = category_interactions(x, np.array([ASPECT_TO_INDEX[code] for code in codes]))
    predictions = sentiment["classifier"].predict(x)
    return [(code, SENTIMENT_LABELS[int(label)]) for code, label in zip(codes, predictions)]


def score_predictions(dev, indices, bundle):
    gold_by_id = {}
    for cid, code, _, sentiment in dev["categories"]:
        gold_by_id.setdefault(cid, set()).add((code, SENTIMENT_LABELS[sentiment]))
    gold = [gold_by_id.get(dev["ids"][i], set()) for i in indices]
    pred = [set(predict_text(bundle, dev["texts"][i])) for i in indices]
    scores = pair_metrics(gold, pred)
    # LR-only per-comment predictions for shared group-bootstrap/statistical team.
    rows = [{"comment_id": dev["ids"][i], "group_id": str(dev["groups"][i]),
             "gold_pairs": sorted([list(pair) for pair in g]),
             "predicted_pairs": sorted([list(pair) for pair in p])}
            for i, g, p in zip(indices, gold, pred)]
    return scores, rows




def score_components(dev, indices, bundle):
    """ACD and gold-category ACSA scores independently of cascading errors."""
    indices = np.asarray(indices)
    aspect = bundle["aspect"]
    texts = [dev["texts"][i] for i in indices]
    probabilities = aspect["classifier"].predict_proba(aspect["vectorizer"].transform(texts))
    predicted = aspect_mask(probabilities, aspect["thresholds"])
    gold = dev["aspects"][indices]
    acd = {
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(gold, predicted, average="micro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(gold, predicted)),
        "per_aspect": classification_report(gold, predicted, target_names=list(ASPECT_CODES),
                                             output_dict=True, zero_division=0),
    }
    sentiment = bundle["sentiment"]
    in_ids = {dev["ids"][i] for i in indices}
    records = [r for r in dev["categories"] if r[0] in in_ids]
    xs = sentiment["vectorizer"].transform([r[2] for r in records])
    feats = category_interactions(xs, np.array([ASPECT_TO_INDEX[r[1]] for r in records]))
    truth = np.array([r[3] for r in records])
    predictions = sentiment["classifier"].predict(feats)
    acsa = {
        "macro_f1": float(f1_score(truth, predictions, labels=[0,1,2], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "labels": list(SENTIMENT_LABELS),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0,1,2]).tolist(),
        "per_sentiment": classification_report(truth, predictions, labels=[0,1,2],
                    target_names=list(SENTIMENT_LABELS), output_dict=True, zero_division=0),
    }
    return {"aspect": acd, "conditional_category_sentiment": acsa}




def report_subgroups_and_errors(dev, predicted_rows):
    """Development OOF descriptive diagnostics, never city-population estimates."""
    by_id = {row["comment_id"]: row for row in predicted_rows}
    def _score(indices):
        gold = [set(map(tuple, by_id[dev["ids"][i]]["gold_pairs"])) for i in indices]
        pred = [set(map(tuple, by_id[dev["ids"][i]]["predicted_pairs"])) for i in indices]
        return {"n_comments": len(indices),
                "fixed_33_pair_macro_f1": pair_metrics(gold, pred)["macro_f1_33_pairs"]}
    language = {}
    for group in sorted(set(dev["primary_language"])):
        indices = [i for i, item in enumerate(dev["primary_language"]) if item == group]
        if indices:
            language[group or "Unspecified"] = _score(indices)
    code_switch = {}
    for group in sorted(set(dev["code_switched"])):
        indices = [i for i, item in enumerate(dev["code_switched"]) if item == group]
        if indices:
            code_switch[group or "Unspecified"] = _score(indices)
    counts = {"missed_aspect_categories": 0, "spurious_aspect_categories": 0,
              "wrong_sentiment_correct_category": 0}
    for row in predicted_rows:
        gold = set(map(tuple, row["gold_pairs"]))
        pred = set(map(tuple, row["predicted_pairs"]))
        gold_a = {code for code, _ in gold}
        pred_a = {code for code, _ in pred}
        counts["missed_aspect_categories"] += len(gold_a - pred_a)
        counts["spurious_aspect_categories"] += len(pred_a - gold_a)
        counts["wrong_sentiment_correct_category"] += sum(
            1 for code in gold_a & pred_a
            if {sent for c, sent in gold if c == code} != {sent for c, sent in pred if c == code}
        )
    return {"language": language, "code_switching": code_switch, "error_counts": counts}


def serializable_selection(result):
    """Keep model-selection evidence, not only the winning scores."""
    return {"candidate": asdict(result["candidate"]),
            "inner_oof_macro_f1": result["inner_oof_macro_f1"],
            "thresholds": result["thresholds"],
            "threshold_rule": result["threshold_rule"],
            "unsupported_folds": result["unsupported_folds"]}


def nested_grouped_run(*, workbook=WORKBOOK_DEFAULT,
                       manifest="data/splits/group_manifest.csv", resolutions=None,
                       output="models/logistic_regression/artifacts/grouped_cv",
                       feature_factory=None, max_iter=5000):
    """5 outer x 3 inner grouped CV; final 3-fold selection on full development.

    This can take substantial CPU time: 24 settings/task/inner procedure.
    It never opens Testing or uses old Training/Validation for model selection.
    """
    dev = load_development(workbook=workbook, manifest=manifest, resolutions=resolutions)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output} (no overwriting frozen artifacts)")
    pooled = []
    folds = []
    search_evidence = []
    for fold in range(5):
        tr = np.flatnonzero(dev["outer"] != fold)
        va = np.flatnonzero(dev["outer"] == fold)
        if not len(va):
            raise ValueError(f"Empty outer fold {fold}")
        if set(dev["groups"][tr]) & set(dev["groups"][va]):
            raise ValueError("Outer fold leaks training groups")
        chosen_a, candidates_a = choose_task(dev, tr, "aspect", feature_factory=feature_factory, max_iter=max_iter)
        chosen_s, candidates_s = choose_task(dev, tr, "sentiment", feature_factory=feature_factory, max_iter=max_iter)
        search_evidence.append({"outer_fold": fold,
                                "aspect": [serializable_selection(r) for r in candidates_a],
                                "sentiment": [serializable_selection(r) for r in candidates_s]})
        bundle = fit_pair(dev, tr, aspect_choice=chosen_a, sentiment_choice=chosen_s,
                          feature_factory=feature_factory, max_iter=max_iter)
        score, predictions = score_predictions(dev, va, bundle)
        components = score_components(dev, va, bundle)
        pooled.extend(predictions)
        folds.append({"fold": fold, "n_comments": len(va), "pair_metrics": score, "component_metrics": components,
                      "aspect_candidate": asdict(chosen_a["candidate"]),
                      "sentiment_candidate": asdict(chosen_s["candidate"]),
                      "unsupported_aspects": bundle["aspect"]["unsupported_categories"]})
    # Recreate *pooled* gold/pred arrays in original comment order.
    by_id = {r["comment_id"]: r for r in pooled}
    from models.logistic_regression.metrics import metrics as pair_metrics, group_bootstrap_ci
    gold = [set(map(tuple, by_id[cid]["gold_pairs"])) for cid in dev["ids"]]
    pred = [set(map(tuple, by_id[cid]["predicted_pairs"])) for cid in dev["ids"]]
    pooled_metrics = pair_metrics(gold, pred)
    pooled_diagnostics = report_subgroups_and_errors(dev, pooled)
    pooled_ci = group_bootstrap_ci(gold, pred, dev["groups"], replicates=10000)
    indices = np.arange(len(dev["ids"]))
    chosen_a, result_a = choose_task(dev, indices, "aspect", feature_factory=feature_factory, max_iter=max_iter)
    chosen_s, result_s = choose_task(dev, indices, "sentiment", feature_factory=feature_factory, max_iter=max_iter)
    final = fit_pair(dev, indices, aspect_choice=chosen_a, sentiment_choice=chosen_s,
                     feature_factory=feature_factory, max_iter=max_iter)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, output / BUNDLE_FILE)
    search_evidence.append({"final_development_3_folds": True,
                           "aspect": [serializable_selection(r) for r in result_a],
                           "sentiment": [serializable_selection(r) for r in result_s]})
    (output / "selection_grid_results.json").write_text(
        json.dumps(search_evidence, indent=2) + "\n", encoding="utf-8")
    report = {"protocol": "nested_group_5_outer_3_inner_plus_final_3_fold",
              "test_used_by_this_procedure": False,
              "prior_test_access_requires_disclosure": True, "development_comments": len(dev["ids"]),
              "primary_metric": "fixed_33_pair_macro_f1",
              "pooled_outer_metrics": pooled_metrics, "pooled_outer_group_bootstrap": pooled_ci,
              "development_subgroups_and_errors": pooled_diagnostics, "outer_folds": folds,
              "final_selections": {
                  "aspect": {"candidate": asdict(chosen_a["candidate"]),
                             "thresholds": chosen_a["thresholds"],
                             "threshold_rule": chosen_a["threshold_rule"]},
                  "sentiment": {"candidate": asdict(chosen_s["candidate"])}},
              "resolution_count": dev["resolution_count"],
              "source_workbook": str(workbook), "source_group_manifest": str(manifest),
              "random_seed": 42, "sklearn_version": sklearn.__version__,
              "tfidf_factory_contract": "shared.tfidf.create_vectorizer(word_ngram_range, char_ngram_range=(3,5), min_df)"}
    (output / "development_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "outer_oof_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in pooled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Pooled outer end-to-end macro-F1 (33 labels):", pooled_metrics["macro_f1_33_pairs"])
    print("Saved frozen development models and report to", output)
    return report


def audit(*, workbook=WORKBOOK_DEFAULT, resolutions=None, output=None):
    """Print development conflicts without reading held-out Testing."""
    frame = pd.concat([read_annotation_sheet(workbook, "Training"),
                       read_annotation_sheet(workbook, "Validation")], ignore_index=True)
    conflicts = audit_conflicts(frame)
    report = {"conflict_count": len(conflicts), "conflicts": conflicts,
              "resolution_template": {key: None for key in conflicts}}
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", action="store_true", help="Report conflicting development labels only")
    p.add_argument("--workbook", default=str(WORKBOOK_DEFAULT))
    p.add_argument("--manifest", default="data/splits/group_manifest.csv")
    p.add_argument("--resolutions", help="Human-reviewed JSON: comment_id|aspect_code -> sentiment")
    p.add_argument("--output", default="models/logistic_regression/artifacts/grouped_cv")
    p.add_argument("--audit-output", help="Optional JSON conflict report path")
    args = p.parse_args(argv)
    try:
        if args.audit:
            out = audit(workbook=args.workbook, output=args.audit_output)
        else:
            out = nested_grouped_run(workbook=args.workbook, manifest=args.manifest,
                                     resolutions=args.resolutions, output=args.output)
    except (ValueError, FileNotFoundError, RuntimeError, FileExistsError) as exc:
        p.exit(1, f"LR grouped training stopped: {exc}\n")
    print(json.dumps(out if args.audit else {"primary": out["pooled_outer_metrics"]}, indent=2))
    return out


if __name__ == "__main__":
    main()
