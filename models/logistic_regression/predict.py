"""Logistic Regression inference for citizen feedback.

Sentence-only category sentiments (requires the optional category head):
    python -m models.logistic_regression.category_sentiment --artifacts models/logistic_regression/artifacts/tuned
    python -m models.logistic_regression.predict --artifacts models/logistic_regression/artifacts/tuned \\
        --text 'Mahal ang parking fee pero maganda ang bagong patakaran.'

More specific annotated/supplied targets (original benchmark model):
    python -m models.logistic_regression.predict --text 'Mahal ang parking fee.' \\
        --aspect-code A03 --aspect-text 'parking fee'
    python -m models.logistic_regression.predict --text '...' \\
        --targets-json '[{"aspect_code":"A03","aspect_text":"parking fee"}]'

Category-level outputs do not identify target spans and must not be reported as
results of the target-conditioned sentiment benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from models.logistic_regression.category_sentiment import (
    MODEL_FILE, make_category_input, select_sentiments,
)
from models.logistic_regression.thresholds import (
    apply_thresholds, aspect_probabilities, load_thresholds,
)
from shared.data import make_sentiment_input
from shared.labels import ASPECT_CODES, ASPECT_NAMES, SENTIMENT_LABELS, normalize_aspect
from shared.tfidf import load_aspect_vectorizer, load_sentiment_vectorizer


class LogisticRegressionPredictor:
    """Load original LR and frozen shared TF-IDF; optionally load category head."""

    def __init__(
        self,
        *,
        artifacts: str | Path = "models/logistic_regression/artifacts/tuned",
        features: str | Path = "models/tfidf",
    ) -> None:
        artifacts_dir = Path(artifacts)
        self.artifacts_dir = artifacts_dir
        self.aspect_model = joblib.load(artifacts_dir / "aspect_model.joblib")
        self.sentiment_model = joblib.load(artifacts_dir / "sentiment_model.joblib")
        self.aspect_vectorizer = load_aspect_vectorizer(features)
        self.sentiment_vectorizer = load_sentiment_vectorizer(features)
        self.aspect_thresholds = load_thresholds(artifacts_dir)
        self.category_sentiment_model = None
        if self.aspect_model.n_features_in_ != len(self.aspect_vectorizer.vocabulary_):
            raise ValueError("Aspect classifier does not match the shared aspect TF-IDF vocabulary")
        if self.sentiment_model.n_features_in_ != len(self.sentiment_vectorizer.vocabulary_):
            raise ValueError("Sentiment classifier does not match the shared sentiment TF-IDF vocabulary")

    def predict_aspects(self, model_text: str) -> list[dict[str, str]]:
        if not str(model_text).strip():
            raise ValueError("model_text must not be blank")
        x = self.aspect_vectorizer.transform([model_text])
        if self.aspect_thresholds:
            prediction = apply_thresholds(aspect_probabilities(self.aspect_model, x), self.aspect_thresholds)
        else:
            prediction = self.aspect_model.predict(x)
        if prediction.shape != (1, len(ASPECT_CODES)):
            raise ValueError("Aspect model does not produce the expected 11-label output")
        return [
            {"code": code, "name": ASPECT_NAMES[code]}
            for code, selected in zip(ASPECT_CODES, prediction[0]) if selected
        ]

    def predict_sentiment(self, *, model_text: str, aspect_code: str, aspect_text: str) -> str:
        code = normalize_aspect(aspect_code)
        supplied_input = make_sentiment_input(model_text, code, aspect_text)
        x = self.sentiment_vectorizer.transform([supplied_input])
        index = int(self.sentiment_model.predict(x)[0])
        if index not in range(len(SENTIMENT_LABELS)):
            raise ValueError("Classifier returned an unknown sentiment label index")
        return SENTIMENT_LABELS[index]

    def predict_targets(self, model_text: str, targets: list[dict[str, str]]) -> dict:
        """Preserve each supplied target, including opposing sentiments in one category."""
        if not str(model_text).strip():
            raise ValueError("model_text must not be blank")
        if not isinstance(targets, list) or not targets:
            raise ValueError("targets must be a nonempty list of aspect_code/aspect_text pairs")
        result = []
        for target in targets:
            if not isinstance(target, dict) or "aspect_code" not in target or "aspect_text" not in target:
                raise ValueError("Each target needs aspect_code and aspect_text")
            code = normalize_aspect(target["aspect_code"])
            phrase = str(target["aspect_text"]).strip()
            result.append({
                "aspect_code": code,
                "aspect_name": ASPECT_NAMES[code],
                "aspect_text": phrase,
                "sentiment": self.predict_sentiment(
                    model_text=model_text, aspect_code=code, aspect_text=phrase,
                ),
                "target_source": "supplied_by_caller",
            })
        return {"aspects": self.predict_aspects(model_text), "target_sentiments": result,
                "sentiment_scope": "target_conditioned"}

    def _get_category_model(self):
        if self.category_sentiment_model is None:
            path = self.artifacts_dir / MODEL_FILE
            if not path.is_file():
                raise FileNotFoundError(
                    f"Sentence-only sentiment needs {path}; run "
                    "python -m models.logistic_regression.category_sentiment first. "
                    "The original sentiment classifier requires an aspect target."
                )
            self.category_sentiment_model = joblib.load(path)
            if self.category_sentiment_model.n_features_in_ != len(self.sentiment_vectorizer.vocabulary_):
                raise ValueError("Category sentiment classifier does not match shared TF-IDF")
        return self.category_sentiment_model

    def predict_comment(self, model_text: str) -> dict:
        """Website-friendly sentence-only prediction: zero or more aspect categories.

        Category sentiments are from a SEPARATELY trained LR head, not the
        annotated-target sentiment classifier. No extracted target text is claimed.
        """
        # Check availability even if the aspect model would predict nothing:
        # a missing sentiment head must not be silently mistaken for a working pipeline.
        model = self._get_category_model()
        aspects = self.predict_aspects(model_text)
        if not aspects:
            return {
                "aspects": [], "sentiment_scope": "category_level",
                "note": "No aspect category met its detection threshold; no sentiment was assigned.",
            }
        x = self.sentiment_vectorizer.transform([
            make_category_input(model_text, aspect["code"]) for aspect in aspects
        ])
        pred = select_sentiments(np.asarray(model.predict_proba(x)))
        result = []
        for aspect, values in zip(aspects, pred):
            result.append({
                **aspect,
                "sentiments": [name for name, flag in zip(SENTIMENT_LABELS, values) if flag],
                "target_text": None,
                "sentiment_scope": "category_level",
            })
        return {
            "aspects": result,
            "sentiment_scope": "category_level",
            "note": "Multiple aspect categories and multiple sentiments are possible. "
                    "No target phrases were extracted. Do not compare these predictions "
                    "to target-conditioned sentiment benchmark scores.",
        }


def predict_frozen_comment(text: str, *, artifacts: str | Path) -> dict:
    """Predict zero or more (category, sentiment) pairs using a frozen LR bundle."""
    from models.logistic_regression.pipeline import BUNDLE_FILE, predict_text
    path = Path(artifacts) / BUNDLE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved LR model bundle: {path}")
    from shared.labels import ASPECT_NAMES
    # Load only artifacts produced by a trusted project environment.
    bundle = joblib.load(path)
    return {
        "pairs": [
            {"aspect_code": code, "aspect_name": ASPECT_NAMES[code], "sentiment": sentiment}
            for code, sentiment in predict_text(bundle, text)
        ],
        "sentiment_scope": "one_sentiment_per_category",
        "aspect_target_extraction": False,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="models/logistic_regression/artifacts/tuned")
    parser.add_argument("--features", default="models/tfidf")
    parser.add_argument("--text", required=True, help="One citizen feedback entry")
    parser.add_argument("--aspect-code", help="A01–A11, for a supplied target")
    parser.add_argument("--aspect-text", help="Exact supplied target phrase")
    parser.add_argument("--targets-json", help="JSON list of supplied aspect_code/aspect_text pairs")
    args = parser.parse_args(argv)
    if (args.aspect_code is None) != (args.aspect_text is None):
        parser.error("Supply --aspect-code AND --aspect-text together")
    if args.targets_json is not None and args.aspect_code is not None:
        parser.error("Use --targets-json OR --aspect-code/--aspect-text, not both")
    try:
        if (Path(args.artifacts) / "model_bundle.joblib").is_file():
            if args.targets_json is not None or args.aspect_code is not None:
                raise ValueError("Grouped CV model accepts text only; sentiment is category-conditioned")
            output = predict_frozen_comment(args.text, artifacts=args.artifacts)
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return output
        predictor = LogisticRegressionPredictor(artifacts=args.artifacts, features=args.features)
        if args.targets_json is not None:
            targets = json.loads(args.targets_json)
            output = predictor.predict_targets(args.text, targets)
        elif args.aspect_code is not None:
            output = {
                "aspects": predictor.predict_aspects(args.text),
                "sentiment": {
                    "aspect_code": normalize_aspect(args.aspect_code),
                    "aspect_text": args.aspect_text,
                    "label": predictor.predict_sentiment(
                        model_text=args.text, aspect_code=args.aspect_code,
                        aspect_text=args.aspect_text,
                    ),
                    "target_source": "supplied_by_caller",
                },
            }
        else:
            output = predictor.predict_comment(args.text)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
