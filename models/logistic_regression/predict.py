"""Prediction entry point for CITI-Senti Logistic Regression (starter version).

From the project root after training:
    python -m models.logistic_regression.predict \
        --artifacts models/logistic_regression/artifacts/baseline_C1 \
        --text 'Mahal ang parking fee.' \
        --aspect-code A03 --aspect-text 'parking fee'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from shared.data import make_sentiment_input
from shared.labels import ASPECT_CODES, ASPECT_NAMES, SENTIMENT_LABELS, normalize_aspect
from shared.tfidf import load_aspect_vectorizer, load_sentiment_vectorizer


class LogisticRegressionPredictor:
    """Load trained models and frozen TF-IDF vectorizers once per process."""

    def __init__(
        self,
        *,
        artifacts: str | Path = "models/logistic_regression/artifacts",
        features: str | Path = "models/tfidf",
    ) -> None:
        artifacts_dir = Path(artifacts)
        self.aspect_model = joblib.load(artifacts_dir / "aspect_model.joblib")
        self.sentiment_model = joblib.load(artifacts_dir / "sentiment_model.joblib")
        self.aspect_vectorizer = load_aspect_vectorizer(features)
        self.sentiment_vectorizer = load_sentiment_vectorizer(features)
        if self.aspect_model.n_features_in_ != len(self.aspect_vectorizer.vocabulary_):
            raise ValueError("Aspect classifier does not match the shared aspect TF-IDF vocabulary")
        if self.sentiment_model.n_features_in_ != len(self.sentiment_vectorizer.vocabulary_):
            raise ValueError("Sentiment classifier does not match the shared sentiment TF-IDF vocabulary")

    def predict_aspects(self, model_text: str) -> list[dict[str, str]]:
        if not str(model_text).strip():
            raise ValueError("model_text must not be blank")
        x = self.aspect_vectorizer.transform([model_text])
        prediction = self.aspect_model.predict(x)
        if prediction.shape != (1, len(ASPECT_CODES)):
            raise ValueError("Aspect model does not produce the expected 11-label output")
        return [
            {"code": code, "name": ASPECT_NAMES[code]}
            for code, selected in zip(ASPECT_CODES, prediction[0]) if selected
        ]

    def predict_sentiment(
        self,
        *,
        model_text: str,
        aspect_code: str,
        aspect_text: str,
    ) -> str:
        code = normalize_aspect(aspect_code)
        supplied_input = make_sentiment_input(model_text, code, aspect_text)
        x = self.sentiment_vectorizer.transform([supplied_input])
        index = int(self.sentiment_model.predict(x)[0])
        if index not in range(len(SENTIMENT_LABELS)):
            raise ValueError("Classifier returned an unknown sentiment label index")
        return SENTIMENT_LABELS[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="models/logistic_regression/artifacts")
    parser.add_argument("--features", default="models/tfidf")
    parser.add_argument("--text", required=True, help="One citizen feedback entry")
    parser.add_argument("--aspect-code", help="A01-A11; required for target-conditioned sentiment")
    parser.add_argument("--aspect-text", help="Target phrase from the comment; required with --aspect-code")
    args = parser.parse_args()
    if (args.aspect_code is None) != (args.aspect_text is None):
        parser.error("Supply --aspect-code AND --aspect-text together for sentiment")
    try:
        predictor = LogisticRegressionPredictor(artifacts=args.artifacts, features=args.features)
        output: dict = {
            "aspects": predictor.predict_aspects(args.text),
            "note": "Aspect predictions are categories, not extracted target phrases.",
        }
        if args.aspect_code is not None:
            output["sentiment"] = {
                "aspect_code": normalize_aspect(args.aspect_code),
                "aspect_text": args.aspect_text,
                "label": predictor.predict_sentiment(
                    model_text=args.text,
                    aspect_code=args.aspect_code,
                    aspect_text=args.aspect_text,
                ),
                "target_source": "supplied_by_caller",
            }
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
