"""Shared TF-IDF vectorizer definitions and artifact loading.

Only the training set may be passed to .fit(). Both baseline algorithms must
load the resulting .joblib files (or identical exported feature matrices).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ASPECT_VECTORIZER_FILE = "aspect_vectorizer.joblib"
SENTIMENT_VECTORIZER_FILE = "sentiment_vectorizer.joblib"
METADATA_FILE = "metadata.json"


def create_vectorizer(*, max_features: int | None = None) -> TfidfVectorizer:
    """Same initial word unigram/bigram TF-IDF settings for both baselines.

    Keep single-character words and Unicode tokens; do not apply English-only
    stopword removal, stemming, or translation to multilingual comments.
    """
    return TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        max_features=max_features,
        dtype=np.float32,
    )


def save_vectorizers(
    output_dir: str | Path,
    aspect_vectorizer: TfidfVectorizer,
    sentiment_vectorizer: TfidfVectorizer,
    metadata: dict[str, Any],
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(aspect_vectorizer, directory / ASPECT_VECTORIZER_FILE)
    joblib.dump(sentiment_vectorizer, directory / SENTIMENT_VECTORIZER_FILE)
    (directory / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_aspect_vectorizer(output_dir: str | Path = "models/tfidf") -> TfidfVectorizer:
    """Load a trusted vectorizer saved by the shared training script."""
    return joblib.load(Path(output_dir) / ASPECT_VECTORIZER_FILE)


def load_sentiment_vectorizer(output_dir: str | Path = "models/tfidf") -> TfidfVectorizer:
    """Load a trusted vectorizer saved by the shared training script."""
    return joblib.load(Path(output_dir) / SENTIMENT_VECTORIZER_FILE)
