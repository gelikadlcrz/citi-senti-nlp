"""Prediction for CITI-Senti Approach A; targets MUST be supplied by caller.

Example:
python -m models.linear_svm.predict_target_absa --text 'Mahal ang parking fee.' --target 'parking fee' --aspect-code A03
"""
from models.linear_svm.experiment_target_absa import AspectSVM
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from shared.labels import ASPECT_CODES, ASPECT_TO_INDEX, SENTIMENT_TO_INDEX
from models.linear_svm.experiment_target_absa import features, margins


def predict(text, targets, artifacts):
    """targets: list of {aspect_code, aspect_text}; never fabricate target spans."""
    artifacts = Path(artifacts)
    aspect = joblib.load(artifacts / 'aspect_bundle.joblib')
    sentiment = joblib.load(artifacts / 'sentiment_bundle.joblib')
    aspect_data = {'text': [text], 'category': None, 'target': None}
    scores = margins(aspect, aspect_data)[0]
    cuts = np.asarray(aspect['thresholds'], dtype=float)
    detected = [code for code, score, cut in zip(ASPECT_CODES, scores, cuts) if score >= cut]
    # Note: target sentiment is always based on the supplied category and span,
    # even if the category detector omitted it. Report category-detection status.
    result = []
    labels = {int(v): k for k, v in SENTIMENT_TO_INDEX.items()}
    for item in targets:
        code = str(item['aspect_code']).strip()
        target = str(item['aspect_text']).strip()
        if code not in ASPECT_TO_INDEX:
            raise ValueError(f'Unknown aspect_code: {code}')
        if not target:
            raise ValueError(f'Missing aspect_text for {code}')
        data = {'text': [text], 'category': np.asarray([ASPECT_TO_INDEX[code]]), 'target': [target]}
        x = features(sentiment['vectorizer'], data, 'sentiment')
        y = int(sentiment['model'].predict(x)[0])
        result.append({'aspect_code': code, 'aspect_text': target, 'sentiment': labels[y],
                       'category_detected': code in detected})
    return {'detected_categories': detected, 'target_sentiments': result,
            'automatic_target_extraction': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', required=True)
    p.add_argument('--target', required=True, help='Explicit text target from the comment')
    p.add_argument('--aspect-code', required=True, choices=ASPECT_CODES)
    p.add_argument('--artifacts', default='models/linear_svm/artifacts/target_absa')
    args = p.parse_args()
    print(json.dumps(predict(args.text, [{'aspect_code': args.aspect_code, 'aspect_text': args.target}], args.artifacts),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
