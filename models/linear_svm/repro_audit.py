"""Read-only audit for the accepted Approach A training artifacts.

python -m models.linear_svm.repro_audit --artifacts models/linear_svm/artifacts/target_absa
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import joblib
from models.linear_svm.experiment_target_absa import AspectSVM  # noqa: F401; supports old pickle class refs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(artifacts: Path, output: Path | None = None):
    artifacts = Path(artifacts)
    required = ['aspect_bundle.joblib', 'sentiment_bundle.joblib', 'report.json']
    for filename in required:
        if not (artifacts / filename).is_file():
            raise FileNotFoundError(artifacts / filename)
    report = json.loads((artifacts / 'report.json').read_text(encoding='utf-8'))
    aspect = joblib.load(artifacts / 'aspect_bundle.joblib')
    sentiment = joblib.load(artifacts / 'sentiment_bundle.joblib')
    if aspect.get('task') != 'aspect' or sentiment.get('task') != 'sentiment':
        raise ValueError('Unexpected saved bundle task fields')
    from sklearn.pipeline import FeatureUnion
    for name, bundle in [('aspect', aspect), ('sentiment', sentiment)]:
        vec = bundle.get('vectorizer')
        if not isinstance(vec, FeatureUnion):
            raise TypeError(f'{name} TF-IDF is not FeatureUnion')
        parts = dict(vec.transformer_list)
        if set(parts) != {'word', 'char'}:
            raise ValueError(f'{name} expected word+char vectorizers')
        if parts['word'].ngram_range != (1, 2) or parts['char'].ngram_range != (3, 5):
            raise ValueError(f'{name} ngram settings do not match accepted configuration')
    libs = {}
    for name in ['numpy', 'scipy', 'scikit-learn', 'joblib', 'pandas', 'openpyxl']:
        try:
            libs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libs[name] = None
    audit = {
        'source': 'Existing Approach A artifacts: no retraining or Testing access',
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'package_versions': libs,
        'file_sha256': {name: sha256(artifacts / name) for name in required},
        'saved_configs': {'aspect': aspect.get('config'), 'sentiment': sentiment.get('config')},
        'validation_macro_f1': {task: report['tasks'][task]['validation']['macro_f1']
                                for task in ('aspect', 'sentiment')},
        'nested_cv_macro_f1': {task: report['tasks'][task]['outer_macro_f1_mean']
                              for task in ('aspect', 'sentiment')},
        'test_used': report.get('test_used'),
        'warning': 'This audits artifacts, not CV fold-level vocabulary isolation or external group-ID correctness.'
    }
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', default='models/linear_svm/artifacts/target_absa')
    p.add_argument('--output', default='results/svm_approach_a_repro_audit.json')
    args = p.parse_args()
    run(Path(args.artifacts), Path(args.output))


if __name__ == '__main__':
    main()
