"""Choose/freeze a final ACSA checkpoint USING VALIDATION ONLY (never Testing).

Include the already-trained V1 baseline as a candidate to prevent blindly
replacing a stronger model with a shorter experiment. This script does NOT
open any split workbook or load any Testing labels.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_candidate(checkpoint: Path, name: str) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    result_path = checkpoint / 'acsa_results.json'
    if not result_path.is_file():
        raise FileNotFoundError(f'Missing checkpoint results: {result_path}')
    report = json.loads(result_path.read_text(encoding='utf-8'))
    for expected in ('validation', 'data_fingerprint', 'label_order', 'aspect_codes'):
        if expected not in report:
            raise ValueError(f'{result_path} missing {expected}')
    if not (checkpoint / 'config.json').is_file():
        raise FileNotFoundError(f'Missing Hugging Face checkpoint config in {checkpoint}')
    weights = ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json')
    if not any((checkpoint / f).is_file() for f in weights):
        raise FileNotFoundError(f'No model checkpoint weight file in {checkpoint}')
    return {
        'name': name,
        'checkpoint_directory': str(checkpoint),
        'validation_macro_f1': float(report['validation']['macro_f1']),
        'validation_accuracy': float(report['validation']['accuracy']),
        'mixed_sentiment_validation_macro_f1': (
            report.get('validation_aspect_conditioning', {})
            .get('mixed_sentiment', {}).get('macro_f1')),
        'data_fingerprint': report['data_fingerprint'],
        'label_order': report['label_order'],
        'aspect_codes': report['aspect_codes'],
    }


def freeze(baseline: Path, search_root: Path, output: Path) -> dict:
    baseline_candidate = read_candidate(baseline, 'baseline_v1')
    candidates = [baseline_candidate]
    if search_root.exists():
        for result_file in sorted(search_root.glob('trial_*/acsa_results.json')):
            candidates.append(read_candidate(result_file.parent, result_file.parent.name))
    for item in candidates[1:]:
        if item['data_fingerprint'] != baseline_candidate['data_fingerprint']:
            raise ValueError(f'Workbook fingerprint differs for {item["name"]}; cannot rank fairly')
        if item['label_order'] != baseline_candidate['label_order']:
            raise ValueError(f'Sentiment label order differs for {item["name"]}')
        if item['aspect_codes'] != baseline_candidate['aspect_codes']:
            raise ValueError(f'Aspect category mapping differs for {item["name"]}')
    best = max(candidates, key=lambda item: (item['validation_macro_f1'],
                                             item['validation_accuracy']))
    manifest = {
        'selection_policy': 'Highest Validation three-class Macro-F1, accuracy tie-break',
        'selected_checkpoint': best['checkpoint_directory'],
        'selected_name': best['name'],
        'selected_validation_macro_f1': best['validation_macro_f1'],
        'data_fingerprint': best['data_fingerprint'],
        'candidates': candidates,
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'warning': 'Selection uses Validation ONLY. Run Testing once; never retune from Test results.',
    }
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Selection already frozen: {output}. Do not silently replace it.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Selected: {best["name"]} | Validation Macro-F1={best["validation_macro_f1"]:.4f}')
    print(f'Frozen selection: {output}')
    return manifest


def main():
    p = argparse.ArgumentParser(description='Freeze final ACSA checkpoint using Validation only')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--search-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    freeze(args.baseline, args.search_root, args.output)


if __name__ == '__main__':
    main()
