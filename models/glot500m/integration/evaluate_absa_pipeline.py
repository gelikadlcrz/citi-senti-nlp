"""CITI-Senti read-only evaluation: ACD, conditional ACSA, end-to-end ABSA.

Default --mode all prints three independent metrics using the same split:
  ACD: multilabel 11-category F1 from all comments.
  ACSA: 3-class sentiment F1, conditioned on GOLD-present categories.
  End-to-end ABSA: exact (category, sentiment) pairs using ACD predictions.

No training, threshold tuning, or result files. Previously visited Testing data must
not be used for checkpoint selection or described as a fresh independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from models.glot500m.integration.absa_runtime import (
    ASPECT_NAMES, CODES, DEFAULT_WORKBOOK, SENTIMENTS, infer_acd, infer_acsa,
    load_acd, load_acsa, load_json, read_acd_report, release_handle,
    require_dir, select_device,
)
from shared.data import make_aspect_task, read_annotation_sheet


def file_sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def validate_workbook_fingerprints(workbook: Path, acd_dir: Path | None,
                                   acsa_dir: Path | None, allow_change: bool) -> None:
    actual = file_sha256(workbook)
    entries = []
    if acd_dir is not None:
        report = read_acd_report(acd_dir)
        entries.append(("ACD", report.get("data_fingerprint")))
    if acsa_dir is not None:
        report = load_json(acsa_dir / "acsa_results.json")
        entries.append(("ACSA", report.get("data_fingerprint")))
    for name, saved in entries:
        if saved and saved != actual:
            message = (
                f"{name}: workbook SHA-256 differs from the one recorded during training. "
                "Verify whether Training/Validation/Testing splits or annotations changed."
            )
            if not allow_change:
                raise ValueError(message + " Use --allow-data-change only after reviewing it.")
            print("WARNING:", message, flush=True)
        if not saved:
            print(f"NOTE: {name} checkpoint has no workbook fingerprint; split identity cannot be verified.")


def read_acd_gold(workbook: Path, sheet: str):
    """Load comment-level 11-label gold matrix from the team's ACD data helper."""
    task = make_aspect_task(read_annotation_sheet(workbook, sheet))
    ids = [str(row["comment_id"]).strip() for row in task.row_metadata]
    texts = [str(text).strip() for text in task.texts]
    gold = np.asarray(task.labels, dtype=np.int8)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("ACD evaluation requires unique, nonempty comment IDs")
    if not all(texts):
        raise ValueError("ACD gold set has an empty comment")
    if gold.shape != (len(ids), len(CODES)) or not np.isin(gold, [0, 1]).all():
        raise ValueError(f"ACD gold matrix should have shape (comments, {len(CODES)}) of binary labels")
    return ids, texts, gold


def read_acsa_gold(workbook: Path, sheet: str, on_conflict: str, on_missing: str):
    """Return uniquely labeled gold comment-category pairs and exclusions.

    A conflicting pair is never assigned an arbitrary sentiment. If dropped,
    its whole comment is later excluded from end-to-end evaluation, avoiding
    incorrect treatment of that pair as an absent category.
    """
    frame = pd.read_excel(workbook, sheet_name=sheet, dtype=str, keep_default_na=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    required = {"comment_id", "model_text", "sentiment"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{sheet}: missing columns: {sorted(required - set(frame.columns))}")
    if "aspect_code" not in frame.columns and "aspect_category" not in frame.columns:
        raise ValueError(f"{sheet}: expected aspect_code or aspect_category column")

    import re
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    text_by_id: dict[str, str] = {}
    bad_pairs: set[tuple[str, str]] = set()
    duplicate_rows = 0
    structurally_invalid = 0
    for index, row in enumerate(frame.to_dict("records"), start=2):
        comment_id = str(row.get("comment_id", "")).strip()
        text = str(row.get("model_text", "")).strip()
        code = str(row.get("aspect_code", "")).strip().upper()
        if not code:
            found = re.match(r"^(A\d{2})\b", str(row.get("aspect_category", "")).strip(), re.I)
            code = found.group(1).upper() if found else ""
        if not comment_id or not text or code not in CODES:
            structurally_invalid += 1
            continue
        if comment_id in text_by_id and text_by_id[comment_id] != text:
            raise ValueError(f"{sheet}: inconsistent text for comment {comment_id}")
        text_by_id[comment_id] = text
        pair = (comment_id, code)
        sentiment = str(row.get("sentiment", "")).strip().lower()
        if sentiment not in (s.lower() for s in SENTIMENTS):
            bad_pairs.add(pair)
            continue
        if sentiment in pairs[pair]:
            duplicate_rows += 1
        pairs[pair].add(sentiment)
    if structurally_invalid:
        raise ValueError(f"{sheet}: {structurally_invalid} structurally invalid rows (missing ID/text/aspect)")

    conflicts = {pair for pair, values in pairs.items() if len(values) > 1}
    missing = bad_pairs
    if conflicts and on_conflict == "error":
        raise ValueError(
            f"{sheet}: {len(conflicts)} conflicting comment-category sentiment pairs; "
            f"examples {sorted(conflicts)[:5]}. Use --on-conflict drop to exclude them."
        )
    if missing and on_missing == "error":
        raise ValueError(
            f"{sheet}: {len(missing)} comment-category pairs have missing/invalid sentiments. "
            "Use --on-missing drop to exclude them."
        )
    valid = {}
    for pair, labels in pairs.items():
        if pair in conflicts or pair in missing:
            continue
        if len(labels) == 1:
            valid[pair] = SENTIMENTS.index(next(s for s in SENTIMENTS if s.lower() in labels))
    excluded = conflicts | missing
    return valid, text_by_id, excluded, {
        "duplicate_same_label_rows": duplicate_rows,
        "conflicting_pairs_dropped": len(conflicts),
        "missing_or_invalid_sentiment_pairs": len(missing),
        "valid_gold_pairs": len(valid),
    }


def print_acd_metrics(gold: np.ndarray, pred: np.ndarray,
                      model_order: tuple[str, ...], thresholds: np.ndarray) -> dict:
    print("\n" + "=" * 76)
    print("ACD — MULTI-LABEL ASPECT CATEGORY DETECTION")
    print("=" * 76)
    macro = float(f1_score(gold, pred, average="macro", zero_division=0))
    micro = float(f1_score(gold, pred, average="micro", zero_division=0))
    exact = float(np.all(gold == pred, axis=1).mean())
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    print(f"Comments={len(gold)}; gold aspects={int(gold.sum())}; predicted aspects={int(pred.sum())}")
    print(f"Macro-F1={macro:.4f}; Micro-F1={micro:.4f}; Exact-set match={exact:.4f}")
    print(f"Micro precision={tp/(tp+fp) if tp+fp else 0:.4f}; Micro recall={tp/(tp+fn) if tp+fn else 0:.4f}")
    print(f"{'Category':<10} {'Support':>7} {'Threshold':>10} {'P':>7} {'R':>7} {'F1':>7}  Name")
    per_category = {}
    for j, code in enumerate(model_order):
        p, r, f, _ = precision_recall_fscore_support(
            gold[:, j], pred[:, j], average="binary", zero_division=0
        )
        per_category[code] = dict(support=int(gold[:, j].sum()), threshold=float(thresholds[j]),
                                  precision=float(p), recall=float(r), f1=float(f))
        print(f"{code:<10} {int(gold[:, j].sum()):>7} {thresholds[j]:>10.3f} "
              f"{p:>7.4f} {r:>7.4f} {f:>7.4f}  {ASPECT_NAMES[code]}")
    return dict(macro_f1=macro, micro_f1=micro, exact_set_match=exact,
                tp=tp, fp=fp, fn=fn, per_category=per_category)


def sentiment_metrics(gold: list[int], predicted: list[int], pairs: list[tuple[str, str]]):
    if not gold:
        raise ValueError("No valid gold-present pairs for conditional ACSA evaluation")
    y = np.asarray(gold, dtype=int)
    p = np.asarray(predicted, dtype=int)
    macro = float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))
    acc = float(accuracy_score(y, p))
    conf = confusion_matrix(y, p, labels=[0, 1, 2])
    scores = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    print("\n" + "=" * 76)
    print("ACSA — CONDITIONAL THREE-CLASS SENTIMENT (GOLD-PRESENT CATEGORIES)")
    print("=" * 76)
    print(f"Pairs={len(y)}; Macro-F1={macro:.4f}; Accuracy/Micro-F1={acc:.4f}")
    print(f"{'Sentiment':<10} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for i, label in enumerate(SENTIMENTS):
        print(f"{label:<10} {int(scores[3][i]):>8} {scores[0][i]:>10.4f} "
              f"{scores[1][i]:>8.4f} {scores[2][i]:>8.4f}")
    print("Confusion matrix (gold rows, predicted columns):")
    print(" " * 13 + " ".join(f"{lab[:3]:>6}" for lab in SENTIMENTS))
    for i, label in enumerate(SENTIMENTS):
        print(f"{label:<13}" + " ".join(f"{int(count):>6}" for count in conf[i]))

    by_aspect = defaultdict(list)
    by_comment = defaultdict(list)
    for i, (comment_id, code) in enumerate(pairs):
        by_aspect[code].append(i)
        by_comment[comment_id].append(i)
    print("Per category (sentiment F1):")
    for code in CODES:
        indices = by_aspect[code]
        if indices:
            category_f1 = f1_score(y[indices], p[indices], labels=[0, 1, 2],
                                   average="macro", zero_division=0)
            print(f"  {code}: pairs={len(indices):>4}, Macro-F1={category_f1:.4f}")
    mixed = [idx for idx in by_comment.values() if len({int(y[i]) for i in idx}) > 1]
    if mixed:
        subset = [i for idx in mixed for i in idx]
        mixed_f1 = float(f1_score(y[subset], p[subset], labels=[0, 1, 2],
                                   average="macro", zero_division=0))
        contrast = sum(len({int(p[i]) for i in idx}) > 1 for idx in mixed)
        exact_mixed = sum(all(y[i] == p[i] for i in idx) for idx in mixed)
        print(f"Mixed-sentiment comments={len(mixed)}, pairs={len(subset)}, "
              f"Macro-F1={mixed_f1:.4f}")
        print(f"  Different sentiments predicted: {contrast}/{len(mixed)} "
              f"({contrast/len(mixed):.1%}); all gold sentiments correct: "
              f"{exact_mixed/len(mixed):.1%}")
    else:
        mixed_f1 = None
        print("No gold mixed-sentiment comments in evaluated set")
    return dict(macro_f1=macro, accuracy=acc, confusion_matrix=conf.tolist(),
                mixed_sentiment_macro_f1=mixed_f1, pairs=len(gold))


def end_to_end_metrics(gold_sets: list[set[tuple[str, str]]],
                       pred_sets: list[set[tuple[str, str]]]):
    """Exact aspect+sentiment pairs; wrong sentiment counts as FP and FN."""
    if len(gold_sets) != len(pred_sets) or not gold_sets:
        raise ValueError("Need aligned nonempty end-to-end gold/predicted comment sets")
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    exact = sum(g == p for g, p in zip(gold_sets, pred_sets)) / len(gold_sets)
    by_aspect = {}
    for code in CODES:
        g = [{sent for a, sent in group if a == code} for group in gold_sets]
        p = [{sent for a, sent in group if a == code} for group in pred_sets]
        ctp = sum(len(gg & pp) for gg, pp in zip(g, p))
        cfp = sum(len(pp - gg) for gg, pp in zip(g, p))
        cfn = sum(len(gg - pp) for gg, pp in zip(g, p))
        f1 = 2 * ctp / (2 * ctp + cfp + cfn) if 2 * ctp + cfp + cfn else 0.0
        by_aspect[code] = {"f1": f1, "support": sum(len(group) for group in g),
                            "tp": ctp, "fp": cfp, "fn": cfn}
    macro = sum(v["f1"] for v in by_aspect.values()) / len(CODES)
    print("\n" + "=" * 76)
    print("END-TO-END ABSA — EXACT (CATEGORY, SENTIMENT) PAIRS")
    print("=" * 76)
    print(f"Eligible complete comments={len(gold_sets)}; true positives={tp}, "
          f"false positives={fp}, false negatives={fn}")
    print(f"Triplet Micro-F1={micro_f1:.4f}; Macro-F1 across 11 aspects={macro:.4f}; "
          f"exact complete-set match={exact:.4f}")
    print(f"Precision={micro_p:.4f}; Recall={micro_r:.4f}")
    print("Per-category end-to-end F1 (both aspect AND sentiment must match):")
    for code in CODES:
        r = by_aspect[code]
        print(f"  {code}: support={r['support']:>4}, F1={r['f1']:.4f} "
              f"(TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")
    return dict(micro_f1=micro_f1, macro_f1_across_aspects=macro,
                exact_set_match=exact, precision=micro_p, recall=micro_r,
                tp=tp, fp=fp, fn=fn, comments=len(gold_sets), per_category=by_aspect)


def run(args):
    needs_acd = args.mode in ("acd", "pipeline", "all")
    needs_acsa = args.mode in ("acsa", "pipeline", "all")
    if needs_acd and args.acd_dir is None:
        raise ValueError("--acd-dir is required for --mode acd, pipeline, or all")
    if needs_acsa and args.acsa_dir is None:
        raise ValueError("--acsa-dir is required for --mode acsa, pipeline, or all")
    if args.acd_batch_size < 1 or args.acsa_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    workbook = args.workbook.resolve()
    if not workbook.is_file():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    acd_dir = require_dir(args.acd_dir, "ACD checkpoint") if needs_acd else None
    acsa_dir = require_dir(args.acsa_dir, "ACSA checkpoint") if needs_acsa else None
    validate_workbook_fingerprints(workbook, acd_dir, acsa_dir, args.allow_data_change)

    device = select_device(args.device)
    print("Read-only evaluation; no optimizer, no threshold fitting, no output files.")
    print(f"Sheet: {args.sheet} | device: {device} | mode: {args.mode}")
    if args.sheet.strip().lower() == "testing":
        print("CAUTION: This Testing split has been examined in earlier experiments; "
              "these scores are diagnostic, not a new independent Test estimate.")

    ids, texts, acd_gold = (None, None, None)
    if needs_acd:
        ids, texts, acd_gold = read_acd_gold(workbook, args.sheet)
    if needs_acsa:
        gold_sentiment, text_by_id, exclusions, audit = read_acsa_gold(
            workbook, args.sheet, args.on_conflict, args.on_missing
        )
        print("Sentiment annotation audit:", audit)
        if not gold_sentiment:
            raise ValueError("No valid gold-present sentiment pairs")
    result = {}
    acd_probs = acd_pred = order = None
    if needs_acd:
        print("Loading ACD (stage 1)...", flush=True)
        handle = load_acd(acd_dir, device)
        order, thresholds = handle.order, handle.thresholds.copy()
        acd_probs = infer_acd(handle, texts, args.acd_batch_size)
        release_handle(handle)
        del handle
        # Gold categories originate in shared.labels order, model scores may differ.
        gold_by_order = acd_gold[:, [CODES.index(code) for code in order]]
        acd_pred = (acd_probs >= thresholds[None, :]).astype(np.int8)
        if args.mode in ("acd", "all"):
            result["acd"] = print_acd_metrics(gold_by_order, acd_pred, order, thresholds)

    if needs_acsa:
        if needs_acd:
            id_to_index = {comment_id: i for i, comment_id in enumerate(ids)}
            for (comment_id, code) in gold_sentiment:
                if comment_id not in id_to_index:
                    raise ValueError(f"Sentiment pair refers to unknown ACD comment: {comment_id}")
                i = id_to_index[comment_id]
                if not acd_gold[i, CODES.index(code)]:
                    raise ValueError(f"Gold ACSA category absent from gold ACD set: {comment_id}/{code}")
                if text_by_id[comment_id] != texts[i]:
                    raise ValueError(f"Gold ACD/ACSA text mismatch at comment {comment_id}")
        else:
            # Standalone conditional evaluation, independent of ACD predictions.
            ids = list(dict.fromkeys(id for id, _ in gold_sentiment))
            texts = [text_by_id[i] for i in ids]
            id_to_index = {comment_id: i for i, comment_id in enumerate(ids)}

        condition_pairs = list(gold_sentiment) if args.mode in ("acsa", "all") else []
        e2e_indices = []
        pred_pairs = []
        if args.mode in ("pipeline", "all"):
            # End-to-end scoring requires complete, unambiguous sentiment gold.
            valid_by_comment: dict[str, set[str]] = defaultdict(set)
            for cid, code in gold_sentiment:
                valid_by_comment[cid].add(code)
            excluded_ids = {cid for cid, _ in exclusions}
            for i, comment_id in enumerate(ids):
                gold_codes = {CODES[j] for j in np.flatnonzero(acd_gold[i])}
                available = valid_by_comment.get(comment_id, set())
                if gold_codes != available or comment_id in excluded_ids:
                    continue
                e2e_indices.append(i)
                pred_pairs.extend((comment_id, code) for j, code in enumerate(order)
                                  if acd_pred[i, j])
            print(f"End-to-end eligible complete comments={len(e2e_indices)} / {len(ids)}; "
                  f"excluded={len(ids)-len(e2e_indices)} due to incomplete/ambiguous gold labels")
            if not e2e_indices:
                raise ValueError("No end-to-end-eligible comments after annotation audit")

        # Deduplicate pair requests so gold conditional and predicted category
        # evaluations share the same ACSA inference pass, using NO gold ACD inputs
        # for the end-to-end predictions.
        needed_pairs = list(dict.fromkeys(condition_pairs + pred_pairs))
        print(f"Loading ACSA (stage 2); evaluating {len(needed_pairs)} unique pairs...", flush=True)
        handle = load_acsa(acsa_dir, device)
        probabilities = infer_acsa(
            handle,
            [(text_by_id[cid], code) for cid, code in needed_pairs],
            args.acsa_batch_size,
        )
        release_handle(handle)
        del handle
        sentiment_of = {pair: SENTIMENTS[int(probabilities[j].argmax())]
                        for j, pair in enumerate(needed_pairs)}
        if condition_pairs:
            gold = [gold_sentiment[pair] for pair in condition_pairs]
            pred = [SENTIMENTS.index(sentiment_of[pair]) for pair in condition_pairs]
            result["acsa"] = sentiment_metrics(gold, pred, condition_pairs)

        if args.mode in ("pipeline", "all"):
            gold_sets = []
            predicted_sets = []
            for i in e2e_indices:
                cid = ids[i]
                gold_sets.append({(code, SENTIMENTS[label]) for (comment_id, code), label
                                  in gold_sentiment.items() if comment_id == cid})
                predicted_sets.append({(code, sentiment_of[(cid, code)])
                                       for j, code in enumerate(order) if acd_pred[i, j]})
            result["end_to_end"] = end_to_end_metrics(gold_sets, predicted_sets)
    print("\nCompleted. This evaluator did not write files.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Console evaluation of CITI-Senti ACD + ACSA + end-to-end ABSA")
    parser.add_argument("--acd-dir", type=Path)
    parser.add_argument("--acsa-dir", type=Path)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet", default="Testing", help="Testing or Validation")
    parser.add_argument("--mode", choices=("all", "acd", "acsa", "pipeline"), default="all")
    parser.add_argument("--acd-batch-size", type=int, default=2)
    parser.add_argument("--acsa-batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--on-conflict", choices=("error", "drop"), default="error")
    parser.add_argument("--on-missing", choices=("error", "drop"), default="error")
    parser.add_argument("--allow-data-change", action="store_true",
                        help="Explicitly acknowledge that workbook SHA-256 differs from checkpoint metadata")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
