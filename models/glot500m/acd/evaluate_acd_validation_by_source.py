"""Audit GLOT500-m ACD predictions on authentic vs. synthetic Validation comments.

Does NOT retrain the model, recalibrate thresholds, or access the Testing sheet.
Reads validation_probabilities.npz written by optimized train_acd.py.

CITI-Senti dataset-specific rule: C1132 and later IDs were synthetically
created, even if source_type still contains a platform name. Set the cutoff
explicitly if your dataset's ID assignment changes.

Run from project root:
python models/glot500m/evaluate_acd_validation_by_source.py \\
  --model-dir models/glot500m/artifacts/acd/optimized_expanded_v1 \\
  --synthetic-start-id 1132
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.data import VAL_SHEET, make_aspect_task, read_annotation_sheet
from shared.labels import ASPECT_CODES, ASPECT_NAMES

FOCUS_DEFAULT = ("A01", "A02", "A04", "A06")


def hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    """Metrics with undefined AP marked null instead of falsely claiming 0."""
    out = {
        "comments": int(len(y)),
        "actual_positive_labels": int(y.sum()),
        "predicted_positive_labels": int(pred.sum()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(y, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, pred, average="micro", zero_division=0)),
        "exact_set_match": float(np.all(y == pred, axis=1).mean()),
    }
    per_class, defined_ap = {}, []
    for j, code in enumerate(ASPECT_CODES):
        gold = y[:, j]
        guess = pred[:, j]
        tp = int(np.sum((gold == 1) & (guess == 1)))
        fp = int(np.sum((gold == 0) & (guess == 1)))
        fn = int(np.sum((gold == 1) & (guess == 0)))
        positives = int(gold.sum())
        ap = None
        if 0 < positives < len(gold):
            ap = float(average_precision_score(gold, prob[:, j]))
            defined_ap.append(ap)
        per_class[code] = {
            "name": ASPECT_NAMES[code],
            "support": positives,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision_score(gold, guess, zero_division=0)),
            "recall": float(recall_score(gold, guess, zero_division=0)),
            "f1": float(f1_score(gold, guess, zero_division=0)),
            "average_precision": ap,
        }
    supported = [j for j in range(y.shape[1]) if int(y[:, j].sum()) > 0]
    out["macro_f1_positive_support_categories_only"] = (
        float(np.mean([per_class[ASPECT_CODES[j]]["f1"] for j in supported]))
        if supported else None
    )
    out["categories_with_positive_support"] = len(supported)
    out["macro_average_precision_defined_categories_only"] = (
        float(np.mean(defined_ap)) if defined_ap else None
    )
    out["category_count_with_defined_ap"] = len(defined_ap)
    out["per_category"] = per_class
    return out


def run(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.expanduser().resolve()
    workbook = args.workbook.expanduser().resolve()
    results_path = model_dir / "acd_results.json"
    prob_path = model_dir / "validation_probabilities.npz"
    if not workbook.is_file():
        raise FileNotFoundError(f"Missing workbook: {workbook}")
    if not results_path.is_file() or not prob_path.is_file():
        raise FileNotFoundError(
            f"Need BOTH {results_path} and {prob_path}; the optimized training run "
            "must complete successfully before this audit."
        )
    info = json.loads(results_path.read_text(encoding="utf-8"))
    expected_hash = info.get("data_fingerprint")
    if expected_hash and (actual_hash := hash_file(workbook)) != expected_hash:
        raise ValueError(
            "Workbook fingerprint differs from the workbook used to train this checkpoint.\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}\n"
            "Use the ORIGINAL unchanged split workbook for this checkpoint."
        )
    with np.load(prob_path, allow_pickle=False) as saved:
        required = {"gold", "probabilities", "comment_ids", "aspect_codes"}
        if not required.issubset(set(saved.files)):
            raise ValueError(f"NPZ missing {sorted(required - set(saved.files))}")
        gold = saved["gold"].astype(np.int8)
        probs = saved["probabilities"].astype(np.float64)
        ids = [str(x) for x in saved["comment_ids"].tolist()]
        codes = [str(x) for x in saved["aspect_codes"].tolist()]
    if codes != list(ASPECT_CODES):
        raise ValueError("Saved aspect order differs from shared.labels.ASPECT_CODES")
    if len(ids) != len(set(ids)) or gold.shape != probs.shape or gold.shape != (len(ids), len(codes)):
        raise ValueError("Inconsistent shapes or duplicate IDs in saved predictions")
    if not (np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()):
        raise ValueError("Probabilities contain non-finite or invalid values")

    frame = read_annotation_sheet(workbook, VAL_SHEET)
    task = make_aspect_task(frame)
    workbook_ids = [str(item["comment_id"]) for item in task.row_metadata]
    if ids != workbook_ids:
        raise ValueError(
            "Saved NPZ comment order does not match the Validation worksheet. "
            "Check that the correct workbook and model directory were supplied."
        )
    if not np.array_equal(gold, task.labels.astype(np.int8)):
        raise ValueError("Saved gold labels differ from current Validation labels")

    # Synthetic comments start at C1132 for this specific CITI-Senti augmentation.
    # Platform-based source_type values are unreliable for generated rows.
    id_numbers = []
    for comment_id in ids:
        match = re.fullmatch(r"C(\d+)", comment_id, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Unexpected comment ID {comment_id!r}. This data-specific classifier "
                "expects IDs like C0001; verify the provenance rule before proceeding."
            )
        id_numbers.append(int(match.group(1)))
    is_real = np.asarray([n < args.synthetic_start_id for n in id_numbers], dtype=bool)
    if not np.any(is_real) or np.all(is_real):
        raise ValueError(
            f"Provenance cutoff C{args.synthetic_start_id} produced a one-source Validation set. "
            "Check whether the cutoff applies to this particular workbook."
        )

    # Source-type disagreement is a warning, not an alternative label assignment.
    grouped = frame.groupby("comment_id", sort=False)
    raw_by_id = {str(cid): group for cid, group in grouped}
    misleading_platform = 0
    inconsistent = []
    for cid, real in zip(ids, is_real):
        values = {str(v).strip().lower() for v in raw_by_id[cid]["source_type"].tolist()} if "source_type" in frame.columns else set()
        if "synthetic" in values and real:
            inconsistent.append(cid)
        if not real and values and "synthetic" not in values:
            misleading_platform += 1
    if inconsistent:
        raise ValueError(
            "Conflict: source_type='synthetic' for IDs below synthetic cutoff: "
            + ", ".join(inconsistent[:10]) + ". Verify provenance before reporting."
        )
    if misleading_platform:
        print(f"WARNING: {misleading_platform} synthetic-ID comments have non-synthetic source_type. "
              "Using the established C1132 ID boundary, not source_type.")

    thresholds_by_code = info.get("category_thresholds") or {}
    thresholds = np.asarray([
        float(thresholds_by_code.get(c, info.get("global_threshold", 0.5))) for c in codes
    ])
    if (thresholds < 0).any() or (thresholds > 1).any():
        raise ValueError("Invalid thresholds in saved acd_results.json")
    predictions = (probs >= thresholds[None, :]).astype(np.int8)

    output_dir = args.output_dir or (model_dir / "real_validation_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for title, mask in (("real", is_real), ("synthetic", ~is_real), ("overall", np.ones(len(ids), dtype=bool))):
        summaries[title] = metrics(gold[mask], predictions[mask], probs[mask])

    focus = tuple(x.upper() for x in args.focus)
    if any(x not in codes for x in focus):
        raise ValueError(f"Unknown focus aspect; expected a subset of {codes}")
    rows = []
    for i, cid in enumerate(ids):
        comment_rows = raw_by_id[cid]
        first = comment_rows.iloc[0]
        text = str(first.get("model_text", ""))
        actual_aspects = ", ".join(c for j,c in enumerate(codes) if gold[i,j])
        guessed_aspects = ", ".join(c for j,c in enumerate(codes) if predictions[i,j])
        for code in focus:
            j = codes.index(code)
            if gold[i,j] == predictions[i,j]:
                continue
            rows.append({
                "provenance": "real" if is_real[i] else "synthetic",
                "comment_id": cid,
                "focus_aspect": code,
                "aspect_name": ASPECT_NAMES[code],
                "error_type": "false_positive" if predictions[i,j] else "false_negative",
                "predicted_probability": round(float(probs[i,j]), 6),
                "threshold": round(float(thresholds[j]), 6),
                "distance_from_threshold": round(abs(float(probs[i,j] - thresholds[j])), 6),
                "all_actual_aspects": actual_aspects,
                "all_predicted_aspects": guessed_aspects,
                "source_type_as_recorded": str(first.get("source_type", "")),
                "source_reference": str(first.get("source_reference", "")),
                "model_text": text,
                "review_note": "",  # Fill this during manual annotation review
            })
    rows.sort(key=lambda x: (x["provenance"] != "real", -x["distance_from_threshold"],
                             x["focus_aspect"], x["comment_id"]))

    csv_path = output_dir / "validation_focus_errors.csv"
    fields = [
        "provenance", "comment_id", "focus_aspect", "aspect_name", "error_type",
        "predicted_probability", "threshold", "distance_from_threshold",
        "all_actual_aspects", "all_predicted_aspects", "source_type_as_recorded",
        "source_reference", "model_text", "review_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "checkpoint": str(model_dir),
        "workbook": str(workbook),
        "synthetic_start_id": args.synthetic_start_id,
        "thresholds_from": "saved acd_results.json (no retuning)",
        "focus_aspects": focus,
        "errors_in_focus_csv": len(rows),
        "errors_by_provenance": {"real": sum(x["provenance"] == "real" for x in rows),
                                "synthetic": sum(x["provenance"] == "synthetic" for x in rows)},
        "real": summaries["real"],
        "synthetic": summaries["synthetic"],
        "overall": summaries["overall"],
        "caution": (
            "Real Validation contains very few examples and is not an independent test set; "
            "checkpoint/threshold were already selected using Validation. Do not tune on Testing."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "validation_source_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Validation performance using SAVED threshold(s) ===")
    for group in ("real", "synthetic", "overall"):
        m = summaries[group]
        print(f"{group.upper():9s} n={m['comments']:3d} "
              f"macro_F1={m['macro_f1']:.4f} micro_F1={m['micro_f1']:.4f} "
              f"exact_set={m['exact_set_match']:.4f}")
    print("\n=== REAL Validation, per-category (small samples: interpret cautiously) ===")
    print("aspect support TP FP FN precision recall F1")
    for code, m in summaries["real"]["per_category"].items():
        print(f"{code:5s} {m['support']:>6d} {m['tp']:>2d} {m['fp']:>2d} {m['fn']:>2d} "
              f"{m['precision']:.3f} {m['recall']:.3f} {m['f1']:.3f}")
    print(f"\nFocus error rows: {len(rows)} "
          f"(real={report['errors_by_provenance']['real']}, "
          f"synthetic={report['errors_by_provenance']['synthetic']})")
    print("Saved:", report_path)
    print("Saved:", csv_path)
    print("No Testing data were read. No model training or threshold tuning performed.")


def main():
    p = argparse.ArgumentParser(description="Audit GLOT500-m on real Validation only")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--workbook", type=Path,
                   default=PROJECT_ROOT / "data/splits/citizen_feedback_split.xlsx")
    p.add_argument("--synthetic-start-id", type=int, default=1132,
                   help="CITI-Senti cutoff: IDs C1132 and above are synthetic")
    p.add_argument("--focus", nargs="+", default=FOCUS_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
