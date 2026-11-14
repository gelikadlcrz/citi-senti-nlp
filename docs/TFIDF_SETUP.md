# Shared TF-IDF Setup

Technical documentation for the shared **Term Frequency–Inverse Document Frequency (TF-IDF)** feature pipeline used by the CITI-Senti **Logistic Regression** and **Linear SVM** baselines.

The feature extraction code is shared across both baselines. Each model trains its own classifiers using the **same numerical inputs** for a given task. Glot500-m does not use these TF-IDF matrices; it uses its own tokenizer and pretrained representations.

## Overview

CITI-Senti trains and evaluates two classification tasks:

| Task | Unit of prediction | Model input | Target label |
| --- | --- | --- | --- |
| Multi-label aspect detection | One unique citizen comment | `model_text` | An 11-element multi-hot vector (A01–A11) |
| Aspect-based sentiment classification | One annotated aspect occurrence | `model_text` + `aspect_code` + `aspect_text` | `Negative`, `Neutral`, or `Positive` |

The aspect-level sentiment input explicitly includes `aspect_text` because an individual comment may contain several targets within the same aspect category with different sentiments. **This configuration assumes an aspect target is supplied.** An end-to-end system must identify or otherwise obtain the target expression before making this type of sentiment prediction. A category-level aspect detector alone does not extract target expressions.

The input is the existing `citizen_feedback_split.xlsx` workbook, with three worksheets: `Training`, `Validation`, and `Testing`. **The pipeline does not create or revise the split.** Only the Training worksheet is used to fit the TF-IDF vocabulary and IDF weights. Validation and Testing are transformed with those same fitted vectorizers.

## Repository layout

```text
citi-senti-nlp/
├── data/
│   └── splits/
│       └── citizen_feedback_split.xlsx      # Local research data; not publicly committed
├── shared/
│   ├── __init__.py
│   ├── data.py                               # Read workbook; prepare task examples
│   ├── labels.py                             # A01–A11 and sentiment label mappings
│   ├── tfidf.py                              # TF-IDF configuration; save/load vectorizers
│   ├── prepare_tfidf.py                      # Fit on Training; transform Validation
│   ├── features.py                           # Load saved matrices and labels
│   ├── train_common.py                       # Shared baseline training/evaluation logic
│   └── transform_split.py                    # Transform held-out Testing later
├── models/
│   ├── tfidf/                                # Generated shared artifacts (Git-ignored)
│   ├── logistic_regression/
│   │   ├── train.py
│   │   └── artifacts/                        # Generated model outputs (Git-ignored)
│   ├── linear_svm/
│   │   ├── train.py
│   │   └── artifacts/                        # Generated model outputs (Git-ignored)
│   └── glot500m/                             # Separate transformer implementation
├── tests/
│   └── test_shared_tfidf.py
├── requirements-baselines.txt
└── docs/
    └── TFIDF_SETUP.md
```

`__init__.py` declares Python packages. It is not an executable training script. The generated `models/tfidf/` files and each model's `artifacts/` directory should be treated as outputs, not source files to edit manually.

## Prerequisites

Use a Python environment with the dependencies specified in `requirements-baselines.txt`. From the **repository root**:

```bash
python -m pip install -r requirements-baselines.txt
```

The distributed pre-fitted vectorizers were created with **scikit-learn 1.8.0**, which is pinned in the requirements file. Loading serialized scikit-learn artifacts across different versions is not guaranteed to work. Only load `.joblib` artifacts from a trusted source.

To regenerate features or later transform Testing, place the original workbook at:

```text
data/splits/citizen_feedback_split.xlsx
```

The workbook must contain worksheets named `Training`, `Validation`, and `Testing`, with the columns `comment_id`, `aspect_id`, `aspect_code`, `aspect_text`, `sentiment`, and `model_text`. The additional language and grouping fields are retained as available in row manifests.

## Execution order

| Step | Command | Reads | Writes | When |
| --- | --- | --- | --- | --- |
| 0. Install | `python -m pip install -r requirements-baselines.txt` | Dependency list | Local Python environment | Once per environment |
| 1. Prepare TF-IDF | `python -m shared.prepare_tfidf --workbook data/splits/citizen_feedback_split.xlsx` | Training and Validation worksheets | 15 files in `models/tfidf/` | Only if artifacts are not already available |
| 2a. Train LR | `python -m models.logistic_regression.train` | Shared Training/Validation feature files | Two LR models and validation metrics | Baseline development |
| 2b. Train SVM | `python -m models.linear_svm.train` | Same shared feature files | Two SVM models and validation metrics | Baseline development; may run alongside 2a |
| 3. Transform test | `python -m shared.transform_split --workbook data/splits/citizen_feedback_split.xlsx` | Testing worksheet and existing fitted vectorizers | Six test feature/label/manifest files | After selecting final models and hyperparameters |

If the pre-fitted `models/tfidf/` artifacts were provided privately to the team, **skip Step 1**. The preparation script intentionally refuses to replace existing vectorizers unless `--force` is supplied.

### Step 1 — Generate shared TF-IDF features

```bash
python -m shared.prepare_tfidf --workbook data/splits/citizen_feedback_split.xlsx
```

Execution path:

```text
citizen_feedback_split.xlsx
          │
          ▼
shared/prepare_tfidf.py
    ├── shared/data.py       → read worksheets; build task-specific examples
    ├── shared/labels.py     → encode aspect and sentiment targets
    └── shared/tfidf.py      → fit vectorizers on Training only
          │
          ▼
models/tfidf/                  → saved vectorizers, matrices, labels, manifests, metadata
```

The pipeline creates **two fitted vectorizers**, one for comment-level aspect detection and one for target-conditioned sentiment classification. Both use word unigrams and bigrams, lowercase text, sublinear term frequency, L2 normalization, and Unicode-aware tokenization. No automatic translation, stemming, or English-only stopword removal is applied.

With the supplied dataset version, the generated feature dimensions and example counts are:

| Task | Training examples | Validation examples | Feature dimensions |
| --- | ---: | ---: | ---: |
| Aspect detection | 681 unique comments | 146 unique comments | 18,616 |
| Aspect sentiment | 928 annotation rows | 200 annotation rows | 20,185 |

These counts describe the current workbook and saved feature configuration. They will change if the input data or configuration is intentionally revised.

#### Files generated in `models/tfidf/`

| File(s) | Meaning |
| --- | --- |
| `aspect_train_X.npz`, `aspect_val_X.npz` | Sparse TF-IDF feature matrices for comment-level aspect detection |
| `aspect_train_y.npy`, `aspect_val_y.npy` | Eleven-column multi-hot aspect targets |
| `aspect_train_rows.csv`, `aspect_val_rows.csv` | Comment identifiers and associated row metadata |
| `aspect_vectorizer.joblib` | Fitted aspect-detection TF-IDF vocabulary and IDF weights |
| `sentiment_train_X.npz`, `sentiment_val_X.npz` | Sparse TF-IDF feature matrices for aspect-specific sentiment inputs |
| `sentiment_train_y.npy`, `sentiment_val_y.npy` | Numeric sentiment targets (`Negative=0`, `Neutral=1`, `Positive=2`) |
| `sentiment_train_rows.csv`, `sentiment_val_rows.csv` | Annotation identifiers, aspect targets, and associated row metadata |
| `sentiment_vectorizer.joblib` | Fitted sentiment TF-IDF vocabulary and IDF weights |
| `metadata.json` | Source schema, label mappings, vectorizer settings, dimensions, counts, and dependency versions |

The naming convention is `<task>_<split>_X.npz` for features, `<task>_<split>_y.npy` for labels, and `<task>_<split>_rows.csv` for row manifests. `train` means Training; `val` means Validation.

File formats:

- **`.npz`**: compressed sparse matrices representing numerical text features.
- **`.npy`**: NumPy arrays containing the corresponding target labels.
- **`.csv`**: row manifests used to trace predictions back to source examples.
- **`.joblib`**: fitted TF-IDF vectorizers, reused for later text transformation.
- **`.json`**: metadata describing the feature generation run.

### Step 2a — Train the Logistic Regression baseline

```bash
python -m models.logistic_regression.train
```

### Step 2b — Train the Linear SVM baseline

```bash
python -m models.linear_svm.train
```

Both entry points use `shared/train_common.py`, which calls `shared/features.py` to load the **identical** Training and Validation matrices from `models/tfidf/`. Each baseline trains:

1. A one-vs-rest multi-label aspect detector.
2. A three-class aspect sentiment classifier conditioned on the annotated aspect target.

For each baseline, the command creates these files under its respective `artifacts/` directory:

| Generated file | Description |
| --- | --- |
| `aspect_model.joblib` | Trained multi-label aspect classification model |
| `sentiment_model.joblib` | Trained three-class aspect sentiment model |
| `validation_metrics.json` | Validation metrics, including macro-F1 and task-specific classification reports |

The scripts are **starting experiments**, not exhaustive hyperparameter searches. For example, to change a model's regularization parameter and iteration limit:

```bash
python -m models.logistic_regression.train --C 0.5 --max-iter 5000
python -m models.linear_svm.train --C 2.0 --max-iter 5000
```

Running a training command again with the same output directory overwrites its existing model files and validation metrics. Record important experiment configurations or specify another output directory when retaining multiple runs:

```bash
python -m models.logistic_regression.train --output models/logistic_regression/artifacts/experiment_02
```

### Step 3 — Transform the held-out test partition

**Only run after model selection is complete.** This operation prepares the Testing worksheet for later evaluation; it does not evaluate either baseline.

```bash
python -m shared.transform_split --workbook data/splits/citizen_feedback_split.xlsx
```

This command loads the previously fitted `aspect_vectorizer.joblib` and `sentiment_vectorizer.joblib` and generates:

```text
models/tfidf/
├── aspect_test_X.npz
├── aspect_test_y.npy
├── aspect_test_rows.csv
├── sentiment_test_X.npz
├── sentiment_test_y.npy
└── sentiment_test_rows.csv
```

No fitting is performed on Testing. The final evaluation code can load these files after the experimental protocol is frozen.

## Reusing features in other scripts

Both baseline implementations can load prepared feature matrices through the same helper:

```python
from shared.features import load_features

# Aspect detection
X_train_aspect, y_train_aspect = load_features(
    "models/tfidf", task="aspect", split="train"
)
X_val_aspect, y_val_aspect = load_features(
    "models/tfidf", task="aspect", split="val"
)

# Aspect sentiment classification
X_train_sentiment, y_train_sentiment = load_features(
    "models/tfidf", task="sentiment", split="train"
)
X_val_sentiment, y_val_sentiment = load_features(
    "models/tfidf", task="sentiment", split="val"
)
```

`load_features()` returns a SciPy CSR sparse matrix and a NumPy array of labels. It does not refit the vectorizer.

The vectorizers themselves can be loaded when converting new comment inputs:

```python
from shared.tfidf import load_aspect_vectorizer, load_sentiment_vectorizer

aspect_vectorizer = load_aspect_vectorizer("models/tfidf")
sentiment_vectorizer = load_sentiment_vectorizer("models/tfidf")
```

For sentiment input formatting, use `shared.data.make_sentiment_input()` to preserve the exact input structure used during training.

## Tests

From the repository root:

```bash
python -m pytest -q tests/test_shared_tfidf.py
```

These tests use synthetic examples to check feature preparation and reuse. They do **not** retrain the study's final models or use the research test partition.

## Regeneration and reproducibility

Only regenerate when an intentional change has been made to the input data or feature configuration:

```bash
python -m shared.prepare_tfidf \
  --workbook data/splits/citizen_feedback_split.xlsx \
  --force
```

The `--force` option replaces the fitted vectorizers and Training/Validation matrices and removes stale saved split matrices, including any previously transformed test features. **Both baselines must be retrained** after regeneration. Ensure both baseline teams use the same regenerated feature files and corresponding `metadata.json`.

The shared feature representation is held constant across Logistic Regression and Linear SVM to make the algorithmic comparison interpretable. Later vectorizer experiments should be clearly identified and evaluated on Validation rather than silently changing the common input features.

## Data access and Git policy

The input workbook contains collected research annotations and is not included in the public repository by default. The generated `*_rows.csv` manifests also include source identifiers and aspect-target text. Accordingly, the repository's `.gitignore` excludes:

```gitignore
models/tfidf/
models/logistic_regression/artifacts/
models/linear_svm/artifacts/
```

Commit the source scripts, test code, dependency specification, and documentation. Share the original dataset and fitted artifacts through approved private team storage unless a separate data-release review authorizes publication.

## Limitations

- The aspect detector predicts **aspect categories**, not the exact spans of aspect expressions.
- The sentiment classifier is evaluated with **annotated target expressions**; that standalone result is not equivalent to a fully automatic, end-to-end ABSA result.
- A complete inference system requires a way to provide or extract target expressions, or a separately designed sentiment model whose inputs depend only on available predictions.
- The current baseline scripts generate **validation metrics**. Final held-out test comparison and end-to-end pipeline evaluation are separate tasks.

For a general project overview, refer to the repository's root [`README.md`](../README.md).
