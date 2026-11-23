# CITI-Senti — ACD + ACSA Integration and Evaluation

**New scripts only.** You do not need to replace your ACD or ACSA trainers, and no model training occurs here. Place the following three files together inside your existing repository:

```text
citi-senti-nlp/
├── shared/
│   ├── data.py
│   └── labels.py
├── data/splits/citizen_feedback_split.xlsx
└── models/glot500m/
    ├── artifacts/
    │   ├── acd/optimized_expanded_v2/
    │   └── acsa/v1_short_search/trial_02_lr1e5_sqrt/
    ├── acd/                 # existing training/evaluation files; optional arrangement
    ├── acsa/                # existing training/evaluation files; optional arrangement
    └── integration/         # new folder
        ├── absa_runtime.py
        ├── predict_absa_pipeline.py
        └── evaluate_absa_pipeline.py
```

Dependencies: existing project's `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `openpyxl`, `sentencepiece`, `safetensors`; `peft` is required **only** for LoRA ACD checkpoints. Scripts locate the repository's `shared/labels.py` regardless of whether ACD and ACSA trainers have been moved into separate subfolders.

## A. Predict sentiment for a new comment (ACD -> ACSA)

Run in the **repository root** in Windows PowerShell:

```powershell
python models/glot500m/integration/predict_absa_pipeline.py `
  --acd-dir models/glot500m/artifacts/acd/optimized_expanded_v2 `
  --acsa-dir models/glot500m/artifacts/acsa/v1_short_search/trial_02_lr1e5_sqrt `
  --text "Maganda ang traffic policy pero kulang ang enforcement."
```

The script prints JSON with the detected categories (e.g., A02 and A10), each category's ACD sigmoid probability, and the corresponding ACSA sentiment with its softmax probabilities. Actual predictions depend on the trained checkpoints. **No aspect detected** returns `"aspects": []`; no sentiment is fabricated. Loading/status messages go to stderr so stdout remains parseable JSON.

For your **LoRA Trial 4** ACD instead, use:

```powershell
python models/glot500m/integration/predict_absa_pipeline.py `
  --acd-dir models/glot500m/artifacts/acd/v3_focused/trial_04_all_linear_positive_bce `
  --acsa-dir models/glot500m/artifacts/acsa/v1_short_search/trial_02_lr1e5_sqrt `
  --text "Maganda ang traffic policy pero kulang ang enforcement."
```

Both types of ACD checkpoint work: full-fine-tuned `model.safetensors`/`config.json` or `adapter_model.safetensors`/`adapter_config.json` PEFT LoRA adapter **with trained classifier saved in the adapter**. ACD uses the *previously saved Validation-selected thresholds* from `acd_results.json` or `.acd_results.json`. ACSA uses the **original V1 comment-first paired format**: `tokenizer(text=model_text, text_pair="Aspect: CODE — NAME", truncation="only_first")`. It does **not** use an Experiment-2 aspect-first checkpoint.

To fit a 6 GB GPU, the implementation loads ACD, predicts the categories, releases ACD, and then loads ACSA. For a web server, you can import `predict_comments` from `absa_runtime.py` and pass a list of texts; repeated calls reload the models, so high-throughput deployment would require an optimized persistent model service.

## B. Evaluate ACD, ACSA, and integrated ABSA on the same sheet

```powershell
python models/glot500m/integration/evaluate_absa_pipeline.py `
  --mode all `
  --acd-dir models/glot500m/artifacts/acd/optimized_expanded_v2 `
  --acsa-dir models/glot500m/artifacts/acsa/v1_short_search/trial_02_lr1e5_sqrt `
  --workbook data/splits/citizen_feedback_split.xlsx `
  --sheet Testing `
  --on-conflict drop
```

Choose `--sheet Validation` to inspect the development split instead. With `--sheet Testing`, this is a **diagnostic** run on a split you have already examined in previous experiments, not a new independent test.

**Outputs to the terminal only:**

1. **ACD** (comment-level, 11-category multi-label): Macro-F1, Micro-F1, exact-set match, precision and recall, and F1 per A01–A11.
2. **Conditional ACSA** (one row per *gold-present* comment/category pair): three-class Macro-F1 and accuracy, Negative/Neutral/Positive precision/recall/F1, confusion matrix, per-aspect sentiment F1, and mixed-sentiment analysis.
3. **End-to-end ABSA** (from *ACD-predicted* categories, not gold categories): precision, recall, Micro-F1 for exact `(category, sentiment)` pairs, a category-macro F1 over 11 categories, and complete-comment exact-set match. A category that is missed by ACD cannot reach ACSA. A correct category with the wrong sentiment contributes one FP and one FN.

The end-to-end evaluator only scores comments whose **entire gold category set has valid, unambiguous sentiment annotations**. With `--on-conflict drop`, ambiguous pairs are omitted from conditional ACSA, but entire affected comments are excluded from end-to-end ABSA. ACD scoring still includes all comments. It prints how many end-to-end comments were excluded; report that exclusion in the paper.

### Evaluate only one component

**ACD only:**

```powershell
python models/glot500m/integration/evaluate_absa_pipeline.py `
  --mode acd `
  --acd-dir models/glot500m/artifacts/acd/optimized_expanded_v2 `
  --sheet Testing
```

**ACSA only** (no ACD needed; gold category pairs):

```powershell
python models/glot500m/integration/evaluate_absa_pipeline.py `
  --mode acsa `
  --acsa-dir models/glot500m/artifacts/acsa/v1_short_search/trial_02_lr1e5_sqrt `
  --sheet Testing `
  --on-conflict drop
```

**End-to-end only:** use `--mode pipeline` with both model paths.

### Full vs LoRA and model metadata

- Full ACD: `config.json`, `model.safetensors` (or supported shards), `acd_results.json` (or `.acd_results.json`). Tokenizer may load from base `cis-lmu/glot500-base` if not saved locally.
- LoRA ACD: `adapter_config.json`, `adapter_model.safetensors`, `acd_results.json`; `peft` must be installed and adapter must include the saved `classifier` head.
- Full ACSA V1: `config.json`, weights, `acsa_results.json`, and tokenizer files (or base checkpoint reference). Label order is checked to be Negative, Neutral, Positive.
- Shared category order comes from `shared.labels.ASPECT_CODES`, not alphabetical sorting or the model's default `LABEL_0` placeholders.
- The evaluator compares the workbook SHA-256 against each checkpoint's recorded fingerprint when available. If the file changed, it will stop rather than silently compare changed labels. `--allow-data-change` explicitly bypasses this guard **only after you review what changed**.
- No results or predictions are saved by the evaluation or inference scripts. Model downloads may populate your normal Hugging Face cache when missing the base Glot500 model.

## Reproducibility and leakage limitations

The code never trains, tunes ACD thresholds, or calibrates ACSA on Testing labels. **Separate code does not itself guarantee independence**: synthetic derivatives of Training/Validation/Test originals must be grouped correctly. Your Testing partition has already been examined in earlier ACSA experiments; use a new, truly untouched holdout for a fresh final estimate of newly developed models. Don't pick checkpoints based on repeatedly viewed Test scores.

## Validation of this package

- Syntax-checked using `py_compile`.
- Tested the full ACD/ACSA/end-to-end metric path using mocked predictions and a temporary workbook, including conflicting annotation exclusion and wrong-sentiment FP/FN accounting.
- No actual model training or inference with your private saved weights was performed in this environment. Test using your real checkpoints locally and verify the saved label mapping/thresholds.
