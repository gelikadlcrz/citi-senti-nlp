# Glot500-m ACD — Training and Hyperparameter Tuning Protocol

**Project:** CITI-Senti NLP  
**Audience:** Project members running experiments on Windows using VS Code and PowerShell  
**Scope:** **Aspect-Category Detection (ACD) only**  
**Pretrained checkpoint:** `cis-lmu/glot500-base`  
**Implementation:** `models/glot500m/train_acd.py` and `models/glot500m/tune_acd.py`  
**Protocol version:** Threshold-aware checkpoint selection (revised `train_acd.py`)

> **Important:** This README documents what the *current scripts actually run*: ACD fine-tuning and random-search tuning on the existing **Training** and **Validation** worksheets. This implementation does **not** perform nested cross-validation, repeated-seed evaluation, ACSA training, or final held-out test evaluation. Do not report its validation scores as final test results.

## 1. Files required

From the **`citi-senti-nlp` repository root**, confirm the following structure:

```text
citi-senti-nlp/
├── .venv/                                 # Local virtual environment (not committed)
├── data/
│   └── splits/
│       └── citizen_feedback_split.xlsx    # Existing, already-split workbook
├── shared/
│   ├── data.py                            # Existing shared data loader
│   └── labels.py                          # A01–A11 category definitions
├── models/
│   └── glot500m/
│       ├── train_acd.py                   # Revised threshold-aware training script
│       ├── tune_acd.py                    # Random-search trial runner
│       └── artifacts/                     # Created during training
├── requirements-baselines.txt
└── requirements-glot500.txt               # Glot500 dependencies (if maintained in repo)
```

Keep the workbook's existing `Training`, `Validation`, and `Testing` sheets unchanged. **Only Training and Validation are read by these ACD scripts.** The test sheet is reserved for final evaluation.

> Naming: The implemented script is **`train_acd.py`**, not `acd_train.py`.

## 2. One-time environment setup

Open the repository folder in **VS Code**. In **Terminal → New Terminal**, select **PowerShell**. All commands below assume the terminal is at the repository root.

### 2.1 Activate the project environment

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -c "import sys; print(sys.executable)"
```

Use a supported **64-bit Python 3.11 or newer** environment. The executable path should point into the project's `.venv` folder. If PowerShell blocks activation, run the following once in that terminal session and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

If `.venv` has not been created, create it with a compatible installed Python **before** activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 2.2 Install libraries

```powershell
python -m pip install -r requirements-baselines.txt
```

Install **one** appropriate PyTorch build. The CUDA 12.8 example below requires a supported NVIDIA GPU and driver:

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only setup instead (model training will be substantially slower):

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

For a different GPU/driver, use the compatible installation command from [PyTorch](https://pytorch.org/get-started/locally/), rather than forcing the CUDA 12.8 build.

Install the project's Glot500 dependencies if the requirements file exists:

```powershell
python -m pip install -r requirements-glot500.txt
python -m pip install protobuf
```

Otherwise, install them directly:

```powershell
python -m pip install "transformers==4.57.6" "accelerate>=1.3,<2" "sentencepiece>=0.2,<0.3" "safetensors>=0.6,<1" protobuf
```

Check the environment:

```powershell
python -m pip check
python -c "import torch, transformers, sklearn, pandas, sentencepiece; print('Dependencies OK'); print('CUDA:', torch.cuda.is_available())"
```

For NVIDIA GPUs, `CUDA: True` indicates PyTorch can access CUDA. The script automatically uses CUDA when available.

## 3. Check the split workbook before training

Check the required files:

```powershell
Test-Path .\data\splits\citizen_feedback_split.xlsx
Test-Path .\shared\data.py
Test-Path .\models\glot500m\train_acd.py
Test-Path .\models\glot500m\tune_acd.py
```

Each should return `True`. Then run the **ACD-only data check**, which does not download or train Glot500:

```powershell
python models/glot500m/train_acd.py --check-data
```

The workbook used to prepare this implementation has **681 distinct Training comments** and **146 distinct Validation comments**. The data loader builds one 11-element multi-hot category vector per comment using `shared/data.py::make_aspect_task()`. Repeated category annotations for a comment become one positive category indicator; sentiment labels are **not** used for ACD. Existing positive/negative sentiment conflicts therefore do not need to be resolved for this ACD phase.

The command should end with a message equivalent to `Data check passed. Training and Validation are disjoint.` If it does not, correct the workbook path or data issue before training.

## 4. Confirm the revised training script

The revised script selects its best checkpoint using **threshold-tuned validation Macro-F1**, instead of using only the fixed threshold of `0.50`.

To confirm the updated code is present:

```powershell
Select-String -Path .\models\glot500m\train_acd.py -Pattern "val_macro_f1@tuned"
```

The search should find the string. Training logs should show both `val_macro_f1@0.50` and `val_macro_f1@tuned`.

> Do **not** replace `shared/data.py` or resplit the Excel workbook for this change. Only `train_acd.py` needs the threshold-aware checkpoint-selection update; `tune_acd.py` can remain unchanged.

## 5. Optional: run a single ACD fine-tuning experiment

Use this first if a team member has **not yet verified that full training works**:

```powershell
python models/glot500m/train_acd.py --batch-size 1 --gradient-accumulation 8 --gradient-checkpointing --output models/glot500m/artifacts/acd/sanity_seed42
```

This runs one configuration with the script's defaults: learning rate `2e-5`, at most 5 epochs, max token length 128, weight decay 0.01, warmup ratio 0.10, seed 42, and early-stopping patience 2. Effective batch size is `1 × 8 = 8`.

This is **not required** before tuning if the same team member already completed an ACD training run. The tuner trains fresh candidate models from the pretrained `cis-lmu/glot500-base` checkpoint; it **does not continue** from the sanity-check model.

## 6. Run the hyperparameter search

Run the following commands from the same activated project root.

### 6.1 Preview the first three configurations (no training)

```powershell
python models/glot500m/tune_acd.py --trials 3 --dry-run
```

The command prints sampled configurations and ends with `Dry run only; no model training and no files written.` The 12-trial search is deterministic for a given random seed (default: `42`).

### 6.2 Run three training trials

**Use a new output directory for the revised protocol**, so the tuner does not mistake earlier completed experiments for new runs:

```powershell
python models/glot500m/tune_acd.py --trials 3 --output-root models/glot500m/artifacts/acd/search_threshold_aware_seed42 --prune-rejected
```

Each trial starts from the original pretrained Glot500-m checkpoint and independently fine-tunes an 11-label ACD classifier. Results are compared using validation Macro-F1 after threshold tuning.

### 6.3 Complete all twelve trials

When the three trial runs have completed successfully, use the **same output directory**:

```powershell
python models/glot500m/tune_acd.py --trials 12 --output-root models/glot500m/artifacts/acd/search_threshold_aware_seed42 --prune-rejected
```

The tuner reuses valid completed trials 1–3 and runs trials 4–12. It does not retrain completed trials automatically.

> If `search_threshold_aware_seed42` already contains prior trial outputs you want to rerun, choose a **different folder name** (e.g. `search_threshold_aware_seed42_v2`) and use it consistently for both commands. Do not delete or overwrite original experiments without first archiving them.

### 6.4 What is being searched?

The tuner randomly orders **144 possible configurations** and runs the first 12 using seed 42:

| Setting | Search values |
|---|---|
| Learning rate | `1e-5`, `2e-5`, `3e-5` |
| Effective batch size | 8, 16 |
| Maximum epochs | 3, 5, 8 |
| Maximum sequence length | 128, 256 tokens |
| Weight decay | 0, 0.01 |
| Warmup ratio | 0, 0.10 |

The tuner uses **microbatch size 1**, gradient accumulation 8 or 16, gradient checkpointing by default, AdamW, linear learning-rate warmup/decay, mixed precision on CUDA, and early stopping with patience 2.

### 6.5 How checkpoints and thresholds are selected

At the end of **each epoch**, the training script:

1. Computes validation sigmoid probabilities for the 11 category outputs.
2. Reports Macro-F1 at a fixed threshold of `0.50`.
3. Searches a **global threshold** from `0.20` to `0.80` in steps of `0.05` (default threshold mode).
4. Computes threshold-tuned validation Macro-F1.
5. Saves the checkpoint when that tuned Macro-F1 improves; otherwise counts toward early stopping.

Afterward, the saved checkpoint is reloaded, its threshold is selected again on Validation, and the trial records its validation metrics. Tuning ranks trial configurations by **validation 11-label Macro-F1 after threshold selection**, with trial order breaking ties.

The optional `--threshold-mode eligible-category` permits category-specific thresholds only for categories with sufficient validation support. Use this only if your team has agreed on that experimental setting; do not compare it informally against global-threshold runs without recording the difference.

## 7. Monitor progress during training

A correctly updated training log resembles:

```text
Device: cuda | model: cis-lmu/glot500-base
Epoch 1/8: loss=..., val_macro_f1@0.50=..., val_macro_f1@tuned=..., threshold=...
Epoch 2/8: loss=..., val_macro_f1@0.50=..., val_macro_f1@tuned=..., threshold=...
...
Best epoch: ...
Validation Macro-F1 after threshold tuning: ...
```

- **Loss**: training objective; decreasing loss alone does not mean category detection is accurate.
- **`val_macro_f1@0.50`**: fixed-threshold diagnostic, which may be zero even when predictions below 0.50 are informative.
- **`val_macro_f1@tuned`**: value used to choose the best epoch.
- **`threshold`**: selected validation threshold for that epoch.

The warning about newly initialized `classifier.*` weights is normal. It indicates the task-specific classification head has not yet been trained when the pretrained encoder is loaded.

## 8. Find and interpret saved outputs

The standard revised search produces:

```text
models/glot500m/artifacts/acd/search_threshold_aware_seed42/
├── trial_01/
│   ├── acd_results.json
│   ├── config.json
│   ├── model.safetensors              # Kept only if this trial is retained
│   └── tokenizer files
├── trial_02/
├── ...
├── trial_12/
├── search_results.csv
└── best_configuration.json
```

Open `search_results.csv` or inspect it from PowerShell:

```powershell
Import-Csv .\models\glot500m\artifacts\acd\search_threshold_aware_seed42\search_results.csv |
    Format-Table rank, trial, macro_f1, micro_f1, exact_set_match, best_epoch
```

View the winning configuration:

```powershell
Get-Content .\models\glot500m\artifacts\acd\search_threshold_aware_seed42\best_configuration.json
```

For more detailed diagnostics, open the winning trial's `acd_results.json`. Its important fields are:

| JSON field | Meaning |
|---|---|
| `best_epoch` | Epoch whose model weights were saved |
| `best_epoch_macro_f1_at_0_5` | Fixed-threshold F1 at that epoch |
| `best_epoch_macro_f1_after_threshold_tuning` | F1 used to choose that epoch |
| `validation_after_threshold_tuning` | Final tuned Macro-F1, Micro-F1, and exact-set match |
| `global_threshold` | Selected default ACD cutoff |
| `category_thresholds` | The actual A01–A11 cutoffs |
| `validation_per_category` | Support, predicted positives, TP, FP, FN, precision, recall, F1 |
| `history` | Per-epoch loss and F1 information |
| `training_parameters` | Hyperparameters and random seed |

With `--prune-rejected`, the tuner deletes `model.safetensors` files for **completed losing trials** but keeps their metrics. If you intend to compare actual checkpoints for multiple trials, omit this option and ensure disk space is sufficient.

## 9. Repeating experiments without losing results

- Keep the original first-run directory, typically `artifacts/acd/search_seed42/`, for comparison.
- Use a **new output directory** when modifying code, training settings, thresholds, or dataset versions.
- The tuner automatically reuses trials when `acd_results.json` exists and stored configuration/seed/mode match. It **does not verify your source-code revision**. A fresh directory is therefore essential after changing `train_acd.py`.
- Do not launch two searches at the same time into the **same output directory**.
- Record the script revision (Git commit), workbook version, Python/PyTorch/Transformers versions, device/GPU, seed, threshold mode, and output directory for every experiment.

Suggested experiment log:

| Field | Record |
|---|---|
| Date and team member | |
| Git commit hash | |
| Workbook filename/version | |
| GPU model and VRAM | |
| Python / PyTorch / Transformers versions | |
| Seed | 42 |
| Threshold mode | global |
| Search directory | |
| Winning trial / hyperparameters | |
| Validation Macro-F1 / Micro-F1 | |
| Notes / issues | |

You can capture dependency versions with:

```powershell
python -m pip freeze > models\glot500m\environment_seed42.txt
```

## 10. Troubleshooting

| Symptom | Suggested check or action |
|---|---|
| `ModuleNotFoundError: transformers` | Confirm `.venv` is active; install `requirements-glot500.txt` into that environment. |
| Tokenizer says `protobuf` is missing | Run `python -m pip install protobuf`; then retry. |
| `CUDA: False` | Check `nvidia-smi`, the NVIDIA driver, and installed PyTorch variant. |
| CUDA out of memory | Use microbatch 1 and gradient checkpointing; close other GPU programs; reduce sequence length in a single-run test if necessary. |
| `FileNotFoundError` for workbook | Put `citizen_feedback_split.xlsx` under `data/splits/`, or pass `--workbook` with its actual path. |
| All scores at threshold `0.50` are zero | Check `val_macro_f1@tuned` and per-category diagnostics before interpreting the model as unable to learn. |
| Tuner says it is reusing a trial | Existing `acd_results.json` was found; choose a fresh `--output-root` for a true retraining run. |
| Network/DNS error downloading checkpoint | Verify access to Hugging Face; retry on a working network. |
| A winning trial lacks model weights | Inspect whether that trial's weights were pruned or its files moved; use an unpruned new search if needed. |

## 11. Completion criteria and next stage

The current ACD tuning stage is complete when:

- [ ] `--check-data` passes with no overlap between Training and Validation comments.
- [ ] A full three-trial run completes and writes `search_results.csv`.
- [ ] All 12 sampled trials finish successfully and `best_configuration.json` is saved.
- [ ] The winning checkpoint's model files and `acd_results.json` exist.
- [ ] Validation per-category diagnostics and rare-label performance are reviewed.
- [ ] Experiment settings, package versions, and Git commit are recorded.

**After that:** Finalize ACD model-selection procedures, perform the required additional validation/repeated-seed experiments, and develop **ACSA independently from the original `cis-lmu/glot500-base` checkpoint**. Keep the Testing sheet untouched until the final evaluation protocol and model-selection decision are frozen.

### Methodology note

The current tuner uses the fixed **Training/Validation** split, not the manuscript's full nested five-outer/three-inner-fold group-isolated cross-validation. It also uses the *same validation set* to choose checkpoints, thresholds, and the best hyperparameters, so the reported validation score can be optimistic. These scripts are for **development and tuning**, not the paper's final independent evaluation. If the research manuscript requires the full nested protocol, it must be implemented separately or the written protocol must accurately describe the experiments performed.
