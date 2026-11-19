# Glot500-m ACD-first fine-tuning (Windows / VS Code)

This package adds only `models/glot500m/train_acd.py`. It **reuses** the existing repository's `shared/data.py` and `shared/labels.py`; no replacement is needed. It does not train ACSA or touch the Testing worksheet.

## Install into project

1. Copy `models/glot500m/train_acd.py` into the same relative folder of `C:\college\4th year_first sem\citi-senti-nlp`.
2. Put `citizen_feedback_split.xlsx` at `data\splits\citizen_feedback_split.xlsx`.
3. Open the project in VS Code, select the `.venv` interpreter, and use a PowerShell terminal.

## Commands

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip check
python models/glot500m/train_acd.py --check-data
python models/glot500m/train_acd.py --batch-size 1 --gradient-accumulation 8 --gradient-checkpointing
```

Recommended first run on an NVIDIA CUDA GPU. If you have ample GPU memory, you can increase `--batch-size` to 2 or 4 and adjust gradient accumulation for an effective batch of 8. This first run saves the best validation epoch, then tunes a global ACD decision threshold on validation data only.

## What gets saved

At `models/glot500m/artifacts/acd/seed42/`:
- `model.safetensors`, `config.json`, tokenizer files: best ACD checkpoint
- `acd_results.json`: epoch history, validation metrics and tuned thresholds

## Methodology alignment

- 11-label multi-label ACD.
- Pretrained `cis-lmu/glot500-base` encoder with newly initialized classification head.
- BCE-with-logits, AdamW, learning rate 2e-5, maximum 5 epochs, weight decay .01, warmup .10, early stopping patience 2.
- Best epoch chosen using 11-label validation Macro-F1 at threshold 0.5.
- After selecting the epoch, thresholds tuned on validation data (global by default; optional labelwise mode requires >=20 positives and >=20 negatives).
- **This is a first single-run implementation, NOT the full 5×3 nested validation, 12-configuration random search, or five-seed experimental protocol.** Those remain for the final methodology-compliant model-selection/evaluation stage.
- Does not read or use Testing labels; keep the held-out test untouched until evaluation is frozen.

## Troubleshooting

- GPU out of memory: keep `--batch-size 1`, use `--gradient-checkpointing` and `--max-length 128`.
- Model download failure: check access to huggingface.co and your Hugging Face cache.
- `ModuleNotFoundError`: verify VS Code uses `.venv` and install `transformers==4.57.6`, PyTorch, `sentencepiece`, `protobuf`, and scikit-learn inside it.
