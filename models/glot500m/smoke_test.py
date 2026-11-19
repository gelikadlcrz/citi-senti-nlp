import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# I. Model Configuration

MODEL_ID = "cis-lmu/glot500-base"
NUM_LABELS = 11

# II. Load Tokenizer

print("Loading Glot500-m tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# III. Load Pretrained Model

print("Loading Glot500-m model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

# IV. Prepare Sample Input

text = "Okay met daytoy policy but mahal yung penalty."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128,
)

# V. Run Model Inference

model.eval()

with torch.inference_mode():
    outputs = model(**inputs)

# VI. Verify Results

print("\n--- Environment Information ---")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n--- Model Output ---")
print("Input shape:", inputs["input_ids"].shape)
print("Output shape:", outputs.logits.shape)

assert outputs.logits.shape == (1, 11)

print("\nGlot500-m setup successful!")
