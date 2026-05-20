#!/usr/bin/env python3
"""
Fine‑tune BERT‑Base on SST‑2 train split.
Saves model, tokenizer and predictions on validation set.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from utils import SEED, collate_fn, compute_metrics, save_pickle

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "outputs/base_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
train_ds = load_dataset("glue", "sst2", split="train")
valid_ds = load_dataset("glue", "sst2", split="validation")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(preprocess_function, batched=True)
valid_ds = valid_ds.map(preprocess_function, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="epoch",
    seed=SEED,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Get validation predictions
preds = trainer.predict(valid_ds)
labels = preds.label_ids
pred_logits = preds.predictions

# Identify online errors (mis‑predicted val examples)
online_indices = [i for i, (p, l) in enumerate(zip(pred_logits, labels)) if torch.argmax(p).item() != l]
online_examples = [valid_ds[i] for i in online_indices]

# Save relevant data
save_pickle({"indices": online_indices, "examples": online_examples}, os.path.join(OUTPUT_DIR, "online_errors.pkl"))
print(f"Saved {len(online_indices)} online errors to {OUTPUT_DIR}/online_errors.pkl")