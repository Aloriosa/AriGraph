#!/usr/bin/env python3
"""
For each online error, fine‑tune the base model on that single example for a few steps,
then record which training examples become forgotten.
The results are stored in `outputs/refinement.pkl`.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from utils import SEED, collate_fn, compute_metrics, load_pickle, save_pickle
import numpy as np

BASE_DIR = "outputs/base_model"
REFINE_DIR = "outputs/refinement"
os.makedirs(REFINE_DIR, exist_ok=True)

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(BASE_DIR, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)

# Load datasets
train_ds = load_dataset("glue", "sst2", split="train")
valid_ds = load_dataset("glue", "sst2", split="validation")

# Tokenize everything
def tokenize(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
valid_ds = valid_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load online errors
online_data = load_pickle(os.path.join(BASE_DIR, "online_errors.pkl"))
online_indices = online_data["indices"]
online_examples = online_data["examples"]

# Pre‑compute training logits (baseline)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
baseline_logits = []
with torch.no_grad():
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        baseline_logits.append(outputs.logits.cpu())
baseline_logits = torch.cat(baseline_logits, dim=0)  # (N_train, 2)

# Ground‑truth labels for training set
train_labels = torch.tensor(train_ds["label"])

# For each online error, fine‑tune on that example, then compute forgotten examples
forgetting_records = []  # list of dicts: {"online_idx": int, "forgotten": list of train indices}

for o_idx, example in enumerate(online_examples):
    # Create a tiny dataset with one example
    single_ds = [(example["sentence"], example["label"])]
    single_ds = [{"sentence": s, "label": l} for s, l in single_ds]
    single_ds = load_dataset("json", data_files={"train": single_ds})["train"]
    single_ds = single_ds.map(tokenize, batched=True)
    single_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Fine‑tune for 5 gradient steps
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp",  # dummy
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            learning_rate=5e-5,
            weight_decay=0.01,
            fp16=True,
            seed=SEED,
            logging_steps=10,
            disable_tqdm=True,
        ),
        train_dataset=single_ds,
        eval_dataset=None,
    )
    trainer.train()
    # After update, evaluate all training examples
    model.eval()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    updated_logits = []
    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            updated_logits.append(outputs.logits.cpu())
    updated_logits = torch.cat(updated_logits, dim=0)

    # Determine forgotten examples: were correct before, wrong after
    baseline_pred = torch.argmax(baseline_logits, dim=1)
    updated_pred = torch.argmax(updated_logits, dim=1)
    correct_before = baseline_pred == train_labels
    correct_after = updated_pred == train_labels
    forgotten = (correct_before & (~correct_after)).nonzero(as_tuple=False).squeeze().tolist()
    if isinstance(forgotten, int):
        forgotten = [forgotten]
    forgetting_records.append({"online_idx": o_idx, "forgotten": forgotten})

# Save results
save_pickle({
    "records": forgetting_records,
    "train_labels": train_labels.numpy().tolist(),
    "baseline_logits": baseline_logits.numpy().tolist(),
}, os.path.join(REFINE_DIR, "refinement.pkl"))
print(f"Refinement data saved to {os.path.join(REFINE_DIR, 'refinement.pkl')}")