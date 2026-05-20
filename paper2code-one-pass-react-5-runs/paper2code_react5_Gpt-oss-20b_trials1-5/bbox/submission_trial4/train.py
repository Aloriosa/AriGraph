#!/usr/bin/env python3
"""
Training and evaluation script for BBox‑Adapter.

Usage:
    python train.py --dataset gsm8k --adapter_size distilbert-base-uncased \
                    --epochs 3 --k 5 --batch_size 8
"""

import argparse
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Local modules
from adapter import Adapter
from data_utils import get_dataset
from prompt_templates import (
    gsm8k_prompt,
    strategyqa_prompt,
    truthfulqa_prompt,
    scienceqa_prompt,
)
from utils import gsm8k_accuracy, exact_match
from ai_feedback import GPT2Scorer

# -------------------- Argparse --------------------
parser = argparse.ArgumentParser(description="BBox‑Adapter training")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["gsm8k", "strategyqa", "truthfulqa", "scienceqa"])
parser.add_argument("--adapter_size", type=str, default="distilbert-base-uncased",
                    help="DistilBERT model name for the adapter")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--k", type=int, default=5, help="Number of candidates per query")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--max_length", type=int, default=200)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# -------------------- Data loading --------------------
train_data, dev_data, test_data = get_dataset(args.dataset)

# Simple PyTorch dataset wrapper
class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        # field names differ across datasets – normalize
        if "question" in row:
            q = row["question"]
        else:
            q = row["question"]
        if "answer" in row:
            a = row["answer"]
        else:
            a = row["answer"]
        return {"question": q, "answer": a}

train_ds = QADataset(train_data)
dev_ds = QADataset(dev_data)
test_ds = QADataset(test_data)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)
test_loader = DataLoader(test_ds, batch_size=args.batch_size)

# -------------------- Model and tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT‑2 has no pad token

generator = AutoModelForCausalLM.from_pretrained("gpt2").to(args.device)
generator.eval()

adapter = Adapter(args.adapter_size).to(args.device)
optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# -------------------- Prompt helper --------------------
def get_prompt(dataset, question, examples=None):
    if dataset == "gsm8k":
        return gsm8k_prompt(question, examples)
    if dataset == "strategyqa":
        return strategyqa_prompt(question, examples)
    if dataset == "truthfulqa":
        return truthfulqa_prompt(question, examples)
    if dataset == "scienceqa":
        return scienceqa_prompt(question, examples)
    raise ValueError(f"Unknown dataset {dataset}")

# -------------------- Training loop --------------------
print(f"Training on {args.dataset} with {args.k} candidates per query")
for epoch in range(args.epochs):
    adapter.train()
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        questions = batch["question"]
        positives = batch["answer"]

        # Generate K candidates for each question
        candidates = []
        for q in questions:
            prompt = get_prompt(args.dataset, q)
            # GPT‑2 generation
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = generator.generate(
                **inputs,
                max_length=args.max_length,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                num_return_sequences=args.k,
                pad_token_id=generator.config.eos_token_id,
            )
            texts = [tokenizer.decode(o, skip_special_tokens=True)
                     for o in outputs]
            # Keep only the part after the prompt (approximate)
            texts = [t[len(prompt):].strip() for t in texts]
            candidates.append(texts)

        # Build inputs for the adapter
        flat_inputs = []
        for q, cand_list in zip(questions, candidates):
            for cand in cand_list:
                full = q + " " + cand
                flat_inputs.append(full)

        tokenized = tokenizer(
            flat_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(args.device)

        logits = adapter(
            tokenized["input_ids"],
            tokenized["attention_mask"],
        )  # shape (batch*K,)

        # Reshape to (batch, k)
        logits = logits.view(len(questions), -1)

        # Construct target indices: find which candidate matches the ground‑truth
        targets = []
        for i, pos_ans in enumerate(positives):
            idx = 0
            # choose first candidate that contains the true answer (case‑insensitive)
            for j, cand in enumerate(candidates[i]):
                if pos_ans.lower() in cand.lower():
                    idx = j
                    break
            targets.append(idx)
        targets = torch.tensor(targets, dtype=torch.long, device=args.device)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} - loss: {epoch_loss / len(train_loader):.4f}")

    # Validation
    adapter.eval()
    with torch.no_grad():
        preds, golds = [], []
        for batch in dev_loader:
            questions = batch["question"]
            positives = batch["answer"]
            for q, pos_ans in zip(questions, positives):
                prompt = get_prompt(args.dataset, q)
                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                outputs = generator.generate(
                    **inputs,
                    max_length=args.max_length,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    num_return_sequences=args.k,
                    pad_token_id=generator.config.eos_token_id,
                )
                texts = [tokenizer.decode(o, skip_special_tokens=True)
                         for o in outputs]
                texts = [t[len(prompt):].strip() for t in texts]

                # Score each candidate
                flat = [q + " " + t for t in texts]
                tokenized = tokenizer(
                    flat,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(args.device)
                scores = adapter(
                    tokenized["input_ids"],
                    tokenized["attention_mask"],
                ).squeeze(-1)
                best_idx = torch.argmax(scores).item()
                preds.append(texts[best_idx])
                golds.append(pos_ans)

        if args.dataset == "gsm8k":
            acc = gsm8k_accuracy(preds, golds)
        else:
            acc = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(preds)
        print(f"  Validation accuracy: {acc:.4f}")

# -------------------- Test evaluation --------------------
print("\n=== Test evaluation ===")
adapter.eval()
with torch.no_grad():
    preds, golds = [], []
    for batch in tqdm(test_loader, desc="Testing"):
        questions = batch["question"]
        positives = batch["answer"]
        for q, pos_ans in zip(questions, positives):
            prompt = get_prompt(args.dataset, q)
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = generator.generate(
                **inputs,
                max_length=args.max_length,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                num_return_sequences=args.k,
                pad_token_id=generator.config.eos_token_id,
            )
            texts = [tokenizer.decode(o, skip_special_tokens=True)
                     for o in outputs]
            texts = [t[len(prompt):].strip() for t in texts]

            flat = [q + " " + t for t in texts]
            tokenized = tokenizer(
                flat,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(args.device)
            scores = adapter(
                tokenized["input_ids"],
                tokenized["attention_mask"],
            ).squeeze(-1)
            best_idx = torch.argmax(scores).item()
            preds.append(texts[best_idx])
            golds.append(pos_ans)

    if args.dataset == "gsm8k":
        acc = gsm8k_accuracy(preds, golds)
    else:
        acc = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(preds)
    print(f"Test accuracy (adapted): {acc:.4f}")

# -------------------- Baseline: GPT‑2 only --------------------
print("\n=== Baseline: GPT‑2 only ===")
baseline_acc = 0.0
with torch.no_grad():
    preds, golds = [], []
    for batch in tqdm(test_loader, desc="Baseline"):
        questions = batch["question"]
        positives = batch["answer"]
        for q, pos_ans in zip(questions, positives):
            prompt = get_prompt(args.dataset, q)
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = generator.generate(
                **inputs,
                max_length=args.max_length,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=generator.config.eos_token_id,
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = text[len(prompt):].strip()
            preds.append(text)
            golds.append(pos_ans)

    if args.dataset == "gsm8k":
        acc = gsm8k_accuracy(preds, golds)
    else:
        acc = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(preds)
    print(f"Test accuracy (GPT‑2 baseline): {acc:.4f}")