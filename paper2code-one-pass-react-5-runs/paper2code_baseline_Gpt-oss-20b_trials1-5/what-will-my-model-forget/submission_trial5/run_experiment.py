#!/usr/bin/env python
"""
Main script that reproduces the core experiments:
  1. Training a simple representation‑based forgetting predictor.
  2. Using the predictor to select examples for replay.
  3. Measuring Edit Success Rate and EM Drop Ratio.

The experiment uses a tiny subset of the P3‑train data
and a few MMLU (or other) examples for refinement.
"""
import os
import random
import math
import json
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Models
BASE_MODELS = {
    "bart": "facebook/bart-base",
    "t5": "google/flan-t5-base",
}

# Data
UPSTREAM_DATASET = "wikitext"   # small dataset used as upstream examples
UPSTREAM_SPLIT = "wikitext-2-raw-v1.train[:200]"  # 200 examples for speed

REFINEMENT_DATASET = "squad"  # we use a tiny subset of SQuAD for errors
REFINEMENT_NUM_EXAMPLES = 10  # number of online learning examples


# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset wrappers
# ------------------------------------------------------------
class Seq2SeqDataset(Dataset):
    """Simple wrapper around HuggingFace datasets for seq2seq tasks."""

    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # For simplicity, we treat the example as:
        #   input_text = context + question
        #   target_text = answer
        # This is a toy setup.
        if isinstance(ex, dict):
            input_text = ex["context"] + " " + ex["question"]
            target_text = ex["answers"]["text"][0]
        else:
            input_text = ex["sentence"]
            target_text = ex["label"]
        # Tokenize
        input_ids = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.squeeze()
        target_ids = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.squeeze()
        return {
            "input_ids": input_ids,
            "labels": target_ids,
            "original_text": input_text,
            "target_text": target_text,
        }


# ------------------------------------------------------------
# Forecasting model (representation‑based)
# ------------------------------------------------------------
class RepresentationForecaster(nn.Module):
    """
    Simple encoder (2‑layer MLP) that maps the pooled encoder
    representation of an example to a vector. The dot product
    between two such vectors predicts forgetting.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pooled_rep: torch.Tensor):
        return self.encoder(pooled_rep)

    def predict(self, repr_i, repr_j):
        """Return probability that j is forgotten when learning i."""
        logits = torch.matmul(repr_i, repr_j.T)  # (1, N)
        probs = torch.sigmoid(logits)
        return probs.squeeze()


# ------------------------------------------------------------
# Core experiment logic
# ------------------------------------------------------------
def load_and_prepare_datasets(tokenizer, model_name):
    # Upstream examples (pre‑training data)
    upstream_ds = load_dataset(UPSTREAM_DATASET, split=UPSTREAM_SPLIT)
    upstream_examples = [
        {"sentence": sent, "label": sent} for sent in upstream_ds["text"]
    ]
    upstream_dataset = Seq2SeqDataset(upstream_examples, tokenizer)

    # Online learning examples (errors)
    refine_ds = load_dataset(REFINEMENT_DATASET, split="train[:{}]".format(REFINEMENT_NUM_EXAMPLES))
    refine_dataset = Seq2SeqDataset(refine_ds, tokenizer)

    return upstream_dataset, refine_dataset


def cache_logits_and_repr(model, dataset):
    """Run model once to cache logits and pooled representations."""
    model.eval()
    logits_cache = []
    repr_cache = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=8), desc="Caching logits"):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )
            # logits: (B, seq_len, vocab_size)
            logits_cache.append(outputs.logits.cpu())
            # pooled representation: use the hidden state of the first token
            # after passing through the encoder
            # For simplicity, we use the encoder's last hidden state of [CLS] (index 0)
            # Note: for T5/BART, the first token is a special token, but this is fine for demo
            outputs = model.module if hasattr(model, "module") else model
            encoder = outputs.encoder
            encoder_outputs = encoder(input_ids)
            pooled = encoder_outputs.last_hidden_state[:, 0, :]  # (B, hidden_dim)
            repr_cache.append(pooled.cpu())
    return logits_cache, repr_cache


def get_pooled_representation(model, tokenizer, text, max_len=128):
    """Return the pooled representation for a single example."""
    inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    with torch.no_grad():
        encoder = model.module.encoder if hasattr(model, "module") else model.encoder
        encoder_outputs = encoder(input_ids)
        pooled = encoder_outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
    return pooled.squeeze(0)  # (hidden_dim,)


def fine_tune_example(model, tokenizer, example, lr=1e-5, epochs=1):
    """Fine‑tune the model on a single example."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=epochs * len(example["input_ids"])
    )
    inputs = example["input_ids"].unsqueeze(0).to(DEVICE)
    labels = example["labels"].unsqueeze(0).to(DEVICE)
    for _ in range(epochs):
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return model


def compute_forgetting(model_before, model_after, upstream_dataset, tokenizer):
    """
    Return a dict mapping index -> bool indicating whether the upstream example
    was forgotten (prediction changed) after the update.
    """
    model_before.eval()
    model_after.eval()
    forgetting = {}
    for idx, ex in enumerate(tqdm(upstream_dataset, desc="Computing forgetting")):
        input_ids = ex["input_ids"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_before = model_before.generate(input_ids, max_new_tokens=32)
            out_after = model_after.generate(input_ids, max_new_tokens=32)
        pred_before = tokenizer.decode(out_before[0], skip_special_tokens=True).strip()
        pred_after = tokenizer.decode(out_after[0], skip_special_tokens=True).strip()
        # For a toy setup, we consider forgetting if the generated text changed
        forgotten = pred_before != pred_after
        forgetting[idx] = forgotten
    return forgetting


def train_forecaster(
    model, tokenizer, upstream_dataset, refine_dataset, epochs=5, lr=1e-5
):
    """
    Train the representation‑based forecaster on pairs (i, j)
    where i is an online example and j is an upstream example.
    """
    forecaster = RepresentationForecaster().to(DEVICE)
    optimizer = torch.optim.AdamW(forecaster.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    # Pre‑compute pooled representations for upstream examples
    upstream_reprs = []
    for ex in tqdm(upstream_dataset, desc="Pre‑computing upstream reps"):
        pool = get_pooled_representation(model, tokenizer, ex["original_text"])
        upstream_reprs.append(pool)

    # Training loop
    for epoch in range(epochs):
        for _ in tqdm(range(len(refine_dataset)), desc=f"Epoch {epoch+1}"):
            # Sample a random online example
            ex = random.choice(refine_dataset)
            # Fine‑tune on this example
            model_finetuned = fine_tune_example(
                model, tokenizer, ex, lr=lr, epochs=1
            )
            # Compute forgetting on all upstream examples
            forgetting = compute_forgetting(model, model_finetuned, upstream_dataset, tokenizer)

            # For each upstream example, create a training pair
            for j_idx, is_forgotten in forgetting.items():
                # Get representations
                rep_i = get_pooled_representation(model, tokenizer, ex["original_text"])
                rep_j = upstream_reprs[j_idx]
                # Forward
                logits = forecaster.predict(rep_i.unsqueeze(0), rep_j.unsqueeze(0))
                loss = bce(logits, torch.tensor([float(is_forgotten)], device=DEVICE))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return forecaster


def evaluate_forecaster(
    forecaster, model, tokenizer, upstream_dataset, refine_dataset
):
    """
    Evaluate the forecaster on a held‑out online example.
    Return F1, precision, recall.
    """
    # Use the first example in refine_dataset as a test
    ex = refine_dataset[0]
    # Fine‑tune on this example
    model_finetuned = fine_tune_example(model, tokenizer, ex, lr=1e-5, epochs=1)

    # Compute ground truth forgetting
    gt_forgetting = compute_forgetting(model, model_finetuned, upstream_dataset, tokenizer)

    preds = []
    gts = []
    for j_idx, _ in enumerate(upstream_dataset):
        rep_i = get_pooled_representation(model, tokenizer, ex["original_text"])
        rep_j = get_pooled_representation(model, tokenizer, upstream_dataset[j_idx]["original_text"])
        prob = forecaster.predict(rep_i.unsqueeze(0), rep_j.unsqueeze(0))
        pred = prob > 0.5
        preds.append(int(pred))
        gts.append(int(gt_forgetting[j_idx]))
    f1 = f1_score(gts, preds)
    prec = precision_score(gts, preds)
    rec = recall_score(gts, preds)
    return f1, prec, rec


def replay_and_evaluate(
    forecaster, model, tokenizer, upstream_dataset, refine_dataset, num_replay=8
):
    """
    Perform model refinement with replay of examples predicted to be forgotten.
    Return edit success rate and EM drop ratio.
    """
    # Edit success: after fine‑tuning on the online example, is the prediction correct?
    # For the toy setup, we consider success if the generated text equals the target
    # before the update (i.e., we fix the error).
    success_counts = 0
    total = len(refine_dataset)

    # EM before refinement: accuracy on upstream dataset
    def em(model, dataset):
        model.eval()
        correct = 0
        for ex in tqdm(dataset, desc="EM eval"):
            input_ids = ex["input_ids"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=32)
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            tgt = ex["target_text"].strip()
            if pred == tgt:
                correct += 1
        return correct / len(dataset)

    em_before = em(model, upstream_dataset)

    for ex in refine_dataset:
        # Fine‑tune on the online example
        model = fine_tune_example(model, tokenizer, ex, lr=1e-5, epochs=1)

        # Check edit success
        input_ids = ex["input_ids"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=32)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        tgt = ex["target_text"].strip()
        if pred == tgt:
            success_counts += 1

        # Replay: pick num_replay upstream examples predicted to be forgotten
        # by the forecaster on this online example
        rep_i = get_pooled_representation(model, tokenizer, ex["original_text"])
        candidates = []
        for j_idx, up_ex in enumerate(upstream_dataset):
            rep_j = get_pooled_representation(model, tokenizer, up_ex["original_text"])
            prob = forecaster.predict(rep_i.unsqueeze(0), rep_j.unsqueeze(0))
            if prob > 0.5:
                candidates.append(j_idx)
        # Randomly sample if too many
        if len(candidates) > num_replay:
            candidates = random.sample(candidates, num_replay)

        # Fine‑tune on the replayed examples (one step each)
        for idx in candidates:
            up_ex = upstream_dataset[idx]
            model = fine_tune_example(
                model, tokenizer, up_ex, lr=1e-5, epochs=1
            )

    em_after = em(model, upstream_dataset)
    em_drop = (em_before - em_after) / em_before * 100
    success_rate = success_counts / total * 100
    return success_rate, em_drop


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def main():
    # Load base model and tokenizer
    model_name = BASE_MODELS["bart"]
    print(f"Loading base model {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare data
    upstream_ds, refine_ds = load_and_prepare_datasets(tokenizer, model_name)

    # Train forecaster
    print("Training forecaster...")
    forecaster = train_forecaster(
        model, tokenizer, upstream_ds, refine_ds, epochs=2, lr=1e-5
    )

    # Evaluate forecaster
    print("Evaluating forecaster...")
    f1, prec, rec = evaluate_forecaster(forecaster, model, tokenizer, upstream_ds, refine_ds)
    print(f"Forecaster F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")

    # Replay-based refinement
    print("Running replay-based refinement...")
    success_rate, em_drop = replay_and_evaluate(
        forecaster, model, tokenizer, upstream_ds, refine_ds, num_replay=4
    )
    print(f"Edit Success Rate: {success_rate:.1f}%")
    print(f"EM Drop Ratio: {em_drop:.2f}%")

    # Save results
    results = {
        "forecaster": {"f1": f1, "precision": prec, "recall": rec},
        "replay": {"edit_success_rate": success_rate, "em_drop_ratio": em_drop},
    }
    Path("results.json").write_text(json.dumps(results, indent=2))
    print("Results written to results.json")


if __name__ == "__main__":
    main()