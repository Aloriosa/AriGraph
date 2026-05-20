#!/usr/bin/env python
"""
End‑to‑end reproduction of a toy forecasting pipeline.
Author: OpenAI – example adaptation of the paper “What Will My Model Forget?”
"""

import json
import random
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LEN = 128
BATCH_SIZE = 8
FINETUNE_STEPS = 5
ENCODER_HIDDEN = 64
LR = 1e-4
NUM_EPOCHS = 3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    return tokenizer, model

def encode_batch(tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Tokenise a list of texts."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(DEVICE)

def get_logits_and_preds(model, batch):
    """Return logits and predicted labels for a batch."""
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
    return logits, preds

# ------------------------------------------------------------
# Representation encoder
# ------------------------------------------------------------
class Encoder(nn.Module):
    """
    Simple MLP that projects a [CLS] token embedding into a compact space.
    """
    def __init__(self, input_dim: int, hidden_dim: int = ENCODER_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)  # shape: (batch, hidden_dim)

# ------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------
def build_forward_pairs(
    online_data,
    upstream_data,
    model,
    tokenizer,
    tokenizer_upstream,
    device=DEVICE,
):
    """
    For each online example, fine‑tune the base model for a few steps,
    then compute whether each upstream example becomes incorrect.
    Returns a list of tuples: (online_text, upstream_text, label)
    """
    pairs = []
    # Pre‑compute upstream logits before any updates
    upstream_inputs = encode_batch(tokenizer_upstream, upstream_data["sentence"])
    _, upstream_preds_before = get_logits_and_preds(model, upstream_inputs)
    upstream_logits_before = get_logits_and_preds(model, upstream_inputs)[0]

    for idx, online in enumerate(tqdm(online_data, desc="Processing online examples")):
        # Fine‑tune on this single example
        online_inputs = encode_batch(tokenizer, [online["sentence"]])
        online_label = torch.tensor([online["label"]]).to(device)

        # Simple SGD loop
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        for _ in range(FINETUNE_STEPS):
            optimizer.zero_grad()
            outputs = model(**online_inputs, labels=online_label)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # After update, compute logits on upstream examples
        _, upstream_preds_after = get_logits_and_preds(model, upstream_inputs)
        upstream_logits_after = get_logits_and_preds(model, upstream_inputs)[0]

        # Determine forgetting: label 1 if prediction changed from correct to incorrect
        for j, upstream in enumerate(upstream_data):
            correct_before = (
                upstream_preds_before[j] == upstream["label"]
            )
            correct_after = (
                upstream_preds_after[j] == upstream["label"]
            )
            forgotten = int(correct_before and not correct_after)
            pairs.append((online["sentence"], upstream["sentence"], forgotten))

        # Reset model to original weights for next online example
        model.load_state_dict(state_dict_orig)

    return pairs

def train_forecasting_model(
    pairs,
    tokenizer,
    device=DEVICE,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
):
    """
    Train a representation‑based forecasting model on the pairs.
    """
    # Prepare dataset
    texts_i = [p[0] for p in pairs]
    texts_j = [p[1] for p in pairs]
    labels = np.array([p[2] for p in pairs], dtype=np.float32)

    # Tokenise
    inputs_i = encode_batch(tokenizer, texts_i)
    inputs_j = encode_batch(tokenizer, texts_j)

    # Get [CLS] embeddings from base model
    with torch.no_grad():
        cls_i = model(**inputs_i).logits  # shape: (N, 2)
        # We need hidden states: use the base encoder
        hidden_state_i = model.base_model(**inputs_i).last_hidden_state[:, 0, :]  # (N, hidden)
        hidden_state_j = model.base_model(**inputs_j).last_hidden_state[:, 0, :]

    # Build encoder
    encoder = Encoder(input_dim=hidden_state_i.size(-1)).to(device)

    # Frequency prior per upstream example
    # Count how many times each upstream example was forgotten
    # Since we have duplicate upstream sentences, we group by the string
    from collections import Counter

    upstream_counts = Counter(texts_j)
    prior_counts = Counter()
    for idx, text in enumerate(texts_j):
        if labels[idx] == 1:
            prior_counts[text] += 1

    prior_log_odds = {}
    for text in upstream_counts:
        p_forget = prior_counts[text] / upstream_counts[text]
        # avoid 0 or 1
        p_forget = min(max(p_forget, 1e-6), 1 - 1e-6)
        prior_log_odds[text] = np.log(p_forget / (1 - p_forget))

    # Training loop
    dataset = torch.utils.data.TensorDataset(
        hidden_state_i, hidden_state_j, torch.tensor(labels)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(list(encoder.parameters()), lr=LR)

    encoder.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            h_i, h_j, y = batch
            h_i = h_i.to(device)
            h_j = h_j.to(device)
            y = y.to(device)

            z = encoder(h_i) * encoder(h_j)  # element-wise product
            logits = z.sum(dim=1)  # dot product

            # Add frequency prior
            # For simplicity, we use a global bias term (average over all upstream)
            bias = torch.tensor(
                sum(prior_log_odds.values()) / len(prior_log_odds), device=device
            )
            logits_bias = logits + bias

            loss = criterion(logits_bias, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(loader):.4f}")

    return encoder, prior_log_odds

def evaluate_forecasting(
    encoder,
    prior_log_odds,
    pairs,
    tokenizer,
    device=DEVICE,
):
    """
    Evaluate the trained forecasting model on the test pairs.
    """
    texts_i = [p[0] for p in pairs]
    texts_j = [p[1] for p in pairs]
    labels = np.array([p[2] for p in pairs], dtype=np.float32)

    inputs_i = encode_batch(tokenizer, texts_i)
    inputs_j = encode_batch(tokenizer, texts_j)

    with torch.no_grad():
        hidden_state_i = model.base_model(**inputs_i).last_hidden_state[:, 0, :]
        hidden_state_j = model.base_model(**inputs_j).last_hidden_state[:, 0, :]
        h_i = encoder(hidden_state_i).cpu()
        h_j = encoder(hidden_state_j).cpu()
        dot = (h_i * h_j).sum(dim=1)
        bias = torch.tensor(
            sum(prior_log_odds.values()) / len(prior_log_odds)
        )
        logits = dot + bias
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }

def threshold_baseline(pairs, threshold=0.5):
    """
    Simple frequency‑threshold baseline:
    Predict that an upstream example will be forgotten if it was forgotten
    in more than 'threshold' fraction of online examples.
    """
    from collections import defaultdict

    # Count forgotten occurrences per upstream sentence
    counts = defaultdict(lambda: [0, 0])  # [forgotten, total]
    for (_, upstream, forgotten) in pairs:
        counts[upstream][0] += forgotten
        counts[upstream][1] += 1

    preds = []
    labels = []
    for (_, upstream, forgotten) in pairs:
        frac_forget = counts[upstream][0] / counts[upstream][1]
        pred = int(frac_forget >= threshold)
        preds.append(pred)
        labels.append(forgotten)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
if __name__ == "__main__":
    seed_everything()

    print("Loading tokenizer and base model...")
    tokenizer, model = get_tokenizer_and_model()
    state_dict_orig = model.state_dict()  # keep original weights

    print("Loading datasets (toy sentiment tasks)...")
    # Online examples: GLUE SST‑2 (sentiment)
    online_dataset = load_dataset("glue", "sst2", split="train[:200]")
    # Upstream examples: IMDb (sentiment)
    upstream_dataset = load_dataset("imdb", split="train[:200]")
    # For simplicity, we convert IMDb labels to 0/1 (already 0/1)
    # Ensure both datasets have a 'sentence' and 'label' field.

    # Build forward pairs (online × upstream)
    print("Constructing online × upstream pairs and computing forgetting labels...")
    pairs = build_forward_pairs(
        online_dataset,
        upstream_dataset,
        model,
        tokenizer,
        tokenizer,  # reuse same tokenizer
    )
    print(f"Total pairs: {len(pairs)}")

    # Shuffle and split into train/test
    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]

    print("Training representation‑based forecasting model...")
    encoder, prior_log_odds = train_forecasting_model(
        train_pairs, tokenizer
    )

    print("Evaluating on test set...")
    metrics = evaluate_forecasting(encoder, prior_log_odds, test_pairs, tokenizer)

    print("\n=== Forecasting results ===")
    print(f"Precision : {metrics['precision']:.3f}")
    print(f"Recall    : {metrics['recall']:.3f}")
    print(f"F1        : {metrics['f1']:.3f}")
    print(f"Accuracy  : {metrics['accuracy']:.3f}")

    # Baseline
    thresh_metrics = threshold_baseline(train_pairs, threshold=0.5)
    print("\n=== Frequency‑threshold baseline ===")
    print(f"Precision : {thresh_metrics['precision']:.3f}")
    print(f"Recall    : {thresh_metrics['recall']:.3f}")
    print(f"F1        : {thresh_metrics['f1']:.3f}")

    # Save results
    results = {
        "forecasting": metrics,
        "threshold": thresh_metrics,
        "num_pairs": len(pairs),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to results.json")