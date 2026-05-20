"""
Minimal reproduction of the forecasting framework from the paper.
Author: OpenAI ChatGPT
"""

import os
import json
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_SEED = 42
NUM_UPSTREAM = 100   # Number of upstream pre‑training examples
NUM_ONLINE   = 100   # Number of online error examples (from validation)
NUM_FINE_TUNE_STEPS = 10  # Gradient steps per online example
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()

def greedy_decode(outputs, tokenizer):
    """Greedy decode from model logits."""
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def evaluate(model, tokenizer, dataset):
    """Return predictions and accuracy on a dataset."""
    model.eval()
    preds = []
    gts   = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE):
            inputs = tokenizer(
                batch["question"], batch["context"], truncation=True, padding=True, return_tensors="pt"
            ).to(DEVICE)
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=1,
                early_stopping=True,
            )
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gts.extend([ex["answers"]["text"][0] for ex in batch])
            preds.extend(pred_texts)
    exact_matches = [exact_match(p, g) for p, g in zip(preds, gts)]
    return preds, np.mean(exact_matches)

def fine_tune_single_example(model, tokenizer, example, steps=NUM_FINE_TUNE_STEPS):
    """Fine‑tune the model on a single example for a few gradient steps."""
    # Build a tiny dataset with the example repeated
    class SingleExampleDataset(Dataset):
        def __len__(self): return steps
        def __getitem__(self, idx): return example
    dataset = SingleExampleDataset()
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE)
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=steps
    )
    model.train()
    for _ in tqdm(loader, total=steps, leave=False, desc="Fine‑tune"):
        inputs = tokenizer(
            example["question"], example["context"], truncation=True, padding=True, return_tensors="pt"
        ).to(DEVICE)
        labels = tokenizer(example["answers"]["text"][0], truncation=True, padding=True, return_tensors="pt").input_ids.to(DEVICE)
        labels[labels == tokenizer.pad_token_id] = -100  # ignore pad
        outputs = model(
            **inputs,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    model.eval()
    return model

# --------------------------------------------------------------------------- #
# Dataset wrappers
# --------------------------------------------------------------------------- #
class SQuADDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --------------------------------------------------------------------------- #
# Forecasting model
# --------------------------------------------------------------------------- #
class RepresentationForecaster(nn.Module):
    """
    Simple representation‑based forecasting model.
    Encodes each example with a small MLP applied to the mean encoder
    and decoder hidden states. Predicts forgetting via dot product + bias.
    """
    def __init__(self, hidden_dim=512, enc_dim=1024):
        super().__init__()
        self.enc_dim = enc_dim
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def encode(self, model, tokenizer, example):
        # Encode input and target
        with torch.no_grad():
            inputs = tokenizer(
                example["question"], example["context"], truncation=True, padding=True, return_tensors="pt"
            ).to(DEVICE)
            # Encoder hidden states
            encoder_outputs = model.get_encoder()(inputs.input_ids, attention_mask=inputs.attention_mask)
            enc_repr = encoder_outputs.last_hidden_state.mean(dim=1)  # [1, enc_dim]
            # Decoder hidden states (use the ground‑truth target)
            tgt_ids = tokenizer(example["answers"]["text"][0], truncation=True, padding=True, return_tensors="pt").input_ids.to(DEVICE)
            decoder_outputs = model.get_decoder()(tgt_ids, encoder_hidden_states=encoder_outputs.last_hidden_state, encoder_attention_mask=inputs.attention_mask)
            dec_repr = decoder_outputs.last_hidden_state.mean(dim=1)  # [1, enc_dim]
            # Concatenate
            repr_vec = torch.cat([enc_repr, dec_repr], dim=-1).squeeze(0)  # [enc_dim*2]
        return repr_vec

    def forward(self, vec_i, vec_j):
        # Dot product + bias
        score = torch.dot(vec_i, vec_j) + self.bias
        prob  = torch.sigmoid(score)
        return prob

# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    set_seed()
    # Load data
    squad = load_dataset("squad", split={"train": "train[0:200]", "validation": "validation[0:200]"})
    # Upstream pre‑training data (D_PT)
    upstream = SQuADDataset(squad["train"][:NUM_UPSTREAM])
    # Online error data (D_R)
    online   = SQuADDataset(squad["validation"][:NUM_ONLINE])

    # Load pre‑trained model
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large")
    base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(DEVICE)

    # ------------------------------------------------------------------ #
    # 1. Identify online errors (mis‑predicted examples)
    # ------------------------------------------------------------------ #
    preds, accuracy = evaluate(base_model, tokenizer, online)
    print(f"Base model accuracy on online set: {accuracy:.4f}")

    mis_indices = [i for i, (p, ex) in enumerate(zip(preds, online)) if not exact_match(p, ex["answers"]["text"][0])]
    print(f"Number of mis‑predicted online examples: {len(mis_indices)}")

    # ------------------------------------------------------------------ #
    # 2. For each mis‑predicted example, fine‑tune the model
    #    and record which upstream examples are forgotten.
    # ------------------------------------------------------------------ #
    # Cache representations for upstream examples
    print("Encoding upstream examples...")
    upstream_repr = []
    for ex in tqdm(upstream, desc="Cache upstream"):
        vec = RepresentationForecaster().encode(base_model, tokenizer, ex)
        upstream_repr.append(vec)

    # Prepare training pairs for forecasting
    pairs = []  # list of (vec_i, vec_j, label)
    print("Generating training pairs...")
    for idx in mis_indices:
        online_ex = online[idx]
        # Fine‑tune on this example
        finetuned_model = fine_tune_single_example(base_model.clone(), tokenizer, online_ex)
        # Evaluate upstream examples on finetuned model
        preds_pt, _ = evaluate(finetuned_model, tokenizer, upstream)
        # Determine forgetting
        for j, (ex, vec_j) in enumerate(zip(upstream, upstream_repr)):
            before = exact_match(preds[j], ex["answers"]["text"][0])
            after  = exact_match(preds_pt[j], ex["answers"]["text"][0])
            label  = int(before and not after)
            vec_i  = RepresentationForecaster().encode(finetuned_model, tokenizer, online_ex)
            pairs.append((vec_i, vec_j, label))

    # ------------------------------------------------------------------ #
    # 3. Train representation‑based forecasting model
    # ------------------------------------------------------------------ #
    print("Training forecasting model...")
    forecaster = RepresentationForecaster(hidden_dim=256, enc_dim=1024).to(DEVICE)
    optimizer = AdamW(forecaster.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Prepare tensors
    vec_i_list = torch.stack([p[0] for p in pairs])
    vec_j_list = torch.stack([p[1] for p in pairs])
    labels     = torch.tensor([p[2] for p in pairs], dtype=torch.float32).to(DEVICE)

    # Simple epoch loop
    for epoch in range(5):
        perm = torch.randperm(len(pairs))
        epoch_loss = 0.0
        for start in range(0, len(pairs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(pairs))
            idxs = perm[start:end]
            vi = vec_i_list[idxs].to(DEVICE)
            vj = vec_j_list[idxs].to(DEVICE)
            lbl = labels[idxs]
            probs = forecaster(vi.squeeze(1), vj.squeeze(1))
            loss = criterion(probs, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idxs)
        epoch_loss /= len(pairs)
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Evaluate forecasting model
    # ------------------------------------------------------------------ #
    print("Evaluating forecasting model...")
    preds_forcast = []
    for vi, vj, lbl in zip(vec_i_list, vec_j_list, labels):
        prob = forecaster(vi.squeeze(0).unsqueeze(0), vj.squeeze(0).unsqueeze(0))
        pred = 1 if prob.item() > 0.5 else 0
        preds_forcast.append(pred)
    f1 = f1_score(labels.cpu().numpy(), preds_forcast)
    prec = precision_score(labels.cpu().numpy(), preds_forcast, zero_division=0)
    rec  = recall_score(labels.cpu().numpy(), preds_forcast, zero_division=0)
    print(f"Forecasting F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Compute EM drop ratio on upstream data
    # ------------------------------------------------------------------ #
    # Accuracy before fine‑tuning (using base model)
    _, acc_before = evaluate(base_model, tokenizer, upstream)
    # Accuracy after fine‑tuning on all mis‑examples sequentially
    finetuned_model = base_model.clone()
    for idx in mis_indices:
        finetuned_model = fine_tune_single_example(finetuned_model, tokenizer, online[idx])
    _, acc_after = evaluate(finetuned_model, tokenizer, upstream)
    em_drop = (acc_before - acc_after) / acc_before * 100
    print(f"EM drop ratio on upstream data: {em_drop:.2f}%")

    # ------------------------------------------------------------------ #
    # 6. Save metrics
    # ------------------------------------------------------------------ #
    metrics = {
        "base_accuracy": accuracy,
        "mis_indices": len(mis_indices),
        "forecast_f1": f1,
        "forecast_precision": prec,
        "forecast_recall": rec,
        "em_drop_percent": em_drop,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics written to metrics.json")

if __name__ == "__main__":
    main()