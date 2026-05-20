import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

# --------------------------------------------------------------------------- #
# Random seeds (for reproducibility)
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------------------- #
# Helper: load dataset
# --------------------------------------------------------------------------- #
def load_sst2(split="train"):
    """Load SST‑2 from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split=split)
    return ds

# --------------------------------------------------------------------------- #
# Helper: save/load pickles
# --------------------------------------------------------------------------- #
def save_pickle(obj, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

# --------------------------------------------------------------------------- #
# Helper: simple collate for BERT
# --------------------------------------------------------------------------- #
def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc["labels"] = torch.tensor(labels)
    return enc

# --------------------------------------------------------------------------- #
# Helper: compute predictions (argmax)
# --------------------------------------------------------------------------- #
def predict_logits(model, dataloader, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)

# --------------------------------------------------------------------------- #
# Helper: compute metrics
# --------------------------------------------------------------------------- #
def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": (preds == labels).mean(),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }

# --------------------------------------------------------------------------- #
# Helper: create embedding matrix for a dataset
# --------------------------------------------------------------------------- #
def get_embeddings(model, dataset, device, batch_size=32):
    """Return CLS embeddings for each example in dataset."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    all_emb = []
    model.eval()
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
            # CLS token is the first token
            cls_emb = outputs.hidden_states[-1][:, 0, :]
            all_emb.append(cls_emb.cpu())
    return torch.cat(all_emb, dim=0)  # (N, hidden_size)

# --------------------------------------------------------------------------- #
# Helper: simple logit‑change feature
# --------------------------------------------------------------------------- #
def logit_change_feature(logit_before, logit_after):
    """Return L2 norm of logit difference."""
    return torch.norm(logit_after - logit_before).item()