import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_batch(batch):
    return batch

def em_score(preds, targets):
    """Exact Match: 1 if exact string match else 0."""
    return np.mean([int(p.strip() == t.strip()) for p, t in zip(preds, targets)])

def compute_f1(preds, targets):
    return f1_score(targets, preds, zero_division=0)

def compute_precision(preds, targets):
    return precision_score(targets, preds, zero_division=0)

def compute_recall(preds, targets):
    return recall_score(targets, preds, zero_division=0)