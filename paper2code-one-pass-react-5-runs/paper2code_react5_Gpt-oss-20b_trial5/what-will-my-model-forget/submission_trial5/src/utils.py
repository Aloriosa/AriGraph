import random
from typing import List, Tuple, Dict
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()

def compute_em(preds: List[str], golds: List[str]) -> float:
    matches = sum(exact_match(p, g) for p, g in zip(preds, golds))
    return matches / len(golds)

def compute_f1(preds: List[int], golds: List[int]) -> float:
    return f1_score(golds, preds)

def compute_precision(preds: List[int], golds: List[int]) -> float:
    return precision_score(golds, preds)

def compute_recall(preds: List[int], golds: List[int]) -> float:
    return recall_score(golds, preds)