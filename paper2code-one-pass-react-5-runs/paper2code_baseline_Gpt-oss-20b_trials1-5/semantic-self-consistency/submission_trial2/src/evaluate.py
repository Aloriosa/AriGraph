"""
Evaluation utilities.
"""

from typing import List, Dict, Tuple
import numpy as np


def compute_accuracy(preds: List[str], golds: List[str]) -> float:
    """Return accuracy as a percentage."""
    correct = sum(p == g for p, g in zip(preds, golds))
    return 100.0 * correct / len(golds)


def aggregate_results(
    dataset_name: str,
    method: str,
    preds: List[str],
    golds: List[str],
) -> Tuple[str, str, float]:
    acc = compute_accuracy(preds, golds)
    return dataset_name, method, acc