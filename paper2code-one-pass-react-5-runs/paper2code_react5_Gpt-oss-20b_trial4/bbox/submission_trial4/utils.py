"""
Utility helpers for evaluation and preprocessing.
"""

import re
import torch

def extract_numeric(answer):
    """
    Extract the last number from a string (int or float).
    Used for GSM8K exact‑match evaluation.
    """
    nums = re.findall(r"\d+(?:\.\d+)?", answer)
    if nums:
        return float(nums[-1])
    return None

def exact_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()

def gsm8k_accuracy(preds, golds):
    correct = 0
    for pred, gold in zip(preds, golds):
        pred_num = extract_numeric(pred)
        gold_num = extract_numeric(gold)
        if pred_num is not None and gold_num is not None and abs(pred_num - gold_num) < 1e-6:
            correct += 1
    return correct / len(preds)