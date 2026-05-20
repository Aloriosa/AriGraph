"""Data loading and preprocessing."""

import random
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Tuple

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_small_agnews_split() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load the AG News dataset and split it into:
        - D_PT: upstream pre‑training data (first 200 examples)
        - D_R_train: online error examples for training (first 20 examples from test)
        - D_R_test: online error examples for testing (next 20 examples from test)

    Returns:
        D_PT, D_R_train, D_R_test
    """
    dataset = load_dataset("ag_news", split="train[:250]")  # 250 training examples
    # Use the first 200 for upstream pre‑training
    D_PT_raw = dataset[:200]

    # For online errors, we take the test split and simulate mistakes
    test_split = load_dataset("ag_news", split="test[:40]")  # 40 test examples
    D_R_raw = test_split

    # Simulate errors: we flip the label of the first 20 examples
    D_R_train_raw = []
    D_R_test_raw = []
    for i, ex in enumerate(D_R_raw):
        ex_copy = ex.copy()
        if i < 20:
            # Simulate an error: flip the label
            ex_copy["label"] = (ex_copy["label"] + 1) % 4
            D_R_train_raw.append(ex_copy)
        else:
            D_R_test_raw.append(ex_copy)

    # Convert to simple dict lists
    def to_dict_list(ds):
        return [{"text": d["text"], "label": d["label"]} for d in ds]

    D_PT = to_dict_list(D_PT_raw)
    D_R_train = to_dict_list(D_R_train_raw)
    D_R_test = to_dict_list(D_R_test_raw)

    return D_PT, D_R_train, D_R_test