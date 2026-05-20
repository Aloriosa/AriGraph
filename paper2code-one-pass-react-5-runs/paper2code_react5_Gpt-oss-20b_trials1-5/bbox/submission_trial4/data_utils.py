"""
Utilities for loading and splitting the datasets.
"""

from datasets import load_dataset
import random
from sklearn.model_selection import train_test_split

def get_dataset(name):
    """
    Load a dataset from the 🤗 `datasets` hub and split it into
    train/dev/test as used in the paper.
    """
    if name.lower() == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        train = ds["train"]
        test = ds["test"]
    elif name.lower() == "strategyqa":
        ds = load_dataset("strategyqa")
        train = ds["train"]
        test = ds["validation"]
    elif name.lower() == "truthfulqa":
        ds = load_dataset("truthfulqa")
        train = ds["train"]
        test = ds["validation"]
    elif name.lower() == "scienceqa":
        ds = load_dataset("scienceqa")
        train = ds["train"]
        test = ds["validation"]
    else:
        raise ValueError(f"Unknown dataset {name}")

    # Shuffle and split train into train/dev (80/20)
    train = train.shuffle(seed=42)
    train_split = train[: int(0.8 * len(train))]
    dev_split = train[int(0.8 * len(train)) :]

    return train_split, dev_split, test