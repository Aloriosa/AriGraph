# src/data.py
import random
from datasets import load_dataset
from transformers import AutoTokenizer

def load_sst2(split="train", max_samples=None):
    """
    Load the SST-2 (GLUE) dataset.
    Returns a list of dicts with keys: 'sentence', 'label'.
    """
    ds = load_dataset("glue", "sst2", split=split)
    if max_samples is not None:
        ds = ds.select(range(max_samples))
    return ds

def split_dataset(dataset, train_frac=0.5, seed=42):
    """
    Split a dataset into train and test lists.
    """
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_point = int(len(indices) * train_frac)
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    train = [dataset[i] for i in train_idx]
    test = [dataset[i] for i in test_idx]
    return train, test

def prepare_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # BART uses <s> and </s> tokens; set accordingly
    return tokenizer

def encode_batch(tokenizer, batch, max_length=128):
    """
    Encode a batch of examples using the tokenizer.
    """
    texts = [ex['sentence'] for ex in batch]
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )