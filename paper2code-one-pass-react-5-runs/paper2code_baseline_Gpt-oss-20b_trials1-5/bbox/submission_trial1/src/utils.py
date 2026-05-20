import os
import random
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
from tqdm import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class QADataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" not in line:
                    continue
                q, a = line.split("\t", 1)
                self.samples.append((q.strip(), a.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, a = self.samples[idx]
        return {"question": q, "answer": a}

def encode_concat(tokenizer, text1, text2, max_length=128):
    """Encode question+answer concatenated with a separator."""
    sep = "<|sep|>"
    combined = f"{text1} {sep} {text2}"
    enc = tokenizer(
        combined,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # For causal LM generation
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # For embeddings
    embedder = AutoModel.from_pretrained(model_name.replace("-gpt", "-distilbert"))
    return tokenizer, model, embedder

def get_candidate_generation(
    model,
    tokenizer,
    prompt: str,
    num_return_sequences: int = 5,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    # Generate multiple sequences
    outputs = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode
    candidates = []
    for seq in outputs:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        # Remove the prompt from the beginning
        candidate = text[len(prompt) :].strip()
        candidates.append(candidate)
    return candidates

def compute_embedding(embedder, input_ids, attention_mask):
    with torch.no_grad():
        outputs = embedder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooled output (CLS token)
        return outputs.last_hidden_state[:, 0, :]  # shape (batch, hidden)

def load_predictions(pred_path: str):
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(line.strip())
    return preds