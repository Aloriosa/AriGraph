import os
import random
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# -------------------------------------------------------------
# General utilities
# -------------------------------------------------------------
SEED = 42
set_seed(SEED)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Model helpers
# -------------------------------------------------------------
def load_model_and_tokenizer(name, device="cuda"):
    """Download and cache a model + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer

# -------------------------------------------------------------
# Dataset helpers
# -------------------------------------------------------------
def get_wikitext2_prompts(limit=2000):
    """Return a list of (prompt, continuation) pairs from WikiText‑2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    prompts = []
    for i in range(limit):
        # split the text into sentences and pick a random one as prompt
        text = dataset[i]["text"].strip()
        if len(text) == 0:
            continue
        prompts.append(text)
    return prompts

def get_realtoxicity_prompts(limit=2000):
    """Return a list of real toxicity prompts."""
    dataset = load_dataset("realtoxicityprompts", "challenge", split="train")
    prompts = []
    for i in range(limit):
        prompts.append(dataset[i]["prompt"])
    return prompts

# -------------------------------------------------------------
# Toxicity classifier
# -------------------------------------------------------------
def load_toxicity_classifier():
    clf = AutoModelForSequenceClassification.from_pretrained(
        config.TOXICITY_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.TOXICITY_MODEL)
    return clf, tokenizer

def predict_toxicity(clf, tokenizer, texts, batch_size=8):
    """Return probability of the toxic class for each input."""
    clf.eval()
    probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying toxicity"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(next(clf.parameters()).device)
        with torch.no_grad():
            out = clf(**enc)
        # Assume label 1 = toxic
        probs.append(torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    return probs

# -------------------------------------------------------------
# Helper for generating continuations
# -------------------------------------------------------------
def generate_continuations(model, tokenizer, prompt, num=5, length=20):
    """Return a list of generated continuations for a prompt."""
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    input_ids = enc["input_ids"]
    continuations = []
    for _ in range(num):
        out = model.generate(
            input_ids,
            max_new_tokens=length,
            do_sample=True,
            top_k=TOP_K,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][len(input_ids[0]):], skip_special_tokens=True)
        continuations.append(text.strip())
    return continuations

# -------------------------------------------------------------
# Log‑probability helper for DPO
# -------------------------------------------------------------
def seq_log_prob(model, tokenizer, prompt, continuation):
    """Compute log probability of a full sequence = prompt + continuation."""
    full = prompt + continuation
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LENGTH).to(next(model.parameters()).device)
    input_ids = enc["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    # outputs.loss is cross‑entropy averaged over tokens
    # total log prob = -loss * seq_len
    seq_len = input_ids.size(1)
    log_prob = -outputs.loss.item() * seq_len
    return log_prob