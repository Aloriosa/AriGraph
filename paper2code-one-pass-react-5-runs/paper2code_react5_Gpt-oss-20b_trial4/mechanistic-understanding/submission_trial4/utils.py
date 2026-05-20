import os
import random
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torchmetrics
import math
from tqdm.auto import tqdm

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)

def load_model(model_name, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model

def load_encoder(model_name, device="cuda"):
    return AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)

# Simple toxicity classifier using a pre‑trained HuggingFace model
def load_toxicity_classifier(device="cuda"):
    clf = transformers.pipeline(
        "text-classification",
        model="unitary/toxicbert",
        device=0 if device=="cuda" else -1,
        return_all_scores=True,
    )
    return clf

def compute_toxicity_score(text, clf):
    # returns average toxicity probability across labels
    scores = clf(text)[0]
    # labels: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    tox_idx = [i for i, l in enumerate(clf.classes_) if "toxic" in l.lower()]
    probs = [s["score"] for s in scores if s["label"].lower() in {"toxic","severe_toxic","obscene","threat","insult","identity_hate"}]
    return max(probs) if probs else 0.0

def collate_fn(batch):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
        "labels": torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100),
    }

def get_wikitext_prompts(num=1000, split="train"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    prompts = []
    for i in range(min(num, len(ds))):
        text = ds[i]["text"].strip()
        if text:
            prompts.append(text)
    return prompts

def get_realtoxicity_prompts(num=200):
    ds = load_dataset("realtoxicityprompts", split="train")
    prompts = []
    for i in range(min(num, len(ds))):
        prompts.append(ds[i]["prompt"])
    return prompts

def compute_perplexity(model, tokenizer, dataset, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=4, shuffle=False):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    ppl = math.exp(total_loss / total_tokens)
    return ppl

def compute_f1(model, tokenizer, prompts, references, device="cuda"):
    model.eval()
    preds = []
    for prompt in tqdm(prompts, desc="F1 generation"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(text[len(prompt):].strip())
    # compute token-level precision/recall
    pred_tokens = [p.split() for p in preds]
    ref_tokens = [r.split() for r in references]
    precision = 0.0
    recall = 0.0
    for p, r in zip(pred_tokens, ref_tokens):
        p_set = set(p)
        r_set = set(r)
        precision += len(p_set & r_set) / (len(p_set) + 1e-12)
        recall += len(p_set & r_set) / (len(r_set) + 1e-12)
    precision /= len(pred_tokens)
    recall /= len(pred_tokens)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return f1