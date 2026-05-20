#!/usr/bin/env python
import json
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from src.toxicity_classifier import ToxicityClassifier

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
MODEL_DIR = "./dpo_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 50

# --------------------------------------------------------------------------- #
# Load model
# --------------------------------------------------------------------------- #
print("Loading fine-tuned model ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, device_map={"": DEVICE}
).to(DEVICE)

# --------------------------------------------------------------------------- #
# Toxicity classifier
# --------------------------------------------------------------------------- #
print("Loading toxicity classifier ...")
tox_classifier = ToxicityClassifier(device=DEVICE)

# --------------------------------------------------------------------------- #
# Helper: generate continuation
# --------------------------------------------------------------------------- #
def generate_continuation(prompt, max_new_tokens=50, do_sample=False, temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    continuation_ids = generated_ids[0, input_ids.size(1) :]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True)

# --------------------------------------------------------------------------- #
# 1. Toxicity evaluation on RealToxicityPrompts
# --------------------------------------------------------------------------- #
print("Evaluating toxicity on RealToxicityPrompts ...")
rt_dataset = load_dataset("realtoxicityprompts", split="challenge")

toxicity_scores = []
for example in tqdm(rt_dataset, desc="Prompts"):
    prompt = example["prompt"]
    generated = generate_continuation(prompt, max_new_tokens=MAX_LENGTH, do_sample=True)
    score = tox_classifier.predict(generated)
    toxicity_scores.append(score)

avg_toxicity = float(np.mean(toxicity_scores))
print(f"Average toxicity score: {avg_toxicity:.4f}")

# --------------------------------------------------------------------------- #
# 2. Perplexity on Wikitext-2 validation
# --------------------------------------------------------------------------- #
print("Computing perplexity on Wikitext-2 validation ...")
wikitext_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

def compute_perplexity(model, tokenizer, dataset, max_length=512):
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0
    for example in tqdm(dataset, desc="Computing PPL"):
        text = example["text"]
        if not text.strip():
            continue
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        n_tokens = inputs["input_ids"].size(1)
        total_log_likelihood += loss.item() * n_tokens
        total_tokens += n_tokens
    ppl = np.exp(total_log_likelihood / total_tokens)
    return float(ppl)

ppl = compute_perplexity(model, tokenizer, wikitext_val)
print(f"Perplexity on Wikitext-2: {ppl:.2f}")

# --------------------------------------------------------------------------- #
# 3. Save results
# --------------------------------------------------------------------------- #
results = {
    "toxicity": avg_toxicity,
    "perplexity": ppl,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results written to results.json")