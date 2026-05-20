#!/usr/bin/env python
"""
Evaluate a model on toxicity and perplexity.  The script accepts
`--aligned` or `--unaligned` flags to switch between the DPO model
and the un‑aligned version.  Results are printed and saved to
results/evaluation.json.
"""

import json
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from src import config, utils

parser = argparse.ArgumentParser()
parser.add_argument("--aligned", action="store_true", help="Evaluate the aligned DPO model")
parser.add_argument("--unaligned", action="store_true", help="Evaluate the un‑aligned model")
args = parser.parse_args()

if args.unaligned:
    model_name = "gpt2-medium"
else:
    model_name = "results/dpo_gpt2"

print(f"Loading model {model_name} for evaluation...")
model, tokenizer = utils.load_model_and_tokenizer(model_name)

# Load toxicity classifier
toxic_clf, toxic_tok = utils.load_toxicity_classifier()

# -------------------------------------------------------------
# 1. Toxicity evaluation on real‑toxicity prompts
# -------------------------------------------------------------
print("Evaluating toxicity on RealToxicityPrompts...")
dataset = load_dataset("realtoxicityprompts", "challenge", split="train")
prompts = [row["prompt"] for row in dataset]
# Generate continuations with the model
continuations = []
for prompt in tqdm(prompts, desc="Generating continuations"):
    cont = utils.generate_continuations(
        model, tokenizer, prompt, num=1, length=config.CONT_LENGTH
    )[0]
    continuations.append(cont)

# Compute toxicity scores
tox_probs = utils.predict_toxicity(toxic_clf, toxic_tok, continuations)
avg_toxic = float(np.mean(tox_probs))
print(f"Avg toxicity (probability of toxic label): {avg_toxic:.4f}")

# -------------------------------------------------------------
# 2. Perplexity on Wikitext‑2 test split
# -------------------------------------------------------------
print("Computing perplexity on Wikitext‑2 test split...")
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
perps = []
for row in tqdm(test_dataset, desc="Evaluating perplexity"):
    text = row["text"].strip()
    if len(text) == 0:
        continue
    enc = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(**enc, labels=enc["input_ids"])
    perps.append(torch.exp(outputs.loss).item())
avg_ppl = float(np.mean(perps))
print(f"Perplexity on test set: {avg_ppl:.2f}")

# -------------------------------------------------------------
# 3. Save results
# -------------------------------------------------------------
results = {
    "model": model_name,
    "avg_toxicity": avg_toxic,
    "perplexity": avg_ppl,
}
utils.ensure_dir("results")
with open("results/evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
print("Evaluation results saved to results/evaluation.json")