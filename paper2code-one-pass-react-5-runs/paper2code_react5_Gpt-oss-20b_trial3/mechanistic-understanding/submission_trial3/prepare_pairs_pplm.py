"""
Generate a pairwise dataset using a lightweight PPLM style
procedure: a linear toxicity classifier guides generation.
The dataset is written to pairs.jsonl.
"""

import json
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tqdm import tqdm
import numpy as np

# Hyper‑parameters
MAX_PROMPT_LEN = 64
MAX_CONT_LEN = 32
NUM_PAIRS = 2000

# Toxic phrases (used only for validation)
TOXIC_PHRASES = [
    "sh*t", "fuck", "damn", "hell", "bitch", "asshole",
    "idiot", "stupid", "jerk", "dumb"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & base model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.to(device)
model.eval()
# Enable hidden_states output
model.config.output_hidden_states = True

# Load probe as a linear toxicity classifier
probe_state = torch.load("probe.pt", map_location=device)
probe = nn.Linear(probe_state["weight"].shape[1], 2).to(device)
probe.weight.data = probe_state["weight"]
probe.bias.data = torch.zeros(2, device=device)
probe.eval()

# Load Wikitext‑2
dataset = load_dataset("wikitext",
                       "wikitext-2-raw-v1",
                       split="train[:5%]")  # small subset

pairs = []

def pplm_negative(prompt, alpha=0.5, steps=3):
    """
    Very light‑weight PPLM: at each decoding step we add
    alpha * gradient_of_toxicity to the logits.
    """
    # Tokenise prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated = input_ids
    for _ in range(steps):
        outputs = model(generated, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]          # (B, vocab)
        # Enable gradient tracking on logits
        logits = logits.clone().detach().requires_grad_(True)

        # Compute toxicity gradient
        hidden = outputs.hidden_states[-1][:, -1, :]  # (B, hidden)
        logits_tox = probe(hidden)                    # (B, 2)
        probs_tox = torch.softmax(logits_tox, dim=-1)[:, 1]
        # gradient of probs_tox w.r.t. logits
        grad = torch.autograd.grad(
            torch.sum(probs_tox),
            logits,
            retain_graph=False,
            create_graph=False
        )[0]                                          # (B, vocab)
        # Modify logits
        logits = logits + alpha * grad
        # Sample next token
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

    generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:],
                                      skip_special_tokens=True)
    return generated_text

for example in tqdm(dataset, desc="Generating pairs"):
    text = example["text"].strip()
    if not text:
        continue
    prompt = " ".join(text.split()[:MAX_PROMPT_LEN])
    if not prompt:
        continue

    # Positive: greedy decoding
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs,
                             max_new_tokens=MAX_CONT_LEN,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    pos_cont = tokenizer.decode(out[0][inputs["input_ids"].size(1):],
                                skip_special_tokens=True)

    # Negative: PPLM guided generation
    neg_cont = pplm_negative(prompt, alpha=0.5, steps=3)

    pairs.append({
        "prompt": prompt,
        "positive": pos_cont,
        "negative": neg_cont
    })

    if len(pairs) >= NUM_PAIRS:
        break

# Write dataset
with open("pairs.jsonl", "w", encoding="utf-8") as f:
    for item in pairs:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(pairs)} PPLM‑style pairwise examples to pairs.jsonl")