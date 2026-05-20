import os
import math
import json
import torch
import numpy as np
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
from sklearn.metrics import f1_score

def compute_toxicity_scores(texts, device):
    """Use the unitary/toxic‑bert classifier to get toxicity probabilities."""
    clf_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        "unitary/toxic-bert"
    ).to(device)
    clf_model.eval()

    probs = []
    for text in texts:
        inputs = clf_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            logits = clf_model(**inputs).logits
        probs.append(torch.softmax(logits, dim=-1)[0, 1].item())  # toxic prob
    return probs

def compute_perplexity(model, tokenizer, dataset, device):
    total_logp = 0.0
    total_tokens = 0
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        seq_len = input_ids.size(1)
        total_logp += loss.item() * seq_len
        total_tokens += seq_len
    ppl = math.exp(total_logp / total_tokens)
    return ppl

def compute_f1(model, tokenizer, dataset, device):
    """
    Very naive token‑level F1: compare generated tokens with the original
    tokens after the prompt.  This is only a placeholder for the paper.
    """
    pred_tokens = []
    ref_tokens = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred_tokens.extend(pred.split())
        ref_tokens.extend(text.split())

    # Simple token overlap F1
    f1 = f1_score(ref_tokens, pred_tokens, average="macro")
    return f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fine‑tuned DPO model
    model_dir = "results/dpo_model"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()

    # Real toxicity prompts (challenge set)
    rt_ds = load_dataset("realtoxicityprompts", split="challenge")
    prompts = [row["prompt"] for row in rt_ds]
    outputs = []
    for prompt in prompts[:50]:  # limit to first 50 for speed
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))

    # Toxicity
    tox_scores = compute_toxicity_scores(outputs, device)
    avg_tox = np.mean(tox_scores)
    print(f"Average toxicity score: {avg_tox:.4f}")

    # Perplexity on WikiText‑2 (small slice)
    wikidet = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:10]")
    ppl = compute_perplexity(model, tokenizer, wikidet, device)
    print(f"Perplexity on WikiText‑2: {ppl:.2f}")

    # F1 (placeholder)
    f1 = compute_f1(model, tokenizer, wikidet, device)
    print(f"Token‑level F1 (placeholder): {f1:.4f}")

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(
            {"toxicity": float(avg_tox), "ppl": float(ppl), "f1": float(f1)},
            f,
            indent=2,
        )
    print("Metrics saved to results/metrics.json")

if __name__ == "__main__":
    main()