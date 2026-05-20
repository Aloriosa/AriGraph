#!/usr/bin/env python
"""
Fine‑tune GPT‑2 medium with Direct Preference Optimization (DPO).
We build a small pairwise dataset from Wikitext‑2 prompts,
produce a positive continuation with greedy decoding,
and a “negative” continuation by sampling and picking the most toxic
according to the probe classifier.
"""
import argparse
import os
import torch
import math
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from config import *
from utils import set_all_seeds, get_device, compute_toxicity_score

class PairwiseDataset(Dataset):
    def __init__(self, prompts, tokenizer, probe_vector, device):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.probe_vector = probe_vector
        self.device = device

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Positive continuation: greedy decoding
        with torch.no_grad():
            pos_ids = self.tokenizer.encode(prompt)
            pos_ids = torch.tensor(pos_ids, device=self.device).unsqueeze(0)
            pos_out = self.model.generate(pos_ids, max_new_tokens=20, do_sample=False)
            pos_text = self.tokenizer.decode(pos_out[0], skip_special_tokens=True)

        # Negative continuation: sample 5 continuations and pick most toxic
        neg_texts = []
        for _ in range(5):
            sample_ids = self.tokenizer.encode(prompt)
            sample_ids = torch.tensor(sample_ids, device=self.device).unsqueeze(0)
            sample_out = self.model.generate(
                sample_ids,
                max_new_tokens=20,
                do_sample=True,
                top_k=50,
                temperature=1.0,
            )
            neg_texts.append(self.tokenizer.decode(sample_out[0], skip_special_tokens=True))

        # Evaluate toxicity with probe
        neg_tox = [compute_toxicity_score(
            self.model(**self.tokenizer(neg, return_tensors="pt").to(self.device)).logits,
            self.probe_vector
        ) for neg in neg_texts]
        neg_text = neg_texts[torch.argmax(torch.tensor(neg_tox))]

        return {
            "prompt": prompt,
            "pos": pos_text,
            "neg": neg_text,
        }

def dpo_loss(pos_logp, neg_logp, beta=DPO_BETA):
    # DPO loss: -log(sigmoid(beta*(logP - logN)))
    diff = beta * (pos_logp - neg_logp)
    loss = -torch.nn.functional.logsigmoid(diff).mean()
    return loss

def main(args):
    set_all_seeds()
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load reference (frozen) model
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref_model.eval()

    # Load probe vector
    probe_state = torch.load(args.probe_path, map_location=device)
    probe_linear = torch.nn.Linear(PROBE_HIDDEN_DIM, 1).to(device)
    probe_linear.load_state_dict(probe_state)
    probe_vector = probe_linear.weight.squeeze(0)

    # Build pairwise dataset
    raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=WIKITEXT_SPLIT[0])
    prompts = [ex["text"] for ex in raw_ds if len(ex["text"].strip()) > 0][:200]  # small set
    pair_ds = PairwiseDataset(prompts, tokenizer, probe_vector, device)
    loader = DataLoader(pair_ds, batch_size=1, shuffle=True)

    # Fine‑tune model
    model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DPO_LR)

    best_loss = math.inf
    for epoch in range(DPO_EPOCHS):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"DPO Epoch {epoch+1}"):
            prompt = batch["prompt"][0]
            pos_ids = tokenizer.encode(prompt + batch["pos"][0], return_tensors="pt").to(device)
            neg_ids = tokenizer.encode(prompt + batch["neg"][0], return_tensors="pt").to(device)

            # Compute log probabilities
            with torch.no_grad():
                ref_pos = ref_model(pos_ids)
                ref_neg = ref_model(neg_ids)

            # Forward on current model
            pos_out = model(pos_ids)
            neg_out = model(neg_ids)

            # Log probabilities of the *new* tokens only
            pos_logp = pos_out.logits[:, -20:, :].log_softmax(-1)
            neg_logp = neg_out.logits[:, -20:, :].log_softmax(-1)

            # Sum logp over the new tokens
            pos_logp = pos_logp.sum(dim=(1,2))
            neg_logp = neg_logp.sum(dim=(1,2))

            loss = dpo_loss(pos_logp, neg_logp)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DPO_GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[DPO] Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

        # Simple early stopping on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("[DPO] New best model saved.")

    print(f"[DPO] Fine‑tuning finished. Best model at {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--output_dir", default="output/dpo")
    args = parser.parse_args()
    main(args)