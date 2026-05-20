#!/usr/bin/env python
"""
Evaluate baseline GPT‑2, DPO‑fine‑tuned GPT‑2, and an “un‑aligned” version
where we scale a few toxic key vectors by 10× to reactivate toxicity.
"""
import argparse
import os
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from config import *
from utils import set_all_seeds, get_device, compute_toxicity_score

def get_toxicity_probs(model, tokenizer, probe_vector, prompts):
    probs = []
    for prompt in tqdm(prompts, desc="Generating"):
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            enc["input_ids"],
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Compute probe score on the new tokens
        new_ids = out[0][enc["input_ids"].shape[1]:]
        if new_ids.shape[0] == 0:
            probs.append(0.0)
            continue
        new_enc = tokenizer.decode(new_ids, skip_special_tokens=True)
        # For simplicity we compute probe on the whole new text
        new_outputs = model(**tokenizer(new_enc, return_tensors="pt").to(model.device))
        prob = compute_toxicity_score(new_outputs.logits, probe_vector)
        probs.append(prob)
    return probs

def perplexity(model, tokenizer, dataset):
    total_loss = 0.0
    count = 0
    for ex in tqdm(dataset, desc="PPL"):
        text = ex["text"]
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        lm_logits = model(**enc).logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = enc["input_ids"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss += loss.item()
        count += shift_labels.numel()
    return math.exp(total_loss / count)

def unalign(model, tokenizer, probe_vector, scale=10.0, topk=5):
    """
    Scale the key vectors of the top‑k toxic MLP layers by `scale`.
    This is a very rough approximation of the “un‑align” experiment.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if "mlp" in name and hasattr(module, "weight"):
                w = module.weight  # (hidden, hidden*2)
                hidden, hidden2 = w.shape
                W1 = w[:, :hidden]
                W2 = w[:, hidden:]
                # Cosine similarity of W2 columns to probe
                sims = torch.mm(W2.t(), probe_vector.unsqueeze(-1)).squeeze(-1)
                top_indices = torch.topk(sims, k=topk).indices
                # Scale corresponding columns of W2
                W2[:, top_indices] *= scale
                # Write back
                module.weight.data[:, hidden:] = torch.cat([W1, W2], dim=1)
    return model

def main(args):
    set_all_seeds()
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load probe vector
    probe_state = torch.load(args.probe_path, map_location=device)
    probe_linear = torch.nn.Linear(PROBE_HIDDEN_DIM, 1).to(device)
    probe_linear.load_state_dict(probe_state)
    probe_vector = probe_linear.weight.squeeze(0)

    # Load baseline model
    baseline = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    baseline.eval()

    # Load DPO model
    dpo = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    dpo.load_state_dict(torch.load(args.dpo_model, map_location=device))
    dpo.eval()

    # Load a small set of evaluation prompts
    eval_prompts = [ex["text"] for ex in load_dataset("realtoxicityprompts", split="test")[:REALTOXICITY_PROMPTS]]

    # Baseline toxicity
    baseline_probs = get_toxicity_probs(baseline, tokenizer, probe_vector, eval_prompts)
    print(f"[Eval] Baseline toxicity: {sum(baseline_probs)/len(baseline_probs):.3f}")

    # DPO toxicity
    dpo_probs = get_toxicity_probs(dpo, tokenizer, probe_vector, eval_prompts)
    print(f"[Eval] DPO toxicity: {sum(dpo_probs)/len(dpo_probs):.3f}")

    # Un‑align DPO by scaling toxic key vectors
    unaligned = unalign(dpo, tokenizer, probe_vector, scale=10.0, topk=5)
    unaligned_probs = get_toxicity_probs(unaligned, tokenizer, probe_vector, eval_prompts)
    print(f"[Eval] Un‑aligned toxicity: {sum(unaligned_probs)/len(unaligned_probs):.3f}")

    # Perplexity on a tiny Wikitext split
    train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:0.05]")
    baseline_ppl = perplexity(baseline, tokenizer, train_ds)
    dpo_ppl = perplexity(dpo, tokenizer, train_ds)
    print(f"[Eval] Baseline PPL: {baseline_ppl:.2f}")
    print(f"[Eval] DPO PPL: {dpo_ppl:.2f}")

    # Save metrics
    os.makedirs("output/metrics", exist_ok=True)
    torch.save({
        "baseline_toxicity": sum(baseline_probs)/len(baseline_probs),
        "dpo_toxicity": sum(dpo_probs)/len(dpo_probs),
        "unaligned_toxicity": sum(unaligned_probs)/len(unaligned_probs),
        "baseline_ppl": baseline_ppl,
        "dpo_ppl": dpo_ppl,
    }, "output/metrics/results.pt")

    print("[Eval] Results saved to output/metrics/results.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--dpo_model", required=True)
    args = parser.parse_args()
    main(args)