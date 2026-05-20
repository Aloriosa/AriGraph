"""
Inference script that demonstrates plug‑and‑play usage of a trained
BBox‑Adapter with a black‑box LLM.
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.adapter import EnergyAdapter
from src.utils import (
    sample_candidates,
    compute_logprob_llm,
    load_strategyqa,
    compute_accuracy,
    get_generation_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description="BBox‑Adapter inference")
    parser.add_argument("--dataset", type=str, default="strategyqa")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--llm_name", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--candidate_num", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--adapter_model", type=str, default="microsoft/deberta-v3-base")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_llm = 0 if torch.cuda.is_available() else -1

    # ---------------------------------------
    # 1. Load adapter
    # ---------------------------------------
    adapter_tokenizer = AutoTokenizer.from_pretrained(args.adapter_model)
    adapter_tokenizer.pad_token = adapter_tokenizer.eos_token
    adapter = EnergyAdapter(model_name=args.adapter_model).to(device)
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=device))
    adapter.eval()

    # ---------------------------------------
    # 2. LLM model for probabilities
    # ---------------------------------------
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_name).to(device)
    llm_model.eval()

    # LLM generation pipeline (used for candidate sampling)
    llm_gen = get_generation_pipeline(
        model_name=args.llm_name,
        device=device_llm,
        max_length=args.max_seq_len,
        temperature=1.0,
    )

    # ---------------------------------------
    # 3. Load test set
    # ---------------------------------------
    if args.dataset == "strategyqa":
        test_ds = load_strategyqa("validation")
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    preds = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            q = test_ds[i]["prompt"]
            # Generate candidates
            cand_list = sample_candidates(llm_gen, q, args.candidate_num, device=device_llm)

            cand_enc = adapter_tokenizer(
                cand_list,
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            ).to(device)

            energies = adapter(
                cand_enc["input_ids"], cand_enc["attention_mask"]
            )  # (num_cand,)

            # Compute log probabilities for each candidate
            logprobs = []
            for cand in cand_list:
                lp = compute_logprob_llm(
                    llm_model, llm_tokenizer, q, cand, device=torch.device(device)
                )
                logprobs.append(lp)
            logprobs = torch.tensor(logprobs, device=device)

            # Combine adapter energy with LLM log probability
            combined = energies + logprobs
            best_idx = combined.argmax().item()
            preds.append(cand_list[best_idx])

    # Convert to binary labels
    preds_bin = ["Yes" if "yes" in p.lower() else "No" for p in preds]
    labels = [l for l in test_ds["label"]]
    acc = compute_accuracy(preds_bin, labels)
    print(f"Inference accuracy: {acc*100:.1f}%")

    # Save results
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "eval.log"), "w") as f:
        f.write(f"Inference accuracy: {acc*100:.1f}%\n")
        f.write("\nPredictions:\n")
        for q, p, l in zip(test_ds["prompt"], preds, labels):
            f.write(f"Q: {q}\nPred: {p}\nTrue: {l}\n\n")


if __name__ == "__main__":
    main()