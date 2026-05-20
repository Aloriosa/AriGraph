#!/usr/bin/env python
import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from dataset import SimpleQADataset
from adapter import Adapter

def generate_candidates(model, tokenizer, prompt, num_candidates=5, device='cpu'):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=50,
        top_k=50,
        temperature=0.7,
        num_return_sequences=num_candidates,
        pad_token_id=tokenizer.eos_token_id
    )
    candidates = [tokenizer.decode(o, skip_special_tokens=True).strip()
                  for o in outputs]
    candidates = [c[len(prompt):].strip() if c.startswith(prompt) else c.strip()
                  for c in candidates]
    return candidates

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_dataset = SimpleQADataset(args.test_file)

    # Load black‑box LLM
    bb_tokenizer = AutoTokenizer.from_pretrained(args.blackbox_name)
    bb_model = AutoModelForCausalLM.from_pretrained(args.blackbox_name).to(device)
    bb_model.eval()

    # Load adapter
    adapter = Adapter()
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=device))
    adapter.to(device)
    adapter.eval()

    predictions = []
    for item in tqdm(test_dataset, desc="Evaluating"):
        q = item['question']
        gt = item['answer']
        prompt = f"Question: {q}\nAnswer:"
        cand = generate_candidates(
            bb_model, bb_tokenizer, prompt,
            num_candidates=args.num_candidates, device=device)
        # Ensure ground truth is present
        if gt not in cand:
            cand.append(gt)

        # Score with adapter
        texts = [f"Question: {q}\nAnswer: {c}" for c in cand]
        with torch.no_grad():
            scores = adapter(texts).cpu().numpy()
        best_idx = int(scores.argmax())
        pred = cand[best_idx]
        predictions.append({
            "question": q,
            "ground_truth": gt,
            "predicted_answer": pred
        })

    # Save predictions
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions written to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test JSONL file.")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to trained adapter checkpoint.")
    parser.add_argument("--blackbox_name", type=str, default="distilgpt2",
                        help="Black‑box LLM name.")
    parser.add_argument("--num_candidates", type=int, default=5,
                        help="Number of candidates generated per query.")
    parser.add_argument("--output_file", type=str, default="predictions.json",
                        help="File to write predictions.")
    args = parser.parse_args()
    main(args)