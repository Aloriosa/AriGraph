#!/usr/bin/env python
"""
Full experiment pipeline.

Usage:
    python src/main.py --dataset all --model_name meta-llama/Llama-2-7b-hf
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils import (
    parse_answer,
    load_dataset_by_name,
    compute_cpw_weights,
    compute_scw_weights,
)

# 8‑shot prompt from Appendix K (math datasets)
SHOTS = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted $21-15=6$ trees. The answer is 6 .

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are 3 cars in the parking lot already. 2 more arrive. Now there are $3+2=5$ cars. The answer is 5 .

Q: Leah had 32 chocolates and her sister had 42 . If they ate 35 , how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42 . That means there were originally $32+42=74$ chocolates. 35 have been eaten. So in total they still have $74-35=39$ chocolates. The answer is 39 .

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been $20-12=8$ lollipops. The answer is 8 .

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has $5+2=7$ toys. Then he got 2 more from dad, so in total he has $7+2=9$ toys. The answer is 9 .
"""

# For StrategyQA we use an 8‑shot prompt that contains multi‑choice examples.
STRATEGYQA_SHOTS = """
Q: Who was the first person to step on the moon?
A: The first person to step on the moon was Neil Armstrong. Answer: A

Q: What is the capital of France?
A: The capital of France is Paris. Answer: A

Q: Which of the following is a prime number? (a) 4 (b) 6 (c) 9 (d) 11
A: The number 11 is prime. Answer: (d)

Q: Who wrote the play 'Hamlet'?
A: William Shakespeare wrote 'Hamlet'. Answer: A

Q: In which year did the Titanic sink?
A: The Titanic sank in 1912. Answer: A
"""

# Maximum new tokens for each dataset
MAX_TOKENS = {
    "aquarat": 400,
    "svamp": 250,
    "strategyqa": 450,
}

def build_prompt(question: str, dataset: str) -> str:
    """
    Construct the full prompt sent to the LLM.
    """
    if dataset == "strategyqa":
        shots = STRATEGYQA_SHOTS
    else:
        shots = SHOTS
    prompt = f"{shots}\nQ: {question}\nA:"
    return prompt


def generate_rationales(
    generator, prompt: str, n_samples: int, max_new_tokens: int, temperature: float, top_p: float
) -> List[str]:
    """
    Generate n_samples rationales for a single question.
    """
    # The pipeline expects a list of prompts; we repeat prompt n_samples times
    inputs = [prompt] * n_samples
    outputs = generator(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,  # we already duplicate inputs
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    # outputs is a list of dicts with "generated_text"
    rationales = [o["generated_text"] for o in outputs]
    return rationales


def extract_answer_from_rationales(rationales: List[str]) -> List[str]:
    """
    Apply parse_answer to each rationale.
    """
    return [parse_answer(r) for r in rationales]


def majority_vote(answers: List[str]) -> str:
    """
    Return the most common answer (ties broken arbitrarily).
    """
    counter = Counter(answers)
    most_common, _ = counter.most_common(1)[0]
    return most_common


def compute_accuracy(preds: List[str], golds: List[str]) -> float:
    """
    Compute accuracy as the fraction of exact matches.
    """
    correct = sum(p.strip().lower() == g.strip().lower() for p, g in zip(preds, golds))
    return correct / len(golds)


def run_experiment(
    dataset_name: str,
    model_name: str,
    n_samples: int,
    temperature: float,
    device: str,
    top_p: float,
) -> Dict[str, float]:
    """
    Run the full pipeline for a single dataset.
    Returns a dict with baseline, cpw, scw accuracies.
    """
    print(f"\n=== Running {dataset_name} ===")

    ds = load_dataset_by_name(dataset_name)
    questions = ds["question"]
    golds = ds["answer"]

    # Load the generator
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,  # assume single GPU
    )

    # Featurizer
    featurizer = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    all_baseline = []
    all_cpw = []
    all_scw = []

    for idx, (q, gold) in enumerate(zip(questions, golds)):
        prompt = build_prompt(q, dataset_name.lower())
        rationales = generate_rationales(
            generator,
            prompt,
            n_samples,
            max_new_tokens=MAX_TOKENS[dataset_name.lower()],
            temperature=temperature,
            top_p=top_p,
        )
        answers = extract_answer_from_rationales(rationales)

        # Baseline self‑consistency (majority vote)
        pred_baseline = majority_vote(answers)

        # CPW
        embeddings = featurizer.encode(rationales, convert_to_numpy=True, normalize=True)
        cpw_weights = compute_cpw_weights(embeddings)
        # Sum weights per answer
        answer_to_weight = defaultdict(float)
        for ans, w in zip(answers, cpw_weights):
            answer_to_weight[ans] += w
        pred_cpw = max(answer_to_weight.items(), key=lambda x: x[1])[0]

        # SCW
        scw_scores = compute_scw_weights(embeddings)
        answer_to_score = defaultdict(float)
        for ans, s in zip(answers, scw_scores):
            answer_to_score[ans] += s
        pred_scw = max(answer_to_score.items(), key=lambda x: x[1])[0]

        all_baseline.append(pred_baseline)
        all_cpw.append(pred_cpw)
        all_scw.append(pred_scw)

        if idx % 10 == 0:
            print(f"Processed {idx+1}/{len(questions)} questions", end="\r")

    # Compute accuracies
    acc_baseline = compute_accuracy(all_baseline, golds)
    acc_cpw = compute_accuracy(all_cpw, golds)
    acc_scw = compute_accuracy(all_scw, golds)

    print(f"\nResults for {dataset_name}:")
    print(f"  Baseline self‑consistency  : {acc_baseline*100:.2f}%")
    print(f"  CPW (Centroid Proximity)   : {acc_cpw*100:.2f}%")
    print(f"  SCW (Semantic Consensus)   : {acc_scw*100:.2f}%")

    return {
        "dataset": dataset_name,
        "baseline": acc_baseline,
        "cpw": acc_cpw,
        "scw": acc_scw,
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic Self‑Consistency Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to run: aquarat, svamp, strategyqa, or all",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model name (must be compatible with transformers)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to generate per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top‑p sampling value",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        dataset_list = ["AQuA-RAT", "SVAMP", "StrategyQA"]
    else:
        dataset_list = [args.dataset]

    all_results = []
    for ds_name in dataset_list:
        res = run_experiment(
            ds_name,
            args.model_name,
            args.n_samples,
            args.temperature,
            device="cuda",
            top_p=args.top_p,
        )
        all_results.append(res)

    # Save to CSV
    import csv

    out_file = "results.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "baseline", "cpw", "scw"],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "baseline": f"{r['baseline']*100:.2f}",
                    "cpw": f"{r['cpw']*100:.2f}",
                    "scw": f"{r['scw']*100:.2f}",
                }
            )

    print(f"\nAll results written to {out_file}")


if __name__ == "__main__":
    main()