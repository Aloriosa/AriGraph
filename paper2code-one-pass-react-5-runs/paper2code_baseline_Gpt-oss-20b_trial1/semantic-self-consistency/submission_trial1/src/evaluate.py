import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline, set_seed

from .utils import (
    extract_answer_from_text,
    embed_texts,
    centroid_proximity_weighting,
    semantic_consensus_weighting,
    aggregate_by_weight,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NUM_SAMPLES = 5           # number of CoT rationales per question
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200
SEED = 42

# --------------------------------------------------------------------------- #
# 1. Load toy dataset
# --------------------------------------------------------------------------- #
DATA_PATH = Path("data/dataset.jsonl")

if not DATA_PATH.exists():
    # Create a small toy dataset if not present
    toy_data = [
        # AQuA‑RAT style
        {
            "question": "If you have 5 apples and you buy 3 more, how many apples do you have?",
            "answer": "8",
        },
        {
            "question": "A car travels 60 km in 2 hours. What is its average speed?",
            "answer": "30",
        },
        {
            "question": "If 10 students score an average of 80, what is the total score?",
            "answer": "800",
        },
        # SVAMP style
        {
            "question": "Solve for x: 2x + 5 = 13",
            "answer": "4",
        },
        {
            "question": "If y - 3 = 10, what is y?",
            "answer": "13",
        },
        {
            "question": "What is 7 * 6?",
            "answer": "42",
        },
        # StrategyQA style
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
        },
        {
            "question": "Who wrote '1984'?",
            "answer": "George Orwell",
        },
        {
            "question": "What is the chemical symbol for water?",
            "answer": "H2O",
        },
    ]
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for ex in toy_data:
            f.write(json.dumps(ex) + "\n")

# Load dataset
dataset = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

# --------------------------------------------------------------------------- #
# 2. Setup generation pipeline
# --------------------------------------------------------------------------- #
device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "text-generation",
    model="gpt2",
    tokenizer="gpt2",
    device=device,
    do_sample=True,
    temperature=TEMPERATURE,
    num_return_sequences=NUM_SAMPLES,
    max_new_tokens=MAX_NEW_TOKENS,
)

# --------------------------------------------------------------------------- #
# 3. Setup embedding model
# --------------------------------------------------------------------------- #
embedder = __import__("sentence_transformers").SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------------------------------- #
# 4. Evaluation loop
# --------------------------------------------------------------------------- #
def evaluate(dataset):
    random.seed(SEED)
    baseline_correct = 0
    cpw_correct = 0
    scw_correct = 0
    total = len(dataset)

    for idx, ex in enumerate(dataset):
        question = ex["question"]
        gold = ex["answer"].strip().lower()

        # Generate rationales
        gen_outputs = generator(
            question,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=NUM_SAMPLES,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        rationales = [out["generated_text"] for out in gen_outputs]
        # Extract answers
        answers = [extract_answer_from_text(r).lower() for r in rationales]

        # Baseline majority vote
        most_common = Counter(answers).most_common(1)[0][0]
        if most_common == gold:
            baseline_correct += 1

        # Embed rationales
        embeddings = embed_texts(rationales, embedder)

        # CPW
        cpw_weights = centroid_proximity_weighting(embeddings)
        cpw_ans, _ = aggregate_by_weight(answers, cpw_weights)
        if cpw_ans == gold:
            cpw_correct += 1

        # SCW
        scw_weights = semantic_consensus_weighting(embeddings)
        scw_ans, _ = aggregate_by_weight(answers, scw_weights)
        if scw_ans == gold:
            scw_correct += 1

    results = {
        "baseline_accuracy": baseline_correct / total,
        "cpw_accuracy": cpw_correct / total,
        "scw_accuracy": scw_correct / total,
        "total_examples": total,
    }
    return results

# --------------------------------------------------------------------------- #
# 5. Run and save results
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"Evaluating {len(dataset)} examples...")
    res = evaluate(dataset)
    print(f"Baseline accuracy: {res['baseline_accuracy']:.3f}")
    print(f"CPW accuracy: {res['cpw_accuracy']:.3f}")
    print(f"SCW accuracy: {res['scw_accuracy']:.3f}")

    out_path = Path("results/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Results written to {out_path}")