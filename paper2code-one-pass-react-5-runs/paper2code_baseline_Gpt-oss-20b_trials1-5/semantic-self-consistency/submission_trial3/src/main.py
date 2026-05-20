import os
import sys
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Tuple, Dict

from generate import ChainOfThoughtGenerator
from semantic import extract_answer, embed_texts, semantic_consensus_weighting


# Configuration
NUM_SAMPLES = 20          # Process only the first 20 examples for speed
NUM_GENERATIONS = 10      # Number of CoT samples per question
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

# Dataset mapping: name -> (dataset_id, split, question_field, answer_field)
DATASETS = {
    "AQuA-RAT": ("tavily/aqa-rat", "test", "question", "answer"),
    "SVAMP": ("allenai/svamp", "test", "question", "answer"),
    "StrategyQA": ("strategyqa/strategyqa", "test", "question", "answer"),
}


def evaluate_dataset(
    dataset_name: str,
    dataset_id: str,
    split: str,
    question_field: str,
    answer_field: str,
) -> Tuple[float, int]:
    """
    Run the pipeline on a single dataset and return accuracy and number of examples.
    """
    print(f"\n=== {dataset_name} ===")
    ds = load_dataset(dataset_id, split=split)
    ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
    generator = ChainOfThoughtGenerator(device=DEVICE)

    correct = 0
    total = 0

    for row in tqdm(ds, desc="Processing examples"):
        question = row[question_field]
        gold_answer = row[answer_field].strip()

        # Generate CoT samples
        completions = generator.generate(question, n=NUM_GENERATIONS)

        # Extract answers and embeddings
        answers = [extract_answer(c) for c in completions]
        if not all(answers):
            # Skip examples where answer extraction failed
            continue

        embeddings = embed_texts(completions)

        # Baseline self‑consistency (majority vote)
        pred_sc = max(set(answers), key=answers.count)

        # Semantic consensus weighting
        pred_scw = semantic_consensus_weighting(answers, embeddings)

        # For illustration we compare only SCW
        if pred_scw == gold_answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, total


def main():
    results = []
    for name, (ds_id, split, qf, af) in DATASETS.items():
        acc, n = evaluate_dataset(name, ds_id, split, qf, af)
        results.append((name, acc, n))

    # Write results to CSV
    out_path = "results.csv"
    with open(out_path, "w") as f:
        f.write("Dataset,Accuracy,Examples\n")
        for name, acc, n in results:
            f.write(f"{name},{acc:.4f},{n}\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()