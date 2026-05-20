import json
import os
import argparse
import pandas as pd

from generate import generate_responses
from embeddings import embed_texts
from weighting import compute_cpw, compute_scw
from evaluate import (majority_vote, weighted_majority_vote,
                      extract_answer, compute_accuracy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sample_dataset.jsonl",
                        help="Path to JSONL dataset")
    parser.add_argument("--output", default="results/run_results.csv",
                        help="CSV file to store results")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace model for generation")
    args = parser.parse_args()

    # Load config
    with open("config.json") as f:
        cfg = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load dataset
    data = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    results = []

    for idx, item in enumerate(data):
        question = item["question"]
        true_answer = item.get("answer")

        # 1. Generate multiple rationales
        samples = generate_responses(
            question,
            args.model,
            cfg["temperature"],
            cfg["num_samples"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
        )

        # 2. Extract final answers
        final_answers = [extract_answer(s) for s in samples]

        # 3. Embed reasoning paths
        embeddings = embed_texts(samples, cfg["embedding_model"])

        # 4. Compute weighting scores
        cpw_weights = compute_cpw(embeddings)
        scw_weights = compute_scw(embeddings)

        # 5. Predictions
        baseline_pred = majority_vote(final_answers)
        cpw_pred = weighted_majority_vote(final_answers, cpw_weights)
        scw_pred = weighted_majority_vote(final_answers, scw_weights)

        # 6. Accuracy
        baseline_acc = compute_accuracy(baseline_pred, true_answer)
        cpw_acc = compute_accuracy(cpw_pred, true_answer)
        scw_acc = compute_accuracy(scw_pred, true_answer)

        results.append({
            "question_id": idx,
            "question": question,
            "true_answer": true_answer,
            "baseline_pred": baseline_pred,
            "baseline_acc": baseline_acc,
            "cpw_pred": cpw_pred,
            "cpw_acc": cpw_acc,
            "scw_pred": scw_pred,
            "scw_acc": scw_acc,
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()