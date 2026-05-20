"""
Main orchestration script.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import pandas as pd
from tqdm import tqdm

from .generate import generate_responses
from .parse_answer import parse_answer
from .embed import SentenceEmbedder
from .weight import baseline_vote, cpw_weight, scw_weight
from .evaluate import aggregate_results
from .data import get_dataset_info, load_dataset_by_name
from .config import PROMPT_SHOTS, EMBED_MODEL_MATH, EMBED_MODEL_COMMONSENSE

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def process_dataset(
    dataset_key: str,
    data,
    embedder: SentenceEmbedder,
    shots: List[Tuple[str, str]],
) -> Tuple[List[str], List[str]]:
    """Generate predictions for all examples in a dataset."""
    preds_baseline = []
    preds_cpw = []
    preds_scw = []

    golds = []

    for example in tqdm(data, desc=f"Processing {dataset_key}"):
        # The dataset format may vary; we try to find the fields
        if "question" in example:
            question = example["question"]
        elif "question_text" in example:
            question = example["question_text"]
        else:
            continue  # skip if no question field

        # Ground truth
        if "answer" in example:
            gold = example["answer"]
        elif "label" in example:
            gold = example["label"]
        elif "answer_text" in example:
            gold = example["answer_text"]
        else:
            gold = None

        if gold is None:
            continue

        golds.append(gold)

        # Generate responses
        raw_responses = generate_responses(question, shots)

        # Parse answers
        answers = [parse_answer(r) for r in raw_responses]
        # Filter out None
        answers = [a for a in answers if a is not None]

        if not answers:
            # fallback: use the first raw response as answer
            answers = [parse_answer(raw_responses[0])]

        # Baseline majority vote
        pred_baseline = baseline_vote(answers)
        preds_baseline.append(pred_baseline)

        # Embed the full rationales
        embeddings = embedder.embed(answers).numpy()

        # CPW
        pred_cpw = cpw_weight(embeddings, answers)
        preds_cpw.append(pred_cpw)

        # SCW
        pred_scw = scw_weight(embeddings, answers)
        preds_scw.append(pred_scw)

    return preds_baseline, preds_cpw, preds_scw, golds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV file for results.",
    )
    args = parser.parse_args()

    all_results = []

    # Choose embedder based on dataset
    for dataset_key in ["AQuA-RAT", "SVAMP", "StrategyQA"]:
        info = get_dataset_info(dataset_key)
        data = load_dataset_by_name(info["name"], info["split"])
        # Convert to list of dicts
        data_list = list(data)

        embedder_name = (
            EMBED_MODEL_MATH if info.get("math", False) else EMBED_MODEL_COMMONSENSE
        )
        embedder = SentenceEmbedder(embedder_name)

        preds_baseline, preds_cpw, preds_scw, golds = process_dataset(
            dataset_key, data_list, embedder, PROMPT_SHOTS
        )

        # Aggregate results
        for method, preds in zip(
            ["baseline", "CPW", "SCW"],
            [preds_baseline, preds_cpw, preds_scw],
        ):
            dataset, method_name, acc = aggregate_results(
                dataset_key, method, preds, golds
            )
            all_results.append(
                {"dataset": dataset, "method": method_name, "accuracy": acc}
            )

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nResults written to {args.output}")
    print(df)


if __name__ == "__main__":
    main()