#!/usr/bin/env python
"""
Semantic Self‑Consistency Reproduction (simplified version)

This script implements a minimal pipeline that:
  • Loads small toy datasets (AQuA‑RAT, SVAMP, StrategyQA)
  • Generates chain‑of‑thought samples using open LLMs (gpt2, distilgpt2)
  • Extracts answers from the generations
  • Featurises the full rationales with a sentence‑transformer
  • Computes three aggregation strategies:
        – Majority vote (self‑consistency baseline)
        – Centroid Proximity Weighting (CPW)
        – Semantic Consensus Weighting (SCW)
  – Optionally removes outliers with Isolation Forest before re‑applying CPW/SCW
  • Evaluates accuracy and writes a CSV table

The goal is to provide a reproducible, lightweight reference for the
methodology described in the paper.  The numbers will not match the
paper’s large‑scale experiments but the pipeline structure follows the
same principles.
"""

import argparse
import json
import os
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def load_dataset(path: str):
    """Read a JSON‑Lines dataset."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def generate_samples(pipe, prompt: str, n_samples: int = 5, temp: float = 0.7, max_len: int = 200):
    """Generate multiple chain‑of‑thought samples."""
    outputs = pipe(prompt,
                   max_new_tokens=max_len,
                   temperature=temp,
                   num_return_sequences=n_samples,
                   do_sample=True)
    return [o["generated_text"] for o in outputs]


def extract_answer(text: str) -> str:
    """
    Very light‑weight extractor:
      * Split by newlines, take the last non‑empty line
      * If the line contains a digit, return the first number
      * Otherwise return the last token
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    tokens = candidate.split()
    for tok in reversed(tokens):
        if tok.isdigit():
            return tok
    return tokens[-1] if tokens else ""


def compute_cpw(embeddings: np.ndarray, answers: list, normalize: bool = True) -> str:
    """Centroid Proximity Weighting."""
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    if normalize:
        norm_dist = distances / np.sum(distances)
        weights = 1.0 / (norm_dist + 1e-8)
    else:
        weights = 1.0 / distances
    weight_sum = collections.defaultdict(float)
    for w, ans in zip(weights, answers):
        weight_sum[ans] += w
    best = max(weight_sum.items(), key=lambda x: x[1])[0]
    return best


def compute_scw(embeddings: np.ndarray, answers: list) -> str:
    """Semantic Consensus Weighting."""
    cos_sim = cosine_similarity(embeddings)
    scores = np.sum(cos_sim, axis=1)
    score_sum = collections.defaultdict(float)
    for s, ans in zip(scores, answers):
        score_sum[ans] += s
    best = max(score_sum.items(), key=lambda x: x[1])[0]
    return best


def apply_isolation_forest(embeddings: np.ndarray, contamination: float = 0.2):
    """Return indices of non‑outlier samples."""
    if len(embeddings) <= 2:
        return np.arange(len(embeddings))
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(embeddings)
    mask = iso.predict(embeddings) == 1
    return np.where(mask)[0]


def run_pipeline(dataset_name: str,
                 data: list,
                 model_name: str,
                 pipe,
                 featurizer,
                 n_samples: int = 5):
    """Run the full pipeline for one dataset / model pair."""
    results = []
    for q in tqdm(data, desc=f"{dataset_name} – {model_name}"):
        question = q["question"]
        gold = q["answer"]
        prompt = f"{question}\nAnswer:"
        samples = generate_samples(pipe, prompt, n_samples=n_samples,
                                   temp=0.7, max_len=200)
        answers = [extract_answer(s) for s in samples]
        embeddings = featurizer.encode(samples, convert_to_numpy=True)

        # Baseline majority vote
        major = collections.Counter(answers).most_common(1)[0][0]

        # CPW & SCW on all samples
        cpw = compute_cpw(embeddings, answers)
        scw = compute_scw(embeddings, answers)

        # Outlier removal + re‑apply
        idx = apply_isolation_forest(embeddings, contamination=0.2)
        emb_filt = embeddings[idx]
        ans_filt = [answers[i] for i in idx]
        cpw_iso = compute_cpw(emb_filt, ans_filt)
        scw_iso = compute_scw(emb_filt, ans_filt)

        results.append({
            "question": question,
            "gold": gold,
            "major": major,
            "cpw": cpw,
            "scw": scw,
            "cpw_iso": cpw_iso,
            "scw_iso": scw_iso,
            "answers": answers,
        })
    return results


def evaluate(results: list) -> dict:
    """Compute accuracy percentages for each method."""
    metrics = {}
    for key in ["major", "cpw", "scw", "cpw_iso", "scw_iso"]:
        correct = sum(1 for r in results if r[key].strip().lower() == r["gold"].strip().lower())
        metrics[key] = correct / len(results) * 100
    return metrics


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Semantic Self‑Consistency reproduction.")
    parser.add_argument("--data_dir", default="data", help="Folder with the three toy datasets.")
    parser.add_argument("--output", default="results/results.csv", help="CSV file to write results.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load generation pipelines
    models = {
        "gpt2": pipeline(
            "text-generation",
            model="gpt2",
            device=0 if torch.cuda.is_available() else -1,
            batch_size=4,
        ),
        "distilgpt2": pipeline(
            "text-generation",
            model="distilgpt2",
            device=0 if torch.cuda.is_available() else -1,
            batch_size=4,
        ),
    }

    # Featuriser
    featurizer = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Dataset paths
    ds_paths = {
        "AQuA‑RAT": os.path.join(args.data_dir, "aquad_rat.jsonl"),
        "SVAMP": os.path.join(args.data_dir, "svamp.jsonl"),
        "StrategyQA": os.path.join(args.data_dir, "strategyqa.jsonl"),
    }

    all_rows = []

    for ds_name, ds_path in ds_paths.items():
        data = load_dataset(ds_path)
        for model_name, pipe in models.items():
            results = run_pipeline(ds_name, data, model_name, pipe, featurizer, n_samples=5)
            metrics = evaluate(results)
            for method, acc in metrics.items():
                all_rows.append(
                    {
                        "dataset": ds_name,
                        "model": model_name,
                        "method": method,
                        "accuracy": acc,
                    }
                )

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    import torch  # imported here to keep top‑level imports minimal
    main()