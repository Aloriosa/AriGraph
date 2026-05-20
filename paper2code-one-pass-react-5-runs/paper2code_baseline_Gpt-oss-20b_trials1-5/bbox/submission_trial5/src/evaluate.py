#!/usr/bin/env python3
import json
from pathlib import Path

from src.utils import exact_match

def main():
    # Load test set
    test_path = Path("data/test.jsonl")
    test_samples = [json.loads(line) for line in test_path.read_text().splitlines()]

    # Load predictions
    pred_path = Path("outputs/predictions.jsonl")
    pred_samples = [json.loads(line) for line in pred_path.read_text().splitlines()]

    # Compute accuracy
    correct = 0
    for gold, pred in zip(test_samples, pred_samples):
        correct += exact_match(pred["prediction"], gold["answer"])

    acc = correct / len(test_samples)
    print(f"Evaluation finished. accuracy: {acc:.2%}")

    # Save metrics
    Path("outputs/metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2))
    print("metrics written to outputs/metrics.json")

if __name__ == "__main__":
    main()