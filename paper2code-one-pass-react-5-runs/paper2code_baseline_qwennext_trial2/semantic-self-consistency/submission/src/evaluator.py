import json
import os
from typing import List, Dict, Any
import numpy as np

class Evaluator:
    def __init__(self, dataset_name: str, dataset: List[Dict[str, Any]]):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.results = []
    
    def evaluate(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
        """Evaluate predictions against ground truth"""
        correct = sum(1 for p, t in zip(predictions, ground_truth) if p.strip() == t.strip())
        accuracy = correct / len(predictions)
        
        return {
            "dataset": self.dataset_name,
            "total_samples": len(predictions),
            "correct": correct,
            "accuracy": accuracy
    }

    def save_results(self, filepath: str):
        """Save results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Mock data
    data = [{"question": "What is 2+2?", "answer": "4"}]
    evaluator = Evaluator("AQuA-RAT", data)
    acc = evaluator.evaluate(["4", "5"], ["4", "4"])
    print("Accuracy:", acc)