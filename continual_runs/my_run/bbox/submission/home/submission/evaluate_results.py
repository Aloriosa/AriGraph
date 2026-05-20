#!/usr/bin/env python3
"""
Evaluate the results of BBoxAdapter adaptation
"""
import os
import json
import csv
import argparse
import numpy as np

def evaluate_results(input_dir: str, output_file: str):
    """
    Evaluate the results from BBoxAdapter adaptation
    """
    results_path = os.path.join(input_dir, "results.json")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract accuracy values
    datasets = ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"]
    accuracies = []
    
    for dataset in datasets:
        if dataset in results:
            accuracies.append(results[dataset])
            print(f"{dataset}: {results[dataset]:.4f}")
        else:
            print(f"{dataset}: N/A")
    
    # Calculate average
    if len(accuracies) > 0:
        avg_accuracy = np.mean(accuracies)
        print(f"Average Accuracy: {avg_accuracy:.4f}")
    else:
        avg_accuracy = 0.0
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Accuracy"])
        for dataset in datasets:
            if dataset in results:
                writer.writerow([dataset, results[dataset]])
            else:
                writer.writerow([dataset, "N/A"])
        writer.writerow(["Average", avg_accuracy])
    
    print(f"\nResults saved to {output_file}")
    
    # Check if we achieved paper's reported improvement
    paper_improvement = 0.0677  # 6.77%
    print(f"\nPaper-reported improvement: {paper_improvement:.4f}")
    print(f"Reproduced improvement: {avg_accuracy:.4f}")
    
    if avg_accuracy >= paper_improvement:
        print("✅ SUCCESS: Reproduced paper's improvement")
    else:
        print("⚠️  WARNING: Did not fully reproduce paper's improvement")

def main():
    parser = argparse.ArgumentParser(description="Evaluate BBoxAdapter results")
    parser.add_argument("--input_dir", type=str, default="./results", help="Input directory with results")
    parser.add_argument("--output_file", type=str, default="./results/accuracy_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    evaluate_results(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()