import json
import os
import csv
import argparse
import numpy as np

def summarize_results(results_dir, output_file):
    """Summarize results from all experiments"""
    summary = []
    
    # Process LBCS results
    lbc_dirs = [
        os.path.join(results_dir, "fashion_mnist"),
        os.path.join(results_dir, "cifar10"),
        os.path.join(results_dir, "svhn")
    ]
    
    for lbc_dir in lbc_dirs:
        if os.path.exists(lbc_dir):
            results_file = os.path.join(lbc_dir, "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    result = json.load(f)
                
                # Extract dataset name
                dataset = result["dataset"]
                if dataset == "fashion_mnist":
                    dataset_name = "FashionMNIST"
                elif dataset == "cifar10":
                    dataset_name = "CIFAR-10"
                elif dataset == "svhn":
                    dataset_name = "SVHN"
                else:
                    dataset_name = dataset
                
                summary.append({
                    "method": "LBCS",
                    "dataset": dataset_name,
                    "full_data_accuracy": result["full_data_accuracy"],
                    "performance_threshold": result["performance_threshold"],
                    "selected_coreset_size": result["selected_coreset_size"],
                    "final_accuracy": result["final_accuracy"],
                    "test_accuracy": result["test_accuracy"],
                    "coreset_ratio": result["selected_coreset_size"] / 60000 if dataset == "fashion_mnist" else 
                                   result["selected_coreset_size"] / 50000 if dataset == "cifar10" else 
                                   result["selected_coreset_size"] / 73257
                })
    
    # Process baseline results
    baseline_dir = os.path.join(results_dir, "baselines")
    if os.path.exists(baseline_dir):
        results_file = os.path.join(baseline_dir, "baseline_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                baseline_results = json.load(f)
            
            # Extract dataset name from path
            dataset_name = "FashionMNIST"
            
            for method, result in baseline_results.items():
                summary.append({
                    "method": method,
                    "dataset": dataset_name,
                    "full_data_accuracy": None,
                    "performance_threshold": None,
                    "selected_coreset_size": result["coreset_size"],
                    "final_accuracy": result["accuracy"],
                    "test_accuracy": None,
                    "coreset_ratio": result["coreset_ratio"]
                })
    
    # Write summary to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "dataset", "full_data_accuracy", "performance_threshold", 
            "selected_coreset_size", "final_accuracy", "test_accuracy", "coreset_ratio"
        ])
        
        for row in summary:
            writer.writerow([
                row["method"], row["dataset"], row["full_data_accuracy"], row["performance_threshold"],
                row["selected_coreset_size"], row["final_accuracy"], row["test_accuracy"], row["coreset_ratio"]
            ])
    
    print(f"Summary saved to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Summarize LBCS results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing results (default: results)')
    parser.add_argument('--output', type=str, default='results/summary.csv',
                       help='Output file for summary (default: results/summary.csv)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    summarize_results(args.results_dir, args.output)