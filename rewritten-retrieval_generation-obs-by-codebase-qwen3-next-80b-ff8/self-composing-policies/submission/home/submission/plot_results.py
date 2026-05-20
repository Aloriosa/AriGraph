#!/usr/bin/env python3
"""
Plot results from the evaluation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pathlib
import argparse
from tabulate import tabulate
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHOD_NAMES = {
    "simple": "Baseline",
    "componet": "CompoNet",
    "finetune": "FT",
    "prognet": "ProgressiveNet",
    "packnet": "PackNet",
}

METHOD_COLORS = {
    "simple": "darkgray",
    "componet": "tab:blue",
    "finetune": "tab:orange",
    "prognet": "tab:green",
    "packnet": "tab:purple",
}

METHOD_ORDER = ["simple", "componet", "finetune", "prognet", "packnet"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="data/agg_results.csv", type=str,
        help="path to the aggregated results CSV")
    parser.add_argument("--eval-csv", default="data/eval_results.pkl", type=str,
        help="path to the evaluation results pickle")
    parser.add_argument("--output-dir", default="results", type=str,
        help="directory to save plots")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluation results
    with open(args.eval_csv, 'rb') as f:
        eval_results = pickle.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(eval_results)
    
    # Calculate average returns for each model and task
    avg_returns = df.groupby(['model_type', 'task_idx'])['return'].mean().reset_index()
    
    # Plot learning curves
    plt.figure(figsize=(12, 8))
    
    for model_type in METHOD_ORDER:
        if model_type not in avg_returns['model_type'].unique():
            continue
            
        model_data = avg_returns[avg_returns['model_type'] == model_type]
        plt.plot(model_data['task_idx'], model_data['return'], 
                label=METHOD_NAMES[model_type], 
                color=METHOD_COLORS[model_type], 
                linewidth=2)
        
        # Add markers
        plt.scatter(model_data['task_idx'], model_data['return'], 
                   color=METHOD_COLORS[model_type], s=50, alpha=0.7)
    
    plt.xlabel('Task Index')
    plt.ylabel('Average Episodic Return')
    plt.title('Continual RL Performance on Meta-World 20-Task Sequence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(20))
    plt.savefig(os.path.join(args.output_dir, 'task_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and plot forgetting metric
    # Forgetting = average performance drop on previous tasks
    # We'll calculate the average performance on the first 10 tasks vs last 10 tasks
    first_10 = avg_returns[avg_returns['task_idx'] < 10].groupby('model_type')['return'].mean()
    last_10 = avg_returns[avg_returns['task_idx'] >= 10].groupby('model_type')['return'].mean()
    
    forgetting = (first_10 - last_10) / first_10
    
    plt.figure(figsize=(10, 6))
    forgetting_plot = forgetting.reindex(METHOD_ORDER)
    bars = plt.bar(range(len(forgetting_plot)), forgetting_plot.values, 
                   color=[METHOD_COLORS[m] for m in METHOD_ORDER])
    
    plt.xlabel('Method')
    plt.ylabel('Forgetting (Performance Drop)')
    plt.title('Forgetting Metric on Meta-World 20-Task Sequence')
    plt.xticks(range(len(forgetting_plot)), [METHOD_NAMES[m] for m in METHOD_ORDER], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(forgetting_plot.values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'forgetting.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary table
    print("\n=== FINAL RESULTS ===")
    summary_data = []
    for model_type in METHOD_ORDER:
        if model_type not in avg_returns['model_type'].unique():
            continue
            
        model_data = avg_returns[avg_returns['model_type'] == model_type]
        final_avg = model_data['return'].mean()
        final_std = model_data['return'].std()
        forgetting_val = forgetting.get(model_type, 0)
        
        summary_data.append([
            METHOD_NAMES[model_type],
            f"{final_avg:.3f} ± {final_std:.3f}",
            f"{forgetting_val:.3f}"
        ])
    
    print(tabulate(summary_data, headers=["Method", "Avg Return", "Forgetting"], tablefmt="grid"))
    
    # Save summary to file
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write("=== FINAL RESULTS ===\n")
        f.write(tabulate(summary_data, headers=["Method", "Avg Return", "Forgetting"], tablefmt="grid"))
    
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    main()