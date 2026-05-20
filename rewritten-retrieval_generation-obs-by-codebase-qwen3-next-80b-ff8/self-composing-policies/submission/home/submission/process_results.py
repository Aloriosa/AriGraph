#!/usr/bin/env python3
"""
Process results from tensorboard logs and generate aggregated results.
"""
import pandas as pd
import pathlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
import argparse
from tabulate import tabulate
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHOD_NAMES = {
    "simple": "Baseline",
    "finetune": "FT",
    "componet": "CompoNet",
    "prognet": "ProgressiveNet",
    "packnet": "PackNet",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs_all", type=str,
        help="directory where the tensorboard data is stored")
    parser.add_argument("--no-cache", default=False, action="store_true",
        help="whether to disable the cache option. If not provided and `--save-dir` exists, skips processing tensorboard files")
    parser.add_argument("--save-csv", default="data/agg_results.csv", type=str,
        help="filename of the CSV to store the processed tensorboard results. Once processed, can be used as cache.")
    parser.add_argument("--smoothing-window", type=int, default=100)
    parser.add_argument("--eval-csv", type=str, default="data/eval_results.csv",
        help="path to the CSV where the results of evaluations are stored. If the file doesn't exist, forgetting is not computed.")
    parser.add_argument("--no-plots", default=False, action="store_true")
    return parser.parse_args()

def parse_metadata(ea):
    md = ea.Tensors("hyperparameters/text_summary")[0]
    md_bytes = md.tensor_proto.SerializeToString()
    
    # remove first non-ascii characters and parse
    start = md_bytes.index(b"|")
    md_str = md_bytes[start:].decode("ascii")
    
    md = {}
    for row in md_str.split("\n")[2:]:
        s = row.split("|")[1:-1]
        if len(s) == 2:
            key = s[0].strip()
            value = s[1].strip()
            try:
                md[key] = float(value)
            except ValueError:
                md[key] = value
    return md

def process_tensorboard(run_dir):
    """Process tensorboard logs from a run directory."""
    results = []
    
    for subdir in os.listdir(run_dir):
        if not subdir.startswith("run_"):
            continue
            
        run_path = os.path.join(run_dir, subdir)
        if not os.path.isdir(run_path):
            continue
            
        # Look for tensorboard files
        tb_files = [f for f in os.listdir(run_path) if f.startswith("events.out.tfevents")]
        if not tb_files:
            continue
            
        tb_path = os.path.join(run_path, tb_files[0])
        
        # Load tensorboard data
        ea = EventAccumulator(tb_path)
        ea.Reload()
        
        # Parse metadata
        metadata = parse_metadata(ea)
        
        # Extract scalar data
        if "train/episode_return" in ea.Tags()["scalars"]:
            scalars = ea.Scalars("train/episode_return")
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            
            # Create DataFrame
            df = pd.DataFrame({
                "step": steps,
                "episode_return": values,
                "model_type": metadata.get("model_type", "unknown"),
                "seed": metadata.get("seed", 0)
            })
            
            # Add smoothing
            df["episode_return_smoothed"] = df["episode_return"].rolling(window=100, min_periods=1).mean()
            
            results.append(df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    args = parse_args()
    
    # Check if cache exists and we're not forcing reprocessing
    if os.path.exists(args.save_csv) and not args.no_cache:
        print(f"Loading cached results from {args.save_csv}")
        df = pd.read_csv(args.save_csv)
    else:
        print(f"Processing tensorboard files from {args.runs_dir}")
        df = process_tensorboard(args.runs_dir)
        
        # Save to cache
        df.to_csv(args.save_csv, index=False)
        print(f"Saved processed results to {args.save_csv}")
    
    # Load evaluation results if available
    if os.path.exists(args.eval_csv):
        eval_df = pd.read_pickle(args.eval_csv)
        print(f"Loaded evaluation results from {args.eval_csv}")
        
        # Calculate forgetting metric
        # Forgetting = average performance drop on previous tasks
        # This is a simplified version - in practice we'd need more detailed tracking
        if len(eval_df) > 0:
            # Group by model and task
            task_performance = eval_df.groupby(['model_type', 'task_idx'])['return'].mean()
            
            # Calculate average performance on last 5 tasks
            last_task_performance = eval_df[eval_df['task_idx'] >= 15].groupby('model_type')['return'].mean()
            
            # Calculate average performance on first 5 tasks
            first_task_performance = eval_df[eval_df['task_idx'] < 5].groupby('model_type')['return'].mean()
            
            # Calculate forgetting as performance drop
            forgetting = (first_task_performance - last_task_performance) / first_task_performance
            
            print("\nForgetting metrics:")
            print(forgetting.round(3))
    
    # Print summary
    print("\nSummary of results:")
    summary = df.groupby('model_type')['episode_return_smoothed'].mean().round(3)
    print(summary)
    
    # Create plots if requested
    if not args.no_plots:
        plt.figure(figsize=(12, 8))
        
        for model_type in df['model_type'].unique():
            if model_type not in METHOD_NAMES:
                continue
                
            model_df = df[df['model_type'] == model_type]
            mean_return = model_df.groupby('step')['episode_return_smoothed'].mean()
            std_return = model_df.groupby('step')['episode_return_smoothed'].std()
            
            plt.plot(mean_return.index, mean_return.values, label=METHOD_NAMES[model_type])
            plt.fill_between(mean_return.index, 
                           mean_return.values - std_return.values, 
                           mean_return.values + std_return.values, 
                           alpha=0.2)
        
        plt.xlabel('Timesteps')
        plt.ylabel('Average Episodic Return')
        plt.title('Continual RL Performance on Meta-World')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plots saved to results/learning_curves.png")

if __name__ == "__main__":
    main()