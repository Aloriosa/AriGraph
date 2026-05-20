#!/usr/bin/env python3
"""
Generate summary of reproduction results
"""
import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate APT reproduction results summary")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("APT REPRODUCTION RESULTS SUMMARY")
    print("=" * 80)
    
    # Check if results directory exists
    if not os.path.exists(args.output_dir):
        print(f"Results directory {args.output_dir} not found!")
        return
    
    # Process each subdirectory
    for subdir in os.listdir(args.output_dir):
        subdir_path = os.path.join(args.output_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        # Look for metrics.json
        metrics_file = os.path.join(subdir_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                
            print(f"\n{subdir.upper()}:")
            print("-" * 40)
            print(f"Best evaluation score: {metrics['best_eval_score']:.4f}")
            print(f"Final sparsity: {metrics['final_sparsity']:.1%}")
            print(f"Final tuning factor: {metrics['final_tuning_factor']:.1f}")
            print(f"Training losses: {metrics['train_losses'][-1]:.4f}")
            print(f"Final evaluation score: {metrics['eval_scores'][-1]:.4f}")
            
            # Map to paper results
            if "sst2" in subdir.lower():
                print(f"Paper result (SST-2): 94.5% accuracy with 60% sparsity")
                print(f"Reproduction result: {metrics['best_eval_score']:.1%} accuracy")
                print(f"Performance retention: {metrics['best_eval_score']:.1%} of paper target")
            elif "cnn_dm" in subdir.lower():
                print(f"Paper result (CNN/DM): ROUGE scores close to full fine-tuning")
                print(f"Reproduction result: {metrics['best_eval_score']:.4f} (proxy score)")
            elif "llama" in subdir.lower():
                print(f"Paper result (LLaMA-2 7B): 50.0% average on Open LLM leaderboard")
                print(f"Reproduction result: {metrics['best_eval_score']:.4f} (proxy score)")
                
    print("\n" + "=" * 80)
    print("NOTE: This is a reproduction with simplified implementation.")
    print("The actual paper results require full training on the original datasets and models.")
    print("This reproduction demonstrates the core APT methodology within computational constraints.")
    print("=" * 80)

if __name__ == "__main__":
    main()