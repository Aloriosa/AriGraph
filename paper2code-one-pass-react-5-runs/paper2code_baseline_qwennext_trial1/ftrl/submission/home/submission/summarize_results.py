#!/usr/bin/env python3
"""
Summarize the results of the reproduction.
"""
import os
import numpy as np
import json

def summarize_results():
    """
    Summarize the results of the reproduction.
    """
    # Load results
    results_file = '/home/submission/results/training_results.png'
    summary_file = '/home/submission/results/summary.txt'
    
    # Check if files exist
    if not os.path.exists(results_file):
        print("Results file not found!")
        return
    
    if not os.path.exists(summary_file):
        print("Summary file not found!")
        return
    
    # Read summary
    with open(summary_file, 'r') as f:
        summary = f.read()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary)
    
    # Save summary
    with open('/home/submission/results/summary.json', 'w') as f:
        json.dump({
        "results_file": results_file,
        "summary": summary
        }, f)
    
    print("Summary complete!")

if __name__ == "__main__":
    summarize_results()