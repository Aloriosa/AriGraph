#!/usr/bin/env python3
"""
Plot results from BaM experiments
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_results(input_file: str, output_file: str):
    """
    Plot results from BaM experiments
    """
    with open(input_file, 'rb') as f:
        results = pickle.load(f)
    
    # Extract results
    mu = results['mu']
    Sigma = results['Sigma']
    history = results['history']
    args = results['args']
    
    # Create plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot mean evolution
    iterations = [h['iteration'] for h in history]
    means = [np.linalg.norm(h['mu']) for h in history]
    ax[0].plot(iterations, means, label='BaM', linewidth=2)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('||mu||')
    ax[0].set_title('Mean Evolution')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot covariance trace
    traces = [np.trace(h['Sigma']) for h in history]
    ax[1].plot(iterations, traces, label='BaM', linewidth=2)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('tr(Sigma)')
    ax[1].set_title('Covariance Trace')
    ax[1].legend()
    ax[1].grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot BaM results')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file')
    parser.add_argument('--output', type=str, required=True, help='Output PNG file')
    
    args = parser.parse_args()
    
    plot_results(args.input, args.output)

if __name__ == '__main__':
    main()