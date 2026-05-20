"""
Plot the results from the evaluation
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_results(input_file, output_file):
    """
    Plot the results from the evaluation
    """
    # Load results
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot clean and robust accuracy
    labels = ['Clean Accuracy', 'Robust Accuracy']
    values = [results['clean_accuracy'], results['robust_accuracy']]
    
    bars = ax.bar(labels, values, color=['blue', 'red'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.2f}%', ha='center', va='bottom')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CLEAN AND ROBUST PERFORMANCE OF FARE MODEL')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis from 0 to 100
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot results')
    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    
    args = parser.parse_args()
    
    plot_results(args.input, args.output)

if __name__ == '__main__':
    main()