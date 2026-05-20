"""
Summary statistics for NPSE algorithm.
"""
import numpy as np
import os
import sys

def generate_summary(input_file, output_file):
    """
    Generate summary statistics from the input file.
    """
    # Load the input file
    data = np.loadtxt(input_file, delimiter=',', skiprows=1)
    
    # Calculate summary statistics
    r_count = data[:, 1]
    mean_r_count = np.mean(r_count)
    std_r_count = np.std(r_count)
    min_r_count = np.min(r_count)
    max_r_count = np.max(r_count)
    
    # Write the summary file
    with open(output_file, 'w') as f:
        f.write("Summary Statistics for NPSE Algorithm\n")
        f.write("="*50 + "\n")
        f.write(f"Mean 'r' count: {mean_r_count:.4f}\n")
        f.write(f"Std 'r' count: {std_r_count:.4f}\n")
        f.write(f"Min 'r' count: {min_r_count:.4f}\n")
        f.write(f"Max 'r' count: {max_r_count:.4f}\n")
    
    print(f"Summary statistics saved to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python summary.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    generate_summary(input_file, output_file)