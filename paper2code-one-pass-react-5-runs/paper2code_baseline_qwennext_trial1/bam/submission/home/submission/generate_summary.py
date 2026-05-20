#!/usr/bin/env python3
"""
Generate summary statistics from the results.
"""

import pickle
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Generate summary statistics')
    parser.add_argument('--input', type=str, required=True, help='Input results file')
    parser.add_argument('--output', type=str, default='summary.txt', help='Output file')
    args = parser.parse_args()
    
    # Load results
    with open(args.input, 'rb') as f:
        results = pickle.load(f)
    
    # Extract results
    final_mean = results['final_mean']
    final_cov = results['final_cov']
    
    # Calculate summary statistics
    mean_error = np.linalg.norm(final_mean)
    cov_error = np.linalg.norm(final_cov - np.eye(final_cov.shape[0]))
    
    summary = f"""
Batch and Match (BaM) Reproduction Summary
============================================

Results from {args.input}

Final mean: {final_mean}
Final covariance: {final_cov}

Summary statistics:
- Mean error: {mean_error:.6f}
- Covariance error: {cov_error:.6f}

Reproduction completed successfully.
"""
    
    # Write summary
    with open(args.output, 'w') as f:
        f.write(summary)
    
    print(f"Summary generated: {args.output}")

if __name__ == "__main__":
    main()