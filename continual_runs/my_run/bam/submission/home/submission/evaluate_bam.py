import numpy as np
import pickle
import argparse
import csv
import os
from scipy.stats import multivariate_normal

def evaluate_bam_results(input_file, target, output_file):
    """
    Evaluate BAM algorithm results and generate summary statistics
    """
    with open(input_file, 'rb') as f:
        results = pickle.load(f)
    
    # Extract results
    final_mean = results['final_mean']
    final_cov = results['final_covariance']
    history = results['history']
    target_dist = results['target']
    dim = results['dim']
    
    # Compute ground truth for comparison
    if target == 'gaussian':
        # Ground truth is known: mean = [1,1,...,0,0,...], cov = I
        true_mean = np.zeros(dim)
        true_mean[:dim//2] = 1.0
        true_cov = np.eye(dim)
        
    elif target == 'mixture':
        # For mixture of two Gaussians, we can approximate the mean and covariance
        # The mixture has mean 0 and covariance I + 4*I for the first dim/2 dimensions
        true_mean = np.zeros(dim)
        true_cov = np.eye(dim)
        true_cov[:dim//2, :dim//2] += 4.0  # Variance increases due to mixture
        
    elif target == 'hierarchical':
        # For hierarchical model, we need to compute the true posterior
        # In linear regression with Gaussian prior and likelihood, posterior is Gaussian
        # We'll compute it analytically
        n_obs = 50
        X = np.random.randn(n_obs, dim)
        true_beta = np.random.randn(dim) * 0.5
        y = np.dot(X, true_beta) + np.random.randn(n_obs) * 0.1
        
        # Posterior: p(β|y) ~ N(μ_post, Σ_post)
        # Σ_post = (X^T X + I)^(-1)
        # μ_post = Σ_post X^T y
        XtX = np.dot(X.T, X)
        XtY = np.dot(X.T, y)
        true_cov = np.linalg.inv(XtX + np.eye(dim))
        true_mean = np.dot(true_cov, XtY)
    
    # Compute evaluation metrics
    mean_error = np.linalg.norm(final_mean - true_mean)
    cov_error = np.linalg.norm(final_cov - true_cov)
    
    # Compute relative errors
    mean_relative_error = mean_error / (np.linalg.norm(true_mean) + 1e-8)
    cov_relative_error = cov_error / (np.linalg.norm(true_cov) + 1e-8)
    
    # Compute score divergence at convergence
    final_score_divergence = history['score_divergences'][-1] if history['score_divergences'] else np.nan
    
    # Compute number of iterations to convergence
    iterations = len(history['score_divergences'])
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'target', 'dim', 'batch_size', 'max_iterations', 'regularization',
            'final_mean_error', 'final_cov_error', 'mean_relative_error', 
            'cov_relative_error', 'final_score_divergence', 'iterations', 
            'converged'
        ])
        writer.writerow([
            target, dim, results['batch_size'], results['max_iterations'], 
            results['regularization'], mean_error, cov_error, mean_relative_error,
            cov_relative_error, final_score_divergence, iterations,
            results['convergence']
        ])
    
    print(f"Results for {target} target saved to {output_file}")
    print(f"Mean error: {mean_error:.6f}")
    print(f"Covariance error: {cov_error:.6f}")
    print(f"Final score divergence: {final_score_divergence:.6f}")
    print(f"Iterations: {iterations}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate BAM Algorithm Results')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file with results')
    parser.add_argument('--target', type=str, required=True, choices=['gaussian', 'mixture', 'hierarchical'],
                       help='Target distribution type')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    
    args = parser.parse_args()
    evaluate_bam_results(args.input, args.target, args.output)

if __name__ == "__main__":
    main()