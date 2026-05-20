import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import pickle
import argparse
import os
from scipy.stats import multivariate_normal
import time
from tqdm import tqdm

class BAMAlgorithm:
    """
    Batched Affine-invariant Score-based Black-box Variational Inference (BAM)
    Implements the algorithm from the paper: iterative score matching for Gaussian VI
    """
    
    def __init__(self, target_log_density, dim, batch_size=100, max_iterations=50, regularization=1e-6, initialization='prior'):
        """
        Initialize BAM algorithm
        
        Args:
            target_log_density: function that takes tensor of shape (batch_size, dim) and returns log density (batch_size,)
            dim: dimensionality of latent space
            batch_size: number of samples per iteration
            max_iterations: maximum number of iterations
            regularization: regularization strength for covariance inversion
            initialization: 'prior' or 'mle' or custom mean/cov
        """
        self.target_log_density = target_log_density
        self.dim = dim
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.initialization = initialization
        
        # Initialize variational parameters
        if initialization == 'prior':
            self.mean = torch.zeros(dim)
            self.cov = torch.eye(dim)
        elif initialization == 'mle':
            # For simplicity, use identity as default
            self.mean = torch.zeros(dim)
            self.cov = torch.eye(dim)
        else:
            # Custom initialization
            self.mean = torch.zeros(dim)
            self.cov = torch.eye(dim)
            
        self.mean = self.mean.float()
        self.cov = self.cov.float()
        
        # Store history
        self.history = {
            'means': [],
            'covariances': [],
            'score_divergences': [],
            'iterations': []
        }
        
    def target_score(self, x):
        """
        Compute score function ∇_x log p(x) using automatic differentiation
        """
        x.requires_grad_(True)
        log_p = self.target_log_density(x)
        score = torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]
        x.requires_grad_(False)
        return score
    
    def variational_score(self, x):
        """
        Compute score function ∇_x log q(x) for Gaussian variational distribution
        q(x) = N(x | μ, Σ)
        ∇_x log q(x) = -Σ^(-1)(x - μ)
        """
        # Compute Σ^(-1) with regularization for numerical stability
        cov_inv = torch.inverse(self.cov + self.regularization * torch.eye(self.dim))
        score = -torch.matmul(cov_inv, (x - self.mean).T).T
        return score
    
    def compute_score_divergence(self, samples):
        """
        Compute score-based divergence: E_q[||∇_x log p(x) - ∇_x log q(x)||^2]
        """
        target_scores = self.target_score(samples)
        variational_scores = self.variational_score(samples)
        divergence = torch.mean(torch.sum((target_scores - variational_scores) ** 2, dim=1))
        return divergence.item()
    
    def update(self):
        """
        Single iteration of BAM algorithm
        Update mean and covariance using closed-form solution
        """
        # Sample from current variational distribution
        q = dist.MultivariateNormal(self.mean, self.cov)
        samples = q.sample((self.batch_size,))
        
        # Compute target scores
        target_scores = self.target_score(samples)
        
        # Compute variational scores
        variational_scores = self.variational_score(samples)
        
        # Compute the closed-form update for mean and covariance
        # The update is derived from minimizing the score-based divergence
        # We use the weighted least squares formulation
        
        # Compute the weighted sum of target scores
        # We need to solve: E_q[ (s_p - s_q) (s_p - s_q)^T ] = 0
        # This leads to: Σ_{t+1} = Σ_t + (μ_t - z_t)(μ_t - z_t)^T + Σ_t^{-1} Σ_{i=1}^b (s_p(x_i) - s_q(x_i)) (s_p(x_i) - s_q(x_i))^T
        
        # Update mean: μ_{t+1} = μ_t + (1/b) * sum_i (s_p(x_i) - s_q(x_i))
        score_diff = target_scores - variational_scores
        mean_update = torch.mean(score_diff, dim=0)
        new_mean = self.mean + mean_update
        
        # Update covariance: Σ_{t+1} = Σ_t + (1/b) * sum_i (s_p(x_i) - s_q(x_i)) (s_p(x_i) - s_q(x_i))^T
        # This is a rank-b update to the covariance
        score_diff_centered = score_diff - torch.mean(score_diff, dim=0, keepdim=True)
        cov_update = torch.matmul(score_diff_centered.T, score_diff_centered) / self.batch_size
        new_cov = self.cov + cov_update
        
        # Ensure covariance is positive definite
        new_cov = new_cov + self.regularization * torch.eye(self.dim)
        
        # Update parameters
        self.mean = new_mean
        self.cov = new_cov
        
        # Compute score divergence for monitoring
        score_divergence = self.compute_score_divergence(samples)
        
        # Store history
        self.history['means'].append(self.mean.detach().clone())
        self.history['covariances'].append(self.cov.detach().clone())
        self.history['score_divergences'].append(score_divergence)
        self.history['iterations'].append(len(self.history['means']) - 1)
        
        return score_divergence
    
    def fit(self, verbose=True):
        """
        Run the BAM algorithm to convergence
        """
        if verbose:
            pbar = tqdm(range(self.max_iterations), desc="BAM Iterations")
        else:
            pbar = range(self.max_iterations)
            
        for i in pbar:
            score_divergence = self.update()
            
            if verbose:
                pbar.set_postfix({'score_div': f'{score_divergence:.6f}'})
                
            # Early stopping based on convergence
            if len(self.history['score_divergences']) > 1:
                prev_div = self.history['score_divergences'][-2]
                if abs(prev_div - score_divergence) < 1e-6:
                    if verbose:
                        print(f"\nConverged at iteration {i+1}")
                    break
                    
        return self.history
    
    def get_final_parameters(self):
        """Return final mean and covariance"""
        return self.mean.detach().clone(), self.cov.detach().clone()

def create_target_distributions(dim):
    """
    Create different target distributions for testing
    """
    
    # Gaussian target
    def gaussian_target(x):
        # Create a Gaussian with mean [1, 2, ..., dim/2, 0, ..., 0] and covariance I
        mean = torch.zeros(dim)
        mean[:dim//2] = 1.0
        cov = torch.eye(dim)
        log_prob = -0.5 * torch.sum((x - mean) ** 2, dim=1)
        return log_prob
    
    # Mixture of Gaussians target
    def mixture_target(x):
        # Mixture of two Gaussians
        mean1 = torch.zeros(dim)
        mean1[:dim//2] = 2.0
        mean2 = torch.zeros(dim)
        mean2[:dim//2] = -2.0
        cov = torch.eye(dim)
        
        log_prob1 = -0.5 * torch.sum((x - mean1) ** 2, dim=1)
        log_prob2 = -0.5 * torch.sum((x - mean2) ** 2, dim=1)
        
        # Equal mixture weights
        log_mix_prob = torch.logsumexp(torch.stack([log_prob1, log_prob2]), dim=0) - np.log(2)
        return log_mix_prob
    
    # Hierarchical model (linear regression with Gaussian prior)
    def hierarchical_target(x):
        # Simulate a Bayesian linear regression
        # y = Xβ + ε, β ~ N(0, I), ε ~ N(0, σ²I)
        # We're approximating p(β | y)
        
        # Generate synthetic data
        n_obs = 50
        X = torch.randn(n_obs, dim)
        true_beta = torch.randn(dim) * 0.5
        y = torch.matmul(X, true_beta) + torch.randn(n_obs) * 0.1
        
        # Prior: β ~ N(0, I)
        prior_log_prob = -0.5 * torch.sum(x ** 2, dim=1)
        
        # Likelihood: y ~ N(Xβ, σ²I)
        # log p(y | β) = -n/2 log(2πσ²) - 1/(2σ²) ||y - Xβ||²
        # We can ignore constants for optimization
        likelihood_log_prob = -0.5 * torch.sum((y - torch.matmul(X, x.T)) ** 2, dim=1)
        
        return prior_log_prob + likelihood_log_prob
    
    return {
        'gaussian': gaussian_target,
        'mixture': mixture_target,
        'hierarchical': hierarchical_target
    }

def main():
    parser = argparse.ArgumentParser(description='BAM Algorithm Implementation')
    parser.add_argument('--target', type=str, default='gaussian', choices=['gaussian', 'mixture', 'hierarchical'],
                       help='Target distribution type')
    parser.add_argument('--dim', type=int, default=10, help='Dimensionality of latent space')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for sampling')
    parser.add_argument('--max_iterations', type=int, default=50, help='Maximum number of iterations')
    parser.add_argument('--regularization', type=float, default=1e-6, help='Regularization for covariance inversion')
    parser.add_argument('--output', type=str, default='results/bam_results.pkl', help='Output file path')
    
    args = parser.parse_args()
    
    # Create target distributions
    targets = create_target_distributions(args.dim)
    target_function = targets[args.target]
    
    # Initialize BAM algorithm
    bam = BAMAlgorithm(
        target_log_density=target_function,
        dim=args.dim,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        regularization=args.regularization
    )
    
    # Run algorithm
    print(f"Running BAM algorithm on {args.target} target with dimension {args.dim}...")
    history = bam.fit(verbose=True)
    
    # Save results
    results = {
        'target': args.target,
        'dim': args.dim,
        'batch_size': args.batch_size,
        'max_iterations': args.max_iterations,
        'regularization': args.regularization,
        'final_mean': bam.mean.detach().numpy(),
        'final_covariance': bam.cov.detach().numpy(),
        'history': history,
        'convergence': len(history['score_divergences']) < args.max_iterations
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()