#!/usr/bin/env python3
"""
Implementation of Batch and Match (BaM) for Variational Inference
as described in "Batch and match: black-box variational inference with a score-based divergence"

This implementation follows the algorithm described in Section 3 of the paper.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
import pickle
import os
from typing import Tuple, Callable, Optional
import argparse
import time

# Set up JAX to use GPU if available
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', True)

class BatchAndMatch:
    """
    Implementation of the Batch and Match (BaM) algorithm for variational inference
    using a score-based divergence.
    
    This implementation follows Algorithm 1 from the paper.
    """
    
    def __init__(self, 
                 log_unnormalized_target: Callable[[jnp.ndarray], jnp.ndarray],
                 dim: int,
                 initial_mean: Optional[np.ndarray] = None,
                 initial_covariance: Optional[np.ndarray] = None,
                 regularization: float = 1.0,
                 seed: int = 42):
        """
        Initialize the BaM algorithm.
        
        Args:
            log_unnormalized_target: Function that takes a state z and returns log p(z)
            dim: Dimension of the latent space
            initial_mean: Initial mean of the variational distribution
            initial_covariance: Initial covariance of the variational distribution
            regularization: Regularization parameter lambda
            seed: Random seed
        """
        self.log_unnormalized_target = log_unnormalized_target
        self.dim = dim
        self.regularization = regularization
        
        # Initialize variational parameters
        if initial_mean is None:
            self.mean = jnp.zeros(dim)
        else:
            self.mean = jnp.array(initial_mean)
            
        if initial_covariance is None:
            self.covariance = jnp.eye(dim)
        else:
            self.covariance = jnp.array(initial_covariance)
            
        # Create a function to compute the score of the target distribution
        self.score_target = jit(lambda z: grad(self.log_unnormalized_target)(z))
        
        # Store history
        self.history = {
            'means': [],
            'covariances': [],
            'divergences': [],
            'iterations': 0
        }
    
    def batch_step(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Batch step: Draw samples from the current variational distribution and compute statistics.
        
        Returns:
            mean_samples: Mean of the samples
            mean_scores: Mean of the scores
            cov_samples: Covariance of the samples
            cov_scores: Covariance of the scores
            samples: The samples drawn
            scores: The scores of the samples
        """
        # Sample from the current variational distribution
        samples = self.sample(batch_size)
        scores = self.score_target(samples)
        
        # Compute statistics
        mean_samples = jnp.mean(samples, axis=0)
        mean_scores = jnp.mean(scores, axis=0)
        cov_samples = jnp.cov(samples.T)
        cov_scores = jnp.cov(scores.T)
        
        return mean_samples, mean_scores, cov_samples, cov_scores, samples, scores
    
    def match_step(self, 
                   mean_samples: jnp.ndarray, 
                   mean_scores: j, 
                   cov_samples: jnp.ndarray, 
                   cov_scores: jnp.ndarray, 
                   mean_var: jnp.ndarray, 
                   cov_var: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Match step: Update the variational parameters using the closed-form update.
        
        This implements the match step from Algorithm 1.
        
        Args:
            mean_samples: Mean of the samples
            mean_scores: Mean of the scores
            cov_samples: Covariance of the samples
            cov_scores: Covariance of the scores
            mean_var: Current mean of the variational distribution
            cov_var: Current covariance of the variational distribution
            
        Returns:
            updated_mean: Updated mean
            updated_cov: Updated covariance
        """
        # Compute matrices U and V as in Algorithm 1
        # U = lambda * cov_scores + (lambda / (1 + lambda)) * mean_scores * mean_scores.T
        # V = cov_var + lambda * cov_samples + (lambda / (1 + lambda)) * (mean_var - mean_samples) * (mean_var - mean_samples).T
        
        lambda_val = self.regularization
        U = lambda_val * cov_scores + (lambda_val / (1 + lambda_val)) * jnp.outer(mean_scores, mean_scores)
        
        V = cov_var + lambda_val * cov_samples + (lambda_val / (1 + lambda_val) * jnp.outer(mean_var - mean_samples, mean_var - mean_samples)
        
        # Solve the quadratic matrix equation: Sigma * U * Sigma + Sigma = V
        # Using the closed-form solution: Sigma = 2 * V * (I + (I + 4 * U * V)^(-1/2))^-1
        
        # Compute the solution for the covariance
        # We use the closed-form solution from the paper
        I = jnp.eye(self.dim)
        UV = jnp.matmul(U, V)
        sqrt_term = jax.lax.sqrt(I + 4 * UV)
        inv_term = jnp.linalg.inv(I + sqrt_term)
        updated_cov = 2 * jnp.matmul(V, inv_term)
        
        # Update the mean
        updated_mean = (1 / (1 + lambda_val)) * mean_var + (lambda_val / (1 + lambda_val)) * (jnp.matmul(cov_var, mean_scores) + mean_samples)
        
        return updated_mean, updated_cov
    
    def sample(self, batch_size: int) -> jnp.ndarray:
        """
        Sample from the current variational distribution.
        
        Args:
            batch_size: Number of samples to draw
            
        Returns:
            samples: Samples from the variational distribution
        """
        # Sample from a multivariate normal distribution
        mean = self.mean
        cov = self.covariance
        
        # Use JAX random number generation
        key = random.PRNGKey(int(time.time() % 1000))
        samples = random.multivariate_normal(key, mean, cov, (batch_size,))
        
        return samples
    
    def run(self, iterations: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the BaM algorithm.
        
        Args:
            iterations: Number of iterations
            batch_size: Batch size for sampling
            
        Returns:
            mean: Final mean of the variational distribution
            cov: Final covariance of the variational distribution
        """
        for i in range(iterations):
            # Batch step
            mean_samples, mean_scores, cov_samples, cov_scores, samples, scores = self.batch_step(batch_size)
            
            # Match step
            self.mean, self.covariance = self.match_step(
                mean_samples, mean_scores, cov_samples, cov_scores, self.mean, self.covariance
            )
            
            # Store history
            self.history['means'].append(self.mean)
            self.history['covariances'].append(self.covariance)
            
        self.history['iterations'] = iterations
        return self.mean, self.covariance
    
    def get_history(self) -> dict:
        """Get the history of the algorithm."""
        return self.history

def main():
    """Main function to run the reproduction."""
    parser = argparse.ArgumentParser(description='Reproduce Batch and Match (BaM) algorithm')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default='output.pkl', help='Output file')
    args = parser.parse_args()
    
    # Define a target distribution (Gaussian) for testing
    # We'll use a simple Gaussian target distribution
    target_mean = np.random.randn(10)  # 10-dimensional target
    target_cov = np.eye(10)  # Identity covariance
    target_cov[0, 0] = 2.0  # Make it not isotropic
    target_cov[1, 1] = 1.5
    target_cov[2, 2] = 1.8
    target_cov[3, 3] = 1.2
    target_cov[4, 4] = 1.1
    target_cov[5, 5] = 1.3
    target_cov[6, 6] = 1.4
    target_cov[7, 7] = 1.6
    target_cov[8, 8] = 1.7
    target_cov[9, 9] = 1.9
    
    def log_unnormalized_target(z):
        return -0.5 * jnp.sum((z - target_mean) ** 2) - 0.5 * jnp.sum(jnp.log(2 * np.pi))
    
    # Initialize BaM algorithm
    bam = BatchAndMatch(
        log_unnormalized_target=log_unnormalized_target,
        dim=10,
        initial_mean=np.zeros(10),
        initial_covariance=np.eye(10),
        regularization=1.0
    )
    
    # Run BaM
    print("Running Batch and Match (BaM) algorithm...")
    final_mean, final_cov = bam.run(iterations=args.iterations, batch_size=args.batch_size)
    
    # Save results
    results = {
        'final_mean': final_mean,
        'final_cov': final_cov,
        'history': bam.get_history(),
        'parameters': {
            'iterations': args.iterations,
            'batch_size': args.batch_size
        }
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()