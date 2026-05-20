#!/usr/bin/env python3
"""
Batch and Match (BaM) Algorithm for Black-Box Variational Inference

This implementation reproduces the Batch and Match algorithm from the paper:
"Batch and match: black-box variational inference with a score-based divergence"

The algorithm optimizes a score-based divergence between the variational distribution and the target distribution.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import json
import os
from typing import Tuple, Optional
import argparse

# Set up JAX to use GPU if available
jax.config.update("jax_platforms", "gpu")
jax.config.update("jax_enable_x64", True)

class BatchAndMatch:
    """
    Implementation of the Batch and Match (BaM) algorithm for variational inference.
    
    The algorithm alternates between:
    1. Batch step: Draw a batch of samples from the current variational approximation
    2. Match step: Update the variational parameters using a closed-form proximal update
    """
    
    def __init__(self, dimension: int, lambda_init: float = 1.0):
        """
        Initialize the Batch and Match algorithm.
        
        Args:
            dimension: Dimension of the target distribution
            lambda_init: Initial inverse regularization parameter
        """
        self.dimension = dimension
        self.lambda_init = lambda_init
        self.lambda_t = lambda_init
        
        # Initialize variational parameters
        self.mu = np.zeros(dimension)
        self.Sigma = np.eye(dimension)
        
        # Store history for analysis
        self.history = {
            'mu': [],
            'Sigma': [],
            'lambda_t': [],
            'divergence': []
        }
    
    def target_score(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the score function (gradient of log target density)
        For reproduction, we use a Gaussian target with known parameters.
        
        Args:
            z: Points at which to evaluate the score (batch_size x dimension)
            
        Returns:
            Score vector (batch_size x dimension)
        """
        # For reproduction, we use a Gaussian target distribution
        # Target mean and covariance
        mu_target = np.array([0.5 + 0.1 * i for i in range(self.dimension)])
        Sigma_target = np.eye(self.dimension) * 1.5
        Sigma_target += np.diag(np.random.normal(0, 0.1, self.dimension))
        Sigma_target = np.dot(Sigma_target, Sigma_target.T)
        
        # Score = Σ^{-1} (μ - z)
        Sigma_target_inv = np.linalg.inv(Sigma_target)
        score = np.dot(Sigma_target_inv, (mu_target - z))
        
        return score
    
    def batch_step(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch step: Draw samples from current variational approximation and compute statistics.
        
        Args:
            batch_size: Number of samples to draw
            
        Returns:
            mean_z: Mean of samples
            cov_z: Covariance of samples
            mean_g: Mean of scores
            cov_g: Covariance of scores
        """
        # Sample from current variational approximation
        # z ~ N(mu, Sigma)
        z = np.random.multivariate_normal(self.mu, self.Sigma, size=batch_size)
        
        # Compute scores at samples
        g = self.target_score(z)
        
        # Compute statistics
        mean_z = np.mean(z, axis=0)
        cov_z = np.cov(z.T)
        mean_g = np.mean(g, axis=0)
        cov_g = np.cov(g.T)
        
        return mean_z, cov_z, mean_g, cov_g
    
    def match_step(self, mean_z: np.ndarray, cov_z: np.ndarray, mean_g: np.ndarray, cov_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match step: Update variational parameters using closed-form proximal update.
        
        Args:
            mean_z: Mean of samples
            cov_z: Covariance of samples
            mean_g: Mean of scores
            cov_g: Covariance of scores
            
        Returns:
            new_mu: Updated mean
            new_Sigma: Updated covariance
        """
        # Extract parameters from batch step
        mu_t = self.mu
        Sigma_t = self.Sigma
        
        # Compute matrices U and V for the quadratic matrix equation
        # U = lambda_t * Gamma + (lambda_t / (1 + lambda_t)) * g_bar * g_bar^T
        # V = Sigma_t + lambda_t * C + (lambda_t / (1 + lambda_t)) * (mu_t - z_bar) * (mu_t - z_bar)^T
        
        # For the match step, we use the equations from the paper
        # U = lambda_t * cov_g + (lambda_t / (1 + lambda_t)) * np.outer(mean_g, mean_g)
        U = self.lambda_t * cov_g + (self.lambda_t / (1 + self.lambda_t)) * np.outer(mean_g, mean_g)
        
        # V = Sigma_t + lambda_t * cov_z + (lambda_t / (1 + lambda_t)) * np.outer(mu_t - mean_z, mu_t - mean_z)
        V = Sigma_t + self.lambda_t * cov_z + (self.lambda_t / (1 + self.lambda_t)) * np.outer(mu_t - mean_z, mu_t - mean_z)
        
        # Solve the quadratic matrix equation: Sigma * U * Sigma + Sigma = V
        # Solution: Sigma = 2 * V * (I + (I + 4 * U * V)^{1/2})^{-1}
        
        # Compute the solution for the covariance matrix
        I = np.eye(self.dimension)
        UV = np.dot(U, V)
        # Compute the square root of (I + 4 * U * V)
        # Use eigenvalue decomposition for the matrix square root
        eigenvals, eigenvecs = np.linalg.eigh(I + 4 * UV)
        # Ensure eigenvalues are positive
        eigenvals = np.maximum(eigenvals, 1e-10)
        sqrt_matrix = np.dot(eigenvecs, np.dot(np.diag(np.sqrt(eigenvals)), eigenvecs.T))
        
        # Compute the new covariance matrix
        new_Sigma = 2 * V @ np.linalg.inv(I + sqrt_matrix)
        
        # Compute the new mean
        # mu_{t+1} = (1 / (1 + lambda_t)) * mu_t + (lambda_t / (1 + lambda_t)) * (Sigma_{t+1} * g_bar + z_bar)
        new_mu = (1 / (1 + self.lambda_t)) * mu_t + (self.lambda_t / (1 + self.lambda_t)) * (np.dot(new_Sigma, mean_g) + mean_z)
        
        return new_mu, new_Sigma
    
    def run(self, iterations: int, batch_size: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Batch and Match algorithm for the specified number of iterations.
        
        Args:
            iterations: Number of iterations to run
            batch_size: Batch size for sampling
            verbose: Whether to print progress
            
        Returns:
            final_mu: Final mean parameter
            final_Sigma: Final covariance parameter
        """
        if verbose:
            print(f"Running Batch and Match algorithm with {iterations} iterations and batch size {batch_size}")
        
        # Initialize parameters
        self.mu = np.zeros(self.dimension)
        self.Sigma = np.eye(self.dimension)
        
        # Run iterations
        for t in range(iterations):
            # Batch step
            mean_z, cov_z, mean_g, cov_g = self.batch_step(batch_size)
            
            # Match step
            new_mu, new_Sigma = self.match_step(mean_z, cov_z, mean_g, cov_g)
            
            # Update parameters
            self.mu = new_mu
            self.Sigma = new_Sigma
            
            # Update regularization parameter
            # In the paper, lambda_t is fixed, but we can update it
            self.lambda_t = self.lambda_init / (1 + t / 10)
            
            # Store history
            self.history['mu'].append(self.mu.copy())
            self.history['Sigma'].append(self.Sigma.copy())
            self.history['lambda_t'].append(self.lambda_t)
            
            # Compute divergence (for analysis)
            divergence = self.compute_divergence()
            self.history['divergence'].append(divergence)
            
            # Print progress
            if verbose and (t % 10 == 0 or t == iterations - 1):
                print(f"Iteration {t}: mu={self.mu[:3]}..., Sigma={self.Sigma[0, :3]}...")
        
        return self.mu, self.Sigma
    
    def compute_divergence(self) -> float:
        """
        Compute the score-based divergence between the current variational distribution and the target distribution.
        
        Returns:
            divergence: Score-based divergence value
        """
        # For the target distribution, we use the same Gaussian target as in the paper
        mu_target = np.array([0.5 + 0.1 * i for i in range(self.dimension)])
        Sigma_target = np.eye(self.dimension) * 1.5
        Sigma_target += np.diag(np.random.normal(0, 0.1, self.dimension))
        Sigma_target = np.dot(Sigma_target, Sigma_target.T)
        
        # The score-based divergence is:
        # D(q; p) = E_q[||∇log q - ∇log p||_{Cov(q)}^2]
        # For Gaussian q, this becomes:
        # D(q; p) = tr(Σ_p Σ_q^{-1}) + tr(Σ_p^{-1} Σ_q) + ||μ_q - μ_p - Σ_q g||_{Σ_q^{-1}}^2
        # where g = Σ_p^{-1} (μ_p - μ_q)
        
        # For our target, we have:
        Sigma_p = Sigma_target
        Sigma_q = self.Sigma
        mu_p = mu_target
        mu_q = self.mu
        
        # Compute g = Σ_p^{-1} (μ_p - μ_q)
        Sigma_p_inv = np.linalg.inv(Sigma_p)
        g = np.dot(Sigma_p_inv, (Sigma_p - mu_q))
        
        # Compute the divergence
        term1 = np.trace(np.dot(Sigma_p, np.linalg.inv(Sigma_q)))
        term2 = np.trace(np.dot(np.linalg.inv(Sigma_p), Sigma_q))
        term3 = np.linalg.norm(np.dot(Sigma_q, g) + mu_q - mu_q) ** 2
        divergence = term1 + term2 + term3
        
        return divergence

def main():
    """
    Main function to run the reproduction script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Reproduce Batch and Match algorithm')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--dimension', type=int, default=10, help='Dimension of target distribution')
    parser.add_argument('--output', type=str, default='output/results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run the Batch and Match algorithm
    print("Initializing Batch and Match algorithm...")
    bam = BatchAndMatch(dimension=args.dimension)
    
    print("Running Batch and Match algorithm...")
    final_mu, final_Sigma = bam.run(iterations=args.iterations, batch_size=args.batch_size, verbose=True)
    
    # Save results
    results = {
        'final_mu': final_mu.tolist(),
        'final_Sigma': final_Sigma.tolist(),
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()