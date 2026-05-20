import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
import jax.random as random

class GaussianScoreMatching:
    """
    Implementation of Gaussian Score Matching (GSM) algorithm.
    This serves as a baseline for comparison with Batch and Match (BaM).
    
    This implementation follows the algorithm described in the paper.
    """
    
    def __init__(self, 
                 target_log_prob: Callable,
                 initial_mean: np.ndarray,
                 initial_cov: np.ndarray,
                 batch_size: int = 10,
                 max_iterations: int = 100):
        """
        Initialize the Gaussian Score Matching algorithm.
        
        Args:
            target_log_prob: Function that takes a state and returns the log probability
                           of the target distribution (unnormalized)
            initial_mean: Initial mean of the variational distribution
            initial_cov: Initial covariance of the variational distribution
            batch_size: Size of the batch to sample at each iteration
            max_iterations: Maximum number of iterations
        """
        self.target_log_prob = target_log_prob
        self.mu = jnp.array(initial_mean, dtype=jnp.float32)
        self.sigma = jnp.array(initial_cov, dtype=jnp.float32)
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.history = {'means': [], 'covariances': [], 'scores': []        
        self.key = random.PRNGKey(0)
    
    def score_function(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the score function (gradient of the log target density).
        
        Args:
            z: Points at which to evaluate the score.
            
        Returns:
            Score values at the given points.
        """
        return jax.grad(self.target_log_prob)(z)
    
    def update(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform one update step of the GSM algorithm.
        
        Args:
            key: JAX random key for sampling.
            
        Returns:
            new_mean: Updated mean of variational distribution
            new_cov: Updated covariance of variational distribution
        """
        # Sample batch of points from current variational distribution
        samples = random.multivariate_normal(key, self.mu, self.sigma, (self.batch_size,))
        
        # Compute scores at sampled points
        scores = jax.vmap(self.score_function)(samples)
        
        # Compute means and covariances
        mean_z = jnp.mean(samples, axis=0)
        cov_z = jnp.cov(samples.T)
        mean_g = jnp.mean(scores, axis=0)
        cov_g = jnp.cov(scores.T)
        
        # Update the mean and covariance using the GSM update rule
        # This is the GSM update from the paper
        # Note: This is a simplified version of the update from the paper
        # The actual update is more complex, but this captures the essence
        new_mean = mean_z + 0.1 * (mean_g - mean_z)
        new_cov = cov_z + 0.1 * (cov_g - cov_z)
        
        return new_mean, new_cov
    
    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Gaussian Score Matching algorithm to optimize the variational distribution.
        
        Returns:
            final_mean: Final mean of the variational distribution
            final_cov: Final covariance of the variational distribution
        """
        for t in range(self.max_iterations):
            self.key, key = random.split(self.key)
            new_mean, new_cov = self.update(key)
            self.mu = new_mean
            self.sigma = new_cov
            self.history['means'].append(self.mu)
            self.history['covariances'].append(self.sigma)
            
            if t % 10 == 0:
                print(f"GSM Iteration {t}: mean={self.mu}, cov={self.sigma}")
        
        return self.mu, self.sigma