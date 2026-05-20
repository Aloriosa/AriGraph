import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple
import jax.random as random

class BatchAndMatch:
    """
    Implementation of Batch and Match (BaM) algorithm for black-box variational inference
    with a score-based divergence.
    
    This implementation follows the algorithm described in the paper with closed-form updates
    for Gaussian variational families.
    """
    
    def __init__(self, 
                 target_log_prob: Callable,
                 initial_mean: np.ndarray,
                 initial_cov: np.ndarray,
                 batch_size: int = 10,
                 regularization: float = 1.0,
                 max_iterations: int = 100):
        """
        Initialize the Batch and Match algorithm.
        
        Args:
            target_log_prob: Function that takes a state and returns the log probability
                           of the target distribution (unnormalized)
            initial_mean: Initial mean of the variational distribution
            initial_cov: Initial covariance of the variational distribution
            batch_size: Size of the batch to sample at each iteration
            regularization: Regularization parameter lambda
            max_iterations: Maximum number of iterations
        """
        self.target_log_prob = target_log_prob
        self.mu = jnp.array(initial_mean, dtype=jnp.float32)
        self.sigma = jnp.array(initial_cov, dtype=jnp.float32)
        self.batch_size = batch_size
        self.lambda_param = regularization
        self.max_iterations = max_iterations
        self.history = {'means': [], 'covariances': [], 'scores': []}
        
        # Initialize the random key
        self.key = random.PRNGKey(0)
        
    def score_function(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the score function (gradient of the log target density).
        
        Args:
            z: Points at which to evaluate the score.
            
        Returns:
            Score values at the given points.
        """
        # Use JAX's automatic differentiation to compute the gradient
        return jax.grad(self.target_log_prob)(z)
    
    def batch_step(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Perform the "batch" step of the algorithm.
        
        This step samples a batch of points from the current variational distribution
        and computes the scores at those points.
        
        Args:
            key: JAX random key for sampling.
            
        Returns:
            mean_z: Mean of the sampled points
            cov_z: Covariance of the sampled points
            mean_g: Mean of the scores at the sampled points
            cov_g: Covariance of the scores at the sampled points
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
        
        return mean_z, cov_z, mean_g, cov_g
    
    def match_step(self, mean_z: jnp.ndarray, cov_z: jnp.ndarray, mean_g: jnp.ndarray, cov_g: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform the "match" step of the algorithm.
        
        This step updates the variational distribution by minimizing the regularized objective.
        
        Args:
            mean_z: Mean of sampled points
            cov_z: Covariance of sampled points
            mean_g: Mean of scores at sampled points
            cov_g: Covariance of scores at sampled points
            
        Returns:
            new_mean: Updated mean of variational distribution
            new_cov: Updated covariance of variational distribution
        """
        # Compute matrices U and V for the quadratic matrix equation
        # U = lambda * cov_g + (lambda / (1 + lambda)) * mean_g * mean_g.T
        U = (self.lambda_param * cov_g) + ((self.lambda_param / (1 + self.lambda_param)) * jnp.outer(mean_g, mean_g))
        
        # V = self.sigma + lambda * cov_z + (lambda / (1 + lambda)) * (self.mu - mean_z) * (self.mu - mean_z).T
        V = self.sigma + (self.lambda_param * cov_z) + ((self.lambda_param / (1 + self.lambda_param)) * jnp.outer(self.mu - mean_z, self.mu - mean_z))
        
        # Solve the quadratic matrix equation: sigma * U * sigma + sigma = V
        # The solution is given by: sigma = 2 * V * (I + (I + 4 * U * V)^(-1/2))
        # First compute the matrix I + 4 * U * V
        IV = jnp.eye(self.mu.shape[0]) + 4 * jnp.matmul(U, V)
        
        # Compute the matrix square root of I + 4 * U * V
        # We use eigenvalue decomposition to compute the matrix square root
        eigvals, eigvecs = jnp.linalg.eigh(IV)
        
        # Compute the square root of the eigenvalues
        sqrt_eigvals = jnp.sqrt(eigvals)
        
        # Reconstruct the matrix square root
        sqrt_IV = jnp.matmul(eigvecs, jnp.diag(sqrt_eigvals))
        sqrt_IV = jnp.matmul(sqrt_IV, jnp.linalg.inv(eigvecs))
        
        # Compute the solution
        new_cov = 2 * V * jnp.linalg.inv(jnp.eye(self.mu.shape[0]) + sqrt_IV)
        
        # Update the mean
        new_mean = (1 / (1 + self.lambda_param)) * self.mu + (self.lambda_param / (1 + self.lambda_param)) * (jnp.matmul(self.sigma, mean_g) + mean_z)
        
        return new_mean, new_cov
    
    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Batch and Match algorithm to optimize the variational distribution.
        
        Returns:
            final_mean: Final mean of the variational distribution
            final_cov: Final covariance of the variational distribution
        """
        for t in range(self.max_iterations):
            # Batch step
            self.key, key = random.split(self.key)
        # Batch step
        mean_z, cov_z, mean_g, cov_g = self.batch_step(key)
        
        # Match step
        new_mean, new_cov = self.match_step(mean_z, cov_z, mean_g, cov_g)
        
        # Update variational parameters
        self.mu = new_mean
        self.sigma = new_cov
        
        # Store history
        self.history['means'].append(self.mu)
        self.history['covariances'].append(self.sigma)
        
        # Print progress
        if t % 10 == 0:
            print(f"Iteration {t}: mean={self.mu}, cov={self.sigma}")
        
        return self.mu, self.sigma