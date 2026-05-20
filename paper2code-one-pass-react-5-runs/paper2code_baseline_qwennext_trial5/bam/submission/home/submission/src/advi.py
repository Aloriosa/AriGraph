import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
import jax.random as random

class ADVI:
    """
    Implementation of Automatic Differentiation Variational Inference (ADVI)
    as a baseline algorithm.
    
    This implementation follows the standard ADVI algorithm based on ELBO maximization.
    """
    
    def __init__(self, 
                 target_log_prob: Callable,
                 initial_mean: np.ndarray,
                 initial_cov: np.ndarray,
                 batch_size: int = 10,
                 learning_rate: float = 0.01,
                 max_iterations: int = 100):
        """
        Initialize the ADVI algorithm.
        
        Args:
            target_log_prob: Function that takes a state and returns the log probability
                           of the target distribution (unnormalized)
            initial_mean: Initial mean of the variational distribution
            initial_cov: Initial covariance of the variational distribution
            batch_size: Size of the batch to sample at each iteration
            learning_rate: Learning rate for the optimizer
            max_iterations: Maximum number of iterations
        """
        self.target_log_prob = target_log_prob
        self.mu = jnp.array(initial_mean, dtype=jnp.float32)
        self.sigma = jnp.array(initial_cov, dtype=jnp.float32)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.history = {'means': [], 'covariances': [], 'elbos': []}
        self.key = random.PRNGKey(0)
    
    def elbo(self, samples: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Evidence Lower Bound (ELBO).
        
        Args:
            samples: Samples from the variational distribution.
            
        Returns:
            ELBO value.
        """
        # Compute log probability under the target distribution
        log_p = jax.vmap(self.target_log_prob)(samples)
        
        # Compute log probability under the variational distribution
        # For a Gaussian variational distribution
        # log_q = -0.5 * jnp.sum((samples - self.mu) ** 2 / self.sigma, axis=1)
        # log_q -= 0.5 * jnp.log(2 * jnp.pi) * samples.shape[1]
        # log_q -= 0.5 * jnp.log(jnp.prod(self.sigma))
        
        # ELBO = E_q[log p(x, z)] - E_q[log q(z)]
        # For our case, we use the ELBO formula
        # ELBO = E_q[log p(x, z)] - E_q[log q(z)]
        # We can use the reparameterization trick
        # We use the standard ELBO formula for Gaussian variational distribution
        log_q = -0.5 * jnp.sum((samples - self.mu) ** 2 / self.sigma)
        log_q -= 0.5 * jnp.log(2 * jnp.pi) * samples.shape[1]
        log_q -= 0.5 * jnp.log(jnp.prod(self.sigma))
        
        elbo = jnp.mean(log_p) - jnp.mean(log_q)
        
        return elbo
    
    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the ADVI algorithm to optimize the variational distribution.
        
        Returns:
            final_mean: Final mean of the variational distribution
            final_cov: Final covariance of the variational distribution
        """
        for t in range(self.max_iterations):
            # Sample batch of points from current variational distribution
            self.key, key = random.split(self.key)
            samples = random.multivariate_normal(key, self.mu, self.sigma, (self.batch_size,))
        
            # Compute ELBO
        # Compute the gradient of the ELBO
        grad_mu = jax.grad(self.elbo)(samples)
        
        # Update the parameters
        self.mu = self.mu + self.learning_rate * grad_mu
        
        # Update the covariance
        # We use a simple update rule for the covariance
        # In practice, we might use a more sophisticated update rule
        self.sigma = self.sigma + 0.1 * (jnp.cov(samples.T) - self.sigma)
        
        # Store history
        self.history['means'].append(self.mu)
        self.history['covariances'].append(self.sigma)
        self.history['elbos'].append(self.elbo(samples))
        
        if t % 10 == 0:
            print(f"ADVI Iteration {t}: mean={self.mu}, cov={self.sigma}")
        
        return self.mu, self.sigma