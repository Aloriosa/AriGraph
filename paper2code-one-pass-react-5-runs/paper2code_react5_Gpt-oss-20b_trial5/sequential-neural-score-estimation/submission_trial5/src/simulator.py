import numpy as np
import torch

class GaussianLinearSimulator:
    """
    Toy simulator:
        Prior: theta ~ N(0, 0.1 I_d)
        Likelihood: x | theta ~ N(theta, 0.1 I_d)
    Observation dimension equals parameter dimension.
    """
    def __init__(self, dim=10, prior_std=0.1, likelihood_std=0.1, seed=0):
        self.dim = dim
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std
        self.rng = np.random.default_rng(seed)

    def sample_prior(self, n):
        """Sample n parameters from the prior."""
        return self.rng.normal(0, self.prior_std, size=(n, self.dim))

    def simulate(self, theta):
        """
        Simulate data x given theta.
        Args:
            theta: (n, dim) or (dim,)
        Returns:
            x: same shape as theta
        """
        theta = np.asarray(theta)
        shape = theta.shape
        theta = theta.reshape((-1, self.dim))
        noise = self.rng.normal(0, self.likelihood_std, size=theta.shape)
        x = theta + noise
        return x.reshape(shape)

    def analytical_posterior(self, x):
        """
        For Gaussian linear model, the posterior is Gaussian.
        Posterior mean and covariance:
            Cov_post = (1/prior_std^2 + 1/likelihood_std^2)^-1
            Mean_post = Cov_post * (x / likelihood_std^2)
        Returns:
            mean: (dim,)
            cov: (dim, dim)
        """
        inv_prior = 1.0 / (self.prior_std ** 2)
        inv_lik = 1.0 / (self.likelihood_std ** 2)
        cov_post = 1.0 / (inv_prior + inv_lik)
        mean_post = cov_post * (x * inv_lik)
        cov = np.eye(self.dim) * cov_post
        return mean_post, cov