import numpy as np
from .utils import kl_gaussian

class GSM:
    """
    Gaussian Score Matching (limiting case of BaM for λ→∞, B=1).
    """

    def __init__(self, dim, mu0, Sigma0, T=200):
        self.D = dim
        self.mu = mu0.copy()
        self.Sigma = Sigma0.copy()
        self.T = T

    def step(self, score_func):
        # Sample single point
        z = self.mu + np.random.multivariate_normal(np.zeros(self.D), self.Sigma)
        g = score_func(z)  # shape (D,)

        # Solve for rho: rho (1+rho) = g^T Σ g + ((μ - z)^T g)^2
        a = g @ self.Sigma @ g
        b = ((self.mu - z) @ g) ** 2
        # Solve quadratic: rho^2 + rho - (a + b) = 0
        rho = (-1 + np.sqrt(1 + 4 * (a + b))) / 2.0

        # Update mean
        denom = 1 + rho + (self.mu - z) @ g
        term = (self.Sigma @ g) - self.mu + z
        delta_mu = (1 / (1 + rho)) * (np.eye(self.D) - np.outer(self.mu - z, g) / denom) @ term

        mu_new = self.mu + delta_mu

        # Update covariance
        delta_Sigma = np.outer(self.mu - z, self.mu - z) - np.outer(mu_new - z, mu_new - z)

        self.mu = mu_new
        self.Sigma = self.Sigma + delta_Sigma

    def run(self, score_func, callback=None):
        for _ in range(self.T):
            self.step(score_func)
            if callback is not None:
                callback(self.mu, self.Sigma)