"""
Two‑Moons task (adapted from sklearn.datasets.make_moons).
θ ∈ [-1, 1]^2
x ∈ R² is generated from θ via a deterministic mapping plus noise.
"""

import numpy as np
from sklearn.datasets import make_moons


class TwoMoonsSim:
    def __init__(self, param_dim=2, data_dim=2, num_samples=100_000):
        self.param_dim = param_dim
        self.data_dim = data_dim
        self.num_samples = num_samples

    def _theta_to_x(self, theta):
        """
        Map parameters θ to mean of x distribution.
        """
        # θ is [batch,2]
        r = np.exp(-np.sum(theta**2, axis=1))  # a simple radial effect
        alpha = theta[:, 0]  # use first component
        # Base two‑moons
        base, _ = make_moons(n_samples=theta.shape[0], noise=0.0)
        # Shift by r
        mean = base + r[:, None] * np.array([np.cos(alpha), np.sin(alpha)])
        return mean

    def sample_batch(self, batch_size, theta=None):
        if theta is None:
            theta = np.random.uniform(-1, 1, size=(batch_size, self.param_dim))
        else:
            theta = np.asarray(theta, dtype=np.float32)
            if theta.ndim == 1:
                theta = theta[np.newaxis, :]
        mean = self._theta_to_x(theta)
        x = mean + np.random.randn(batch_size, self.data_dim) * 0.1
        return theta.astype(np.float32), x.astype(np.float32)