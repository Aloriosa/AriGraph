"""
Gaussian‑Linear task: θ ~ N(0,0.1² I),  x ~ N(θ, 0.1² I)
"""

import numpy as np


class GaussianLinearSim:
    def __init__(self, param_dim=10, data_dim=10, num_samples=100_000):
        self.param_dim = param_dim
        self.data_dim = data_dim
        self.num_samples = num_samples

    def sample_batch(self, batch_size, theta=None):
        """
        Sample a batch of (θ, x). If theta is provided, use it as the true parameters
        and generate x accordingly.
        """
        if theta is None:
            theta = np.random.randn(batch_size, self.param_dim) * 0.1
        else:
            theta = np.asarray(theta, dtype=np.float32)
            if theta.ndim == 1:
                theta = theta[np.newaxis, :]
        x = theta + np.random.randn(batch_size, self.data_dim) * 0.1
        return theta.astype(np.float32), x.astype(np.float32)