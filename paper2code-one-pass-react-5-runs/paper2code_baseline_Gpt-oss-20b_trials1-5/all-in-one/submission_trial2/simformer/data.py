"""
Utility functions to generate the toy Two‑Moons dataset used in the benchmark.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def generate_two_moons(n_samples: int, random_state: int = 42):
    """
    Generate pairs (theta, x) where
        theta ∈ ℝ² is uniform in [-1, 1]²
        x   ∈ ℝ² follows the Two‑Moons distribution conditional on theta
    For the toy benchmark we ignore a true conditional dependence and simply
    sample x from the standard Two‑Moons distribution.
    """
    rng = np.random.default_rng(random_state)
    theta = rng.uniform(-1.0, 1.0, size=(n_samples, 2))

    # Two‑Moons data with a small amount of noise
    x, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)

    # Standardise for nicer training dynamics
    scaler_theta = StandardScaler()
    scaler_x = StandardScaler()
    theta = scaler_theta.fit_transform(theta)
    x = scaler_x.fit_transform(x)

    data = np.concatenate([theta, x], axis=1).astype(np.float32)
    return data