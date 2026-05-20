"""
Two‑Moons simulator and prior sampler used in the paper.
"""

import numpy as np
from typing import Tuple


# --------------------------------------------------------------------------- #
# Prior sampler
# --------------------------------------------------------------------------- #
def sample_prior_uniform(n: int) -> np.ndarray:
    """
    Sample parameters θ from the uniform prior p(θ) = U(−1, 1)^2.
    Returns an array of shape (n, 2).
    """
    return np.random.uniform(low=-1.0, high=1.0, size=(n, 2)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def simulate_two_moons(theta: np.ndarray) -> np.ndarray:
    """
    Forward simulator for the Two‑Moons benchmark.

    Parameters
    ----------
    theta : np.ndarray, shape (n, 2)
        Parameter vector [θ1, θ2] for each simulation.

    Returns
    -------
    x : np.ndarray, shape (n, 2)
        Observed data drawn from the simulator.
    """
    n = theta.shape[0]
    # α ∼ Uniform(−π/2, π/2)
    alpha = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=n)
    # r ∼ N(0.1, 0.01²)
    r = np.random.normal(loc=0.1, scale=0.01, size=n)

    base = np.zeros((n, 2), dtype=np.float32)
    base[:, 0] = r * np.cos(alpha) + 0.25
    base[:, 1] = r * np.sin(alpha)

    shift = np.zeros((n, 2), dtype=np.float32)
    theta1 = theta[:, 0]
    theta2 = theta[:, 1]
    shift[:, 0] = -(np.abs(theta1 + theta2) / np.sqrt(2.0))
    shift[:, 1] = (-theta1 + theta2) / np.sqrt(2.0)

    return (base + shift).astype(np.float32)


# --------------------------------------------------------------------------- #
# Observed data (fixed for reproducibility)
# --------------------------------------------------------------------------- #
X_OBS = np.array([0.5, 0.5], dtype=np.float32)