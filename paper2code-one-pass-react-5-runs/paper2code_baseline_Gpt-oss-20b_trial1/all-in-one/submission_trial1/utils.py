"""
Utility functions for the Two Moons simulation and data handling.
"""

import numpy as np

def two_moons_simulator(theta, seed=None):
    """
    Simulate a single observation from the Two Moons model.

    Parameters
    ----------
    theta : array-like, shape (2,)
        Parameter vector sampled from U(-1, 1)^2.

    Returns
    -------
    x : ndarray, shape (2,)
        Observed data.
    """
    # Unpack parameters
    theta1, theta2 = theta

    # Generate radius and angle
    r = np.random.normal(0.1, 0.012)
    alpha = np.random.uniform(-np.pi / 2, np.pi / 2)

    # Compute base moon coordinates
    base = np.array([r * np.cos(alpha) + 0.25,
                     r * np.sin(alpha)])

    # Compute shift due to theta
    shift = np.array([-(abs(theta1 + theta2)) / np.sqrt(2),
                      (-theta1 + theta2) / np.sqrt(2)])

    x = base + shift
    return x

def generate_simulation_batch(n_samples, rng=None):
    """
    Generate a batch of (theta, x) pairs for training.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    rng : np.random.Generator or None
        Optional NumPy random generator.

    Returns
    -------
    thetas : ndarray, shape (n_samples, 2)
        Parameter samples.
    xs : ndarray, shape (n_samples, 2)
        Corresponding observations.
    """
    rng = rng or np.random.default_rng()
    thetas = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    xs = np.array([two_moons_simulator(t, seed=rng.integers(0, 1e9)) for t in thetas])
    return thetas, xs