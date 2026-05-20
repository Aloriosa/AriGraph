"""
Utility functions for reproducibility and logging.
"""

import numpy as np
import jax
import random

def set_seed(seed: int):
    """
    Set random seeds for numpy, random, and jax.
    Returns a JAX PRNGKey.
    """
    np.random.seed(seed)
    random.seed(seed)
    return jax.random.PRNGKey(seed)