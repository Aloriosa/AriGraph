"""
Simulator implementations for the Simformer toy experiments.
Currently includes the Two‑Moons simulator used in the paper.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def sample_prior_two_moons(key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
    """
    Sample prior parameters θ ~ Uniform(-1, 1)^2.
    """
    return jax.random.uniform(key, (batch_size, 2), minval=-1.0, maxval=1.0)

def sample_data_two_moons(key: jax.random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
    """
    Simulate observations x given parameters θ using the two‑moons distribution.
    theta: shape (batch, 2)
    Returns x: shape (batch, 2)
    """
    rng1, rng2 = jax.random.split(key)
    # Sample random angle α ~ Uniform(-π/2, π/2)
    alpha = jax.random.uniform(rng1, theta.shape[0], minval=-jnp.pi/2, maxval=jnp.pi/2)
    # Sample radial noise r ~ Normal(0.1, 0.012)
    r = jax.random.normal(rng2, theta.shape[0]) * 0.012 + 0.1

    # Compute deterministic part of the two moons
    x_det = jnp.stack([r * jnp.cos(alpha) + 0.25, r * jnp.sin(alpha)], axis=1)

    # Compute data shift based on θ
    shift = jnp.stack(
        [
            -jnp.abs(theta[:, 0] + theta[:, 1]) / jnp.sqrt(2),
            (-theta[:, 0] + theta[:, 1]) / jnp.sqrt(2),
        ],
        axis=1,
    )

    x = x_det + shift
    return x

def batch_simulate_two_moons(key: jax.random.PRNGKey, batch_size: int
                            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a batch of (θ, x) pairs from the two‑moons simulator.
    """
    key_theta, key_data = jax.random.split(key)
    theta = sample_prior_two_moons(key_theta, batch_size)
    x = sample_data_two_moons(key_data, theta)
    return theta, x