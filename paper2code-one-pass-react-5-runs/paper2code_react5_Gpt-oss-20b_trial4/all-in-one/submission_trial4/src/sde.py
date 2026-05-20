"""
Implementation of the Variance Preserving SDE (VPSDE) used for training
the Simformer.  The forward process is:
    x_t = sqrt(alpha(t)) * x0 + sqrt(1 - alpha(t)) * epsilon
where alpha(t) = exp(-beta * t) with a fixed beta = 1.0.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

# Hyper‑parameters for the VPSDE
BETA = 1.0  # constant beta for simplicity

def alpha(t: jnp.ndarray) -> jnp.ndarray:
    """Alpha(t) = exp(-beta * t)."""
    return jnp.exp(-BETA * t)

def sigma(t: jnp.ndarray) -> jnp.ndarray:
    """Sigma(t) = sqrt(1 - alpha(t))."""
    return jnp.sqrt(1.0 - alpha(t))

def forward_sample(rng: jax.random.PRNGKey, x0: jnp.ndarray,
                   t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample x_t from the forward process given x0 and t.
    Returns:
        x_t: noisy observation
        target_score: (x0 - x_t) / sigma(t)^2
    """
    rng, rng_eps = jax.random.split(rng)
    eps = jax.random.normal(rng_eps, x0.shape)
    a = alpha(t)[:, None]      # (B, 1)
    s = sigma(t)[:, None]      # (B, 1)
    x_t = jnp.sqrt(a) * x0 + s * eps
    target_score = (x0 - x_t) / (s**2)
    return x_t, target_score

def reverse_sde_step(x: jnp.ndarray, score: jnp.ndarray,
                    t: float, dt: float,
                    rng: jax.random.PRNGKey) -> jnp.ndarray:
    """
    One Euler‑Maruyama step of the reverse SDE.
    For VPSDE with beta=1: f = -0.5 * x, g = 1.
    """
    f = -0.5 * x
    g = 1.0
    dx = f - g**2 * score
    noise = jax.random.normal(rng, x.shape)
    return x + dx * dt + noise * jnp.sqrt(dt)