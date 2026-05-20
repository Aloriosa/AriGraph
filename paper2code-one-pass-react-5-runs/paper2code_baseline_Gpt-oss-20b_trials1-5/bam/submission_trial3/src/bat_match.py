# src/bat_match.py
import jax
import jax.numpy as jnp
from .utils import mvn_score, sqrtm

def bam_step(mu, Sigma, key, B, lambda_reg, target_score):
    """One BaM iteration."""
    # Sample batch
    z = jax.random.multivariate_normal(key, mu, Sigma, shape=(B,))
    g = target_score(z)                     # (B, D)
    z_bar = jnp.mean(z, axis=0)             # (D,)
    g_bar = jnp.mean(g, axis=0)             # (D,)

    # Empirical covariances
    C = jnp.cov(z, rowvar=False, bias=True)          # (D, D)
    Gamma = jnp.cov(g, rowvar=False, bias=True)      # (D, D)

    # Matrices for the quadratic equation
    U = lambda_reg * Gamma + (lambda_reg / (1 + lambda_reg)) * jnp.outer(g_bar, g_bar)
    V = Sigma + lambda_reg * C + (lambda_reg / (1 + lambda_reg)) * jnp.outer(mu - z_bar, mu - z_bar)

    # Solve Σ_{t+1} U Σ_{t+1} + Σ_{t+1} = V
    I = jnp.eye(Sigma.shape[0])
    sqrt_term = sqrtm(I + 4 * U @ V)
    Sigma_new = 2 * V @ jnp.linalg.inv(I + sqrt_term)

    # Update mean
    mu_new = (1 / (1 + lambda_reg)) * mu + (lambda_reg / (1 + lambda_reg)) * (Sigma_new @ g_bar + z_bar)

    return mu_new, Sigma_new

def bam(mu0, Sigma0, key, B, lambda_reg, target_score, T):
    """Run BaM for T iterations."""
    mu, Sigma = mu0, Sigma0
    for t in range(T):
        key, subkey = jax.random.split(key)
        mu, Sigma = bam_step(mu, Sigma, subkey, B, lambda_reg, target_score)
    return mu, Sigma