# src/utils.py
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

def mvn_logpdf(z, mu, Sigma):
    """Log‑density of a multivariate normal."""
    D = mu.shape[0]
    invS = jnp.linalg.inv(Sigma)
    logdet = jnp.linalg.slogdet(Sigma)[1]
    diff = z - mu
    return -0.5 * (D * jnp.log(2 * jnp.pi) + logdet + diff @ invS @ diff)

def mvn_score(z, mu, Sigma):
    """Gradient of log‑pdf of a multivariate normal."""
    invS = jnp.linalg.inv(Sigma)
    return -invS @ (z - mu)

def kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p):
    """KL(q||p) for two Gaussians."""
    invSp = jnp.linalg.inv(Sigma_p)
    term1 = jnp.trace(invSp @ Sigma_q)
    diff = mu_p - mu_q
    term2 = diff @ invSp @ diff
    term3 = jnp.log(jnp.linalg.det(Sigma_p) / jnp.linalg.det(Sigma_q))
    D = mu_q.shape[0]
    return 0.5 * (term1 + term2 - D + term3)

def sqrtm(A):
    """Principal square root of a symmetric positive‑definite matrix."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T