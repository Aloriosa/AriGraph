# utils.py
"""
Utility functions used by the BaM implementation.
"""

import numpy as np
from scipy.linalg import eigh

def rand_spd_matrix(d, seed=None):
    """
    Generate a random symmetric positive‑definite matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T + np.eye(d) * d  # add a multiple of the identity for stability

def gaussian_logpdf(x, mean, cov):
    """
    Evaluate log N(x | mean, cov) for all rows of x.
    x: (n, d)
    mean: (d,)
    cov: (d, d)
    Returns: (n,)
    """
    d = mean.size
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    quad = np.sum(diff @ inv_cov * diff, axis=1)
    log_det = np.linalg.slogdet(cov)[1]
    return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)

def kl_gaussian(mu_q, cov_q, mu_p, cov_p):
    """
    Compute the KL divergence KL(q || p) where
        q ~ N(mu_q, cov_q)
        p ~ N(mu_p, cov_p)
    Returns a scalar.
    """
    d = mu_q.size
    inv_cov_p = np.linalg.inv(cov_p)
    diff = mu_p - mu_q
    term1 = np.trace(inv_cov_p @ cov_q)
    term2 = diff @ inv_cov_p @ diff
    term3 = -np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
    term4 = d
    return 0.5 * (term1 + term2 + term3 - term4)

def sqrtm_psd(A):
    """
    Compute the symmetric positive‑definite square root of a PSD matrix A.
    A must be symmetric.
    """
    w, v = eigh(A)
    # Numerical stability: clip negative eigenvalues to zero
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T