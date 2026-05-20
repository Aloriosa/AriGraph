"""
Utility functions for Gaussian target / variational distributions.
All functions assume multivariate Gaussian with mean vector `mu` and
covariance matrix `Sigma` (positive definite).
"""

import numpy as np
from scipy.linalg import inv, sqrtm, det

def mvn_logpdf(z, mu, Sigma):
    """Log pdf of multivariate normal."""
    D = mu.shape[0]
    invS = inv(Sigma)
    diff = z - mu
    return -0.5 * (diff @ invS @ diff + np.log((2 * np.pi) ** D * det(Sigma)))

def mvn_grad_logpdf(z, mu, Sigma):
    """Gradient of log pdf w.r.t. z."""
    invS = inv(Sigma)
    return invS @ (mu - z)

def kl_gaussian(q_mu, q_Sigma, p_mu, p_Sigma):
    """
    KL(q || p) for two Gaussians q=N(q_mu, q_Sigma), p=N(p_mu, p_Sigma).
    """
    inv_p = inv(p_Sigma)
    diff = q_mu - p_mu
    term1 = np.trace(inv_p @ q_Sigma)
    term2 = diff @ inv_p @ diff
    term3 = np.log(det(p_Sigma) / det(q_Sigma))
    D = q_mu.shape[0]
    return 0.5 * (term1 + term2 - D + term3)

def kl_gaussian_reverse(p_mu, p_Sigma, q_mu, q_Sigma):
    """KL(p || q)."""
    return kl_gaussian(q_mu, q_Sigma, p_mu, p_Sigma)

def score_based_divergence(q_mu, q_Sigma, p_mu, p_Sigma):
    """
    Closed‑form score‑based divergence for two Gaussians.
    D(q || p) = tr[(I - Σ Σ_*^{-1})^2] + (μ - μ_*)^T Σ_*^{-1} Σ Σ_*^{-1} (μ - μ_*)
    """
    inv_p = inv(p_Sigma)
    diff = q_mu - p_mu
    term1 = np.trace((np.eye(q_mu.shape[0]) - q_Sigma @ inv_p) @ (np.eye(q_mu.shape[0]) - q_Sigma @ inv_p))
    term2 = diff @ inv_p @ q_Sigma @ inv_p @ diff
    return term1 + term2

def target_gaussian_score(z, mu_target, Sigma_target_inv):
    """
    Gradient of log p(z) for a Gaussian target.
    For p = N(mu_target, Sigma_target), grad log p = Sigma_target_inv @ (mu_target - z)
    """
    return Sigma_target_inv @ (mu_target - z)