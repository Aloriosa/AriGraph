"""KL divergences between two multivariate Gaussians."""
import numpy as np

def kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p):
    """KL(q || p) for q = N(mu_q, Sigma_q),  p = N(mu_p, Sigma_p)."""
    d = mu_q.shape[0]
    inv_Sigma_p = np.linalg.inv(Sigma_p)
    diff = mu_q - mu_p
    term1 = np.trace(inv_Sigma_p @ Sigma_q)
    term2 = diff.T @ inv_Sigma_p @ diff
    term3 = np.log(np.linalg.det(Sigma_p) / np.linalg.det(Sigma_q))
    return 0.5 * (term1 + term2 - d + term3)

def forward_kl(mu_q, Sigma_q, mu_p, Sigma_p):
    """KL(p || q)."""
    return kl_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)

def reverse_kl(mu_q, Sigma_q, mu_p, Sigma_p):
    """KL(q || p)."""
    return kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p)

def kl_kl_divergence(mu_q, Sigma_q, mu_p, Sigma_p):
    """Score‑based divergence for Gaussian families (eq. 5.7 in the paper)."""
    # Proposition A.7 from the paper
    inv_Sigma_p = np.linalg.inv(Sigma_p)
    term1 = np.trace((np.eye(len(inv_Sigma_p)) - Sigma_q @ inv_Sigma_p) ** 2)
    diff = mu_q - mu_p
    term2 = diff.T @ inv_Sigma_p @ Sigma_q @ inv_Sigma_p @ diff
    return term1 + term2