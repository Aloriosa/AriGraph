import numpy as np
from numpy.linalg import slogdet, inv, det, trace

def kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p):
    """
    Compute KL(q || p) where q and p are multivariate Gaussian.
    """
    d = len(mu_q)
    inv_Sigma_p = inv(Sigma_p)
    diff = mu_p - mu_q
    term1 = np.trace(inv_Sigma_p @ Sigma_q)
    term2 = diff.T @ inv_Sigma_p @ diff
    term3 = np.log(det(Sigma_p) / det(Sigma_q))
    return 0.5 * (term1 + term2 - d + term3)

def kl_gaussian_reverse(mu_q, Sigma_q, mu_p, Sigma_p):
    """
    Compute KL(p || q) where p and q are multivariate Gaussian.
    """
    d = len(mu_q)
    inv_Sigma_q = inv(Sigma_q)
    diff = mu_q - mu_p
    term1 = np.trace(inv_Sigma_q @ Sigma_p)
    term2 = diff.T @ inv_Sigma_q @ diff
    term3 = np.log(det(Sigma_q) / det(Sigma_p))
    return 0.5 * (term1 + term2 - d + term3)

def sqrtm_sym(M):
    """
    Compute the symmetric square root of a symmetric positive definite matrix M.
    """
    w, v = np.linalg.eigh(M)
    # Numerical safety: keep only non‑negative eigenvalues
    w = np.maximum(w, 0)
    return v @ np.diag(np.sqrt(w)) @ v.T