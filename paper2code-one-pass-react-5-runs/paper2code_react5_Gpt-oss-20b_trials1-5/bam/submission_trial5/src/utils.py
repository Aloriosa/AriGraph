import numpy as np
from scipy.linalg import sqrtm, inv, cholesky

def log_pdf_gaussian(z, mu, Sigma, log_det_Sigma=None):
    """Log probability of a multivariate Gaussian (unnormalised)."""
    D = z.shape[-1]
    diff = z - mu
    if log_det_Sigma is None:
        # Compute log det via Cholesky
        L = cholesky(Sigma, lower=True)
        log_det_Sigma = 2 * np.sum(np.log(np.diag(L)))
    inv_Sigma = inv(Sigma)
    quad = np.einsum("...i,ij,...j->...", diff, inv_Sigma, diff)
    return -0.5 * (quad + log_det_Sigma + D * np.log(2 * np.pi))

def kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p):
    """KL(q||p) for two Gaussians."""
    inv_Sigma_p = inv(Sigma_p)
    term1 = np.trace(inv_Sigma_p @ Sigma_q)
    diff = mu_p - mu_q
    term2 = diff @ inv_Sigma_p @ diff
    term3 = np.log(np.linalg.det(Sigma_p)) - np.log(np.linalg.det(Sigma_q))
    D = mu_q.shape[0]
    return 0.5 * (term1 + term2 - D + term3)

def solve_quadratic_matrix(U, V):
    """
    Solve X U X + X = V for X (positive definite).
    Returns: X = 2 V (I + sqrt(I + 4 U V))^{-1}
    """
    # Ensure symmetric
    U = (U + U.T) / 2.0
    V = (V + V.T) / 2.0
    # Compute sqrtm of I + 4 U V
    M = np.eye(U.shape[0]) + 4 * U @ V
    sqrtM = sqrtm(M)
    # Inverse
    inv_term = inv(np.eye(U.shape[0]) + sqrtM)
    X = 2 * V @ inv_term
    # Symmetrise
    X = (X + X.T) / 2.0
    return X