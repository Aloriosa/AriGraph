import numpy as np
from scipy.linalg import sqrtm, inv

def bam_step(mu, Sigma, batch_samples, batch_scores, lam):
    """
    One BaM iteration: batch + match.
    Parameters
    ----------
    mu : np.ndarray shape (D,)
        Current variational mean.
    Sigma : np.ndarray shape (D, D)
        Current variational covariance (positive definite).
    batch_samples : np.ndarray shape (B, D)
        Samples z_b ~ N(mu, Sigma).
    batch_scores : np.ndarray shape (B, D)
        Scores s(z_b) = ∇ log p(z_b) for the target.
    lam : float
        Regularisation parameter λ_t (>0).

    Returns
    -------
    mu_new, Sigma_new : updated parameters
    """
    B, D = batch_samples.shape
    # statistics
    z_bar = batch_samples.mean(axis=0)
    C = ((batch_samples - z_bar).T @ (batch_samples - z_bar)) / B
    g_bar = batch_scores.mean(axis=0)
    Gamma = ((batch_scores - g_bar).T @ (batch_scores - g_bar)) / B

    # matrices U and V
    U = lam * Gamma + (lam / (1 + lam)) * np.outer(g_bar, g_bar)
    V = Sigma + lam * C + (lam / (1 + lam)) * np.outer(mu - z_bar, mu - z_bar)

    # update covariance via quadratic matrix equation
    # Σ_{t+1} = 2 V (I + (I + 4 U V)^{1/2})^{-1}
    M = np.eye(D) + 4 * U @ V
    # symmetric square root
    sqrtM = sqrtm(M, disp=False)
    inv_term = inv(np.eye(D) + sqrtM)
    Sigma_new = 2 * V @ inv_term

    # update mean
    mu_new = (1 / (1 + lam)) * mu + (lam / (1 + lam)) * (Sigma_new @ g_bar + z_bar)

    return mu_new, Sigma_new