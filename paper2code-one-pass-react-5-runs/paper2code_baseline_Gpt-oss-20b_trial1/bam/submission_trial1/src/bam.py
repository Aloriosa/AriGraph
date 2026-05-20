"""Implementation of the Batch and Match (BaM) algorithm."""
import numpy as np
from .utils import sqrtm, inv_sqrtm

def sample_gaussian(mu, Sigma, n):
    """Draw n samples from N(mu, Sigma)."""
    return np.random.multivariate_normal(mu, Sigma, size=n)

def score_gaussian(z, mu_p, Sigma_p):
    """Score (gradient of log density) for Gaussian target N(mu_p, Sigma_p)."""
    inv = np.linalg.inv(Sigma_p)
    return inv @ (mu_p - z)

def bam_step(mu_t, Sigma_t, B, lambda_t, target_mu, target_Sigma):
    """
    One BaM iteration (batch + match).

    Parameters
    ----------
    mu_t, Sigma_t : current variational parameters
    B             : batch size
    lambda_t      : inverse regularization (learning rate)
    target_mu, target_Sigma : target Gaussian parameters
    """
    # 1. SAMPLE
    z_batch = sample_gaussian(mu_t, Sigma_t, B)  # shape (B, D)

    # 2. SCORE of target at samples
    g_batch = score_gaussian(z_batch, target_mu, target_Sigma)  # (B, D)

    # 3. STATISTICS
    bar_z = np.mean(z_batch, axis=0)
    C = np.cov(z_batch.T, bias=True)  # population covariance (D, D)

    bar_g = np.mean(g_batch, axis=0)
    Gamma = np.cov(g_batch.T, bias=True)

    # 4. COMPUTE U, V
    U = lambda_t * Gamma + (lambda_t / (1 + lambda_t)) * np.outer(bar_g, bar_g)
    V = Sigma_t + lambda_t * C + (lambda_t / (1 + lambda_t)) * np.outer(mu_t - bar_z,
                                                                     mu_t - bar_z)

    # 5. SOLVE QUADRATIC MATRIX EQUATION Σ U Σ + Σ = V
    #    Solution: Σ = 2 V [ I + sqrt(I + 4 U V) ]^{-1}
    #    All matrices are symmetric PD.
    UV = U @ V
    sqrt_term = sqrtm(np.eye(UV.shape[0]) + 4 * UV)   # matrix square root
    inv_term = np.linalg.inv(np.eye(UV.shape[0]) + sqrt_term)
    Sigma_new = 2 * V @ inv_term

    # 6. UPDATE MEAN
    mu_new = (1 / (1 + lambda_t)) * mu_t + \
             (lambda_t / (1 + lambda_t)) * (Sigma_new @ bar_g + bar_z)

    return mu_new, Sigma_new

def bam(
    target_mu,
    target_Sigma,
    mu0,
    Sigma0,
    B=10,
    lambda_t=1.0,
    T=20,
    verbose=True,
    callback=None,
):
    """Run BaM for T iterations.  Optional callback receives (t, mu, Sigma)."""
    mu, Sigma = mu0, Sigma0
    for t in range(T):
        mu, Sigma = bam_step(mu, Sigma, B, lambda_t, target_mu, target_Sigma)
        if callback is not None:
            callback(t, mu, Sigma)
        if verbose:
            print(f"Iteration {t:02d}: mu norm={np.linalg.norm(mu):.4f}")
    return mu, Sigma