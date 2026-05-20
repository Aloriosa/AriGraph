#!/usr/bin/env python3
"""
BaM (Batch and Match) algorithm: a black‑box variational inference method
for Gaussian variational families with full covariance.

The script runs a single experiment on a synthetic Gaussian target and
saves the final variational parameters and KL divergences to stdout
(and to results.txt when called from reproduce.sh).
"""

import argparse
import numpy as np
from utils import rand_spd_matrix, gaussian_logpdf, kl_gaussian, sqrtm_psd

def sample_gaussian(mean, cov, n, seed=None):
    """Draw n samples from N(mean, cov)."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n)

def baM_iteration(mu, Sigma, mu_star, Sigma_star, B, lam, seed=None):
    """
    Perform one BaM iteration.

    Parameters
    ----------
    mu : np.ndarray, shape (D,)
        Current variational mean.
    Sigma : np.ndarray, shape (D, D)
        Current variational covariance.
    mu_star : np.ndarray, shape (D,)
        True target mean.
    Sigma_star : np.ndarray, shape (D, D)
        True target covariance.
    B : int
        Batch size.
    lam : float
        Inverse regularization parameter λ > 0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    mu_next, Sigma_next : np.ndarray
        Updated variational parameters.
    """
    d = mu.size
    # 1. Sample batch from current q
    z_batch = sample_gaussian(mu, Sigma, B, seed=seed)

    # 2. Compute target scores: ∇log p(z) = Σ*^{-1} (μ* - z)
    inv_Sigma_star = np.linalg.inv(Sigma_star)
    g_batch = (mu_star - z_batch) @ inv_Sigma_star.T  # shape (B, d)

    # 3. Compute statistics
    bar_z = np.mean(z_batch, axis=0)
    C = np.cov(z_batch, rowvar=False, bias=True) * B  # unbiased cov * B -> sum of outer products
    C /= B  # now covariance
    bar_g = np.mean(g_batch, axis=0)
    Gamma = np.cov(g_batch, rowvar=False, bias=True) * B
    Gamma /= B

    # 4. Compute U and V
    U = lam * Gamma + (lam / (1 + lam)) * np.outer(bar_g, bar_g)
    V = Sigma + lam * C + (lam / (1 + lam)) * np.outer(mu - bar_z, mu - bar_z)

    # 5. Update covariance Σ_{t+1}
    # Solve Σ U Σ + Σ = V  →  Σ = 2 V (I + sqrt(I + 4 U V))^{-1}
    UV = U @ V
    M = np.eye(d) + 4 * UV
    sqrtM = sqrtm_psd(M)
    inv_term = np.linalg.inv(np.eye(d) + sqrtM)
    Sigma_next = 2 * V @ inv_term

    # 6. Update mean μ_{t+1}
    mu_next = (1 / (1 + lam)) * mu + (lam / (1 + lam)) * (Sigma_next @ bar_g + bar_z)

    return mu_next, Sigma_next

def run_experiment(dim=10, iterations=200, B=50, lam=1.0, seed=42):
    """
    Run BaM on a synthetic Gaussian target.

    Parameters
    ----------
    dim : int
        Dimensionality of the target.
    iterations : int
        Number of BaM iterations.
    B : int
        Batch size.
    lam : float
        Inverse regularization λ.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary containing final variational parameters and KL divergences.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate target Gaussian p ~ N(μ*, Σ*)
    mu_star = np.zeros(dim)
    Sigma_star = rand_spd_matrix(dim, seed=seed)

    # 2. Initialize variational parameters
    mu = rng.uniform(0, 0.1, size=dim)
    Sigma = np.eye(dim)

    # 3. Run iterations
    for t in range(iterations):
        mu, Sigma = baM_iteration(mu, Sigma, mu_star, Sigma_star, B, lam, seed=seed + t)

    # 4. Compute KL divergences
    kl_q_p = kl_gaussian(mu, Sigma, mu_star, Sigma_star)  # KL(q||p)
    kl_p_q = kl_gaussian(mu_star, Sigma_star, mu, Sigma)  # KL(p||q)

    # 5. Monte Carlo estimate of forward KL (E_p[log p - log q])
    z_mc = sample_gaussian(mu_star, Sigma_star, 10000, seed=seed + 999)
    logp = gaussian_logpdf(z_mc, mu_star, Sigma_star)
    logq = gaussian_logpdf(z_mc, mu, Sigma)
    forward_KL_mc = np.mean(logp - logq)

    return {
        "mu_star": mu_star,
        "Sigma_star": Sigma_star,
        "mu_q": mu,
        "Sigma_q": Sigma,
        "kl_q_p": kl_q_p,
        "kl_p_q": kl_p_q,
        "forward_KL_mc": forward_KL_mc,
        "iterations": iterations,
        "batch": B,
        "lambda": lam,
    }

def print_results(results):
    """Pretty‑print the experiment results."""
    print("=== BaM Experiment Results ===")
    print(f"Dimension               : {results['mu_star'].size}")
    print(f"Iterations              : {results['iterations']}")
    print(f"Batch size (B)          : {results['batch']}")
    print(f"Inverse regularization λ: {results['lambda']:.4f}")
    print()
    print("Target mean μ*:")
    print(results['mu_star'])
    print()
    print("Target covariance Σ*:")
    print(results['Sigma_star'])
    print()
    print("Variational mean μ_q:")
    print(results['mu_q'])
    print()
    print("Variational covariance Σ_q:")
    print(results['Sigma_q'])
    print()
    print(f"KL(q || p) : {results['kl_q_p']:.6f}")
    print(f"KL(p || q) : {results['kl_p_q']:.6f}")
    print(f"Forward KL (MC) : {results['forward_KL_mc']:.6f}")
    print("==============================")

def main():
    parser = argparse.ArgumentParser(description="BaM demo on a synthetic Gaussian target")
    parser.add_argument("--dim", type=int, default=10, help="Dimensionality of Gaussian target")
    parser.add_argument("--iterations", type=int, default=200, help="Number of BaM iterations")
    parser.add_argument("--batch", type=int, default=50, help="Batch size B")
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0, help="Inverse regularization λ")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = run_experiment(dim=args.dim,
                             iterations=args.iterations,
                             B=args.batch,
                             lam=args.lam,
                             seed=args.seed)
    print_results(results)

if __name__ == "__main__":
    main()