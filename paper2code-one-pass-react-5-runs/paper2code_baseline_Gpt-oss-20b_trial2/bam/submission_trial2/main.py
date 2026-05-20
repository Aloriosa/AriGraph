#!/usr/bin/env python3
"""
Example implementation of the Batch and Match (BaM) algorithm.
The script generates a synthetic Gaussian target, runs BaM with a
small batch size, and reports the final KL divergence.
"""

import argparse
import time
import numpy as np
from scipy.linalg import sqrtm, solve_sylvester
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def random_spd_matrix(dim, seed=None):
    """Generate a random positive‑definite matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    return A @ A.T + dim * np.eye(dim)  # shift to ensure PD

def kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p):
    """Analytic KL divergence D(q||p) for two Gaussians."""
    dim = mu_q.shape[0]
    inv_Sigma_p = np.linalg.inv(Sigma_p)
    term1 = np.trace(inv_Sigma_p @ Sigma_q)
    diff = mu_p - mu_q
    term2 = diff.T @ inv_Sigma_p @ diff
    term3 = np.log(np.linalg.det(Sigma_p) / np.linalg.det(Sigma_q))
    return 0.5 * (term1 + term2 - dim + term3)

# --------------------------------------------------------------------------- #
# BaM algorithm
# --------------------------------------------------------------------------- #

def bam_step(mu, Sigma, batch_size, lambda_reg, score_func):
    """
    Perform one BaM iteration.
    Args:
        mu: current mean (D,)
        Sigma: current covariance (D,D)
        batch_size: number of samples B
        lambda_reg: inverse regularization (lambda_t)
        score_func: function z -> grad log p(z)  (size D)
    Returns:
        new_mu, new_Sigma
    """
    D = mu.shape[0]
    # Sample batch from current Gaussian
    rng = np.random.default_rng()
    z = rng.multivariate_normal(mu, Sigma, size=batch_size)  # shape (B,D)

    # Scores of the target at the samples
    g = np.vstack([score_func(z_i) for z_i in z])  # shape (B,D)

    # Batch statistics
    bar_z = np.mean(z, axis=0)                       # (D,)
    C = np.cov(z, rowvar=False, bias=True)          # (D,D)
    bar_g = np.mean(g, axis=0)                       # (D,)
    Gamma = np.cov(g, rowvar=False, bias=True)      # (D,D)

    # Compute U and V
    U = lambda_reg * Gamma + (lambda_reg / (1 + lambda_reg)) * np.outer(bar_g, bar_g)
    V = Sigma + lambda_reg * C + (lambda_reg / (1 + lambda_reg)) * np.outer(mu - bar_z, mu - bar_z)

    # Solve quadratic matrix equation: Sigma_{t+1} U Sigma_{t+1} + Sigma_{t+1} = V
    # Closed‑form solution:
    #  Sigma = 2 V (I + (I + 4 U V)^{1/2})^{-1}
    # Compute square root of (I + 4 U V)
    A = np.eye(D) + 4 * U @ V
    sqrtA = sqrtm(A)
    inv_term = np.linalg.inv(np.eye(D) + sqrtA)
    new_Sigma = 2 * V @ inv_term

    # Update mean
    new_mu = (1 / (1 + lambda_reg)) * mu + (lambda_reg / (1 + lambda_reg)) * (new_Sigma @ bar_g + bar_z)

    return new_mu, new_Sigma

# --------------------------------------------------------------------------- #
# Simple ELBO baseline (optional)
# --------------------------------------------------------------------------- #

def elbo_grad(mu, Sigma, batch_size, target_logp, target_logp_grad):
    """
    Estimate gradient of the ELBO w.r.t. mu and Sigma using reparameterisation.
    Returns: grad_mu, grad_Sigma
    """
    D = mu.shape[0]
    rng = np.random.default_rng()
    # Sample epsilon ~ N(0, I)
    eps = rng.standard_normal((batch_size, D))
    # z = mu + L epsilon, where L is Cholesky of Sigma
    L = np.linalg.cholesky(Sigma)
    z = mu + eps @ L.T  # shape (B,D)

    # Evaluate log p and its gradient
    logp = np.array([target_logp(z_i) for z_i in z])          # (B,)
    grad_logp = np.vstack([target_logp_grad(z_i) for z_i in z])  # (B,D)

    # ELBO gradient (derivative of -E_q[log p(z)] + E_q[log q(z)])
    # For Gaussian q, the gradient wrt mu is E[grad log p(z)]
    grad_mu = -np.mean(grad_logp, axis=0)

    # Gradient wrt Sigma: (1/2) * E[grad log p(z) (z-mu)^T] - (1/2) * I
    z_mu = z - mu
    grad_Sigma = -0.5 * np.mean(grad_logp[:, :, None] * z_mu[:, None, :], axis=0) - 0.5 * np.eye(D)

    return grad_mu, grad_Sigma

# --------------------------------------------------------------------------- #
# Target functions (Gaussian)
# --------------------------------------------------------------------------- #

def gaussian_target(mu_true, Sigma_true):
    """Return log p, grad log p, and analytic KL evaluator."""
    inv_Sigma_true = np.linalg.inv(Sigma_true)

    def logp(z):
        diff = z - mu_true
        return -0.5 * (diff @ inv_Sigma_true @ diff + np.log(np.linalg.det(Sigma_true)) + len(mu_true) * np.log(2 * np.pi))

    def grad_logp(z):
        diff = z - mu_true
        return -inv_Sigma_true @ diff

    return logp, grad_logp

# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="BaM experiment on a synthetic Gaussian target")
    parser.add_argument("--dim", type=int, default=16, help="Dimensionality of the target")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size B")
    parser.add_argument("--iterations", type=int, default=200, help="Number of BaM iterations")
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=5.0,
                        help="Inverse regularisation parameter λ")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Generate synthetic Gaussian target
    mu_true = rng.standard_normal(args.dim)
    Sigma_true = random_spd_matrix(args.dim, seed=args.seed + 1)

    logp, grad_logp = gaussian_target(mu_true, Sigma_true)

    # Initialise variational distribution
    mu_q = rng.uniform(0, 0.1, size=args.dim)
    Sigma_q = np.eye(args.dim)

    # Tracking
    kl_history = []
    grad_evals = 0

    start_time = time.time()

    for t in tqdm(range(args.iterations), desc="BaM iterations"):
        mu_q, Sigma_q = bam_step(mu_q, Sigma_q, args.batch_size,
                                 args.lambda_reg, grad_logp)
        grad_evals += args.batch_size  # one gradient eval per sample
        kl = kl_gaussian(mu_q, Sigma_q, mu_true, Sigma_true)
        kl_history.append(kl)

    elapsed = time.time() - start_time

    # Final results
    final_kl = kl_history[-1]
    print("\n=== Final Results ===")
    print(f"Target mean                 : {mu_true}")
    print(f"Target covariance (diag)    : {np.diag(Sigma_true)}")
    print(f"Variational mean (final)    : {mu_q}")
    print(f"Variational covariance (diag): {np.diag(Sigma_q)}")
    print(f"Final KL(q||p)              : {final_kl:.6f}")
    print(f"Total gradient evaluations : {grad_evals}")
    print(f"Runtime (s)                 : {elapsed:.2f}")

if __name__ == "__main__":
    main()