# experiments/run_gaussian.py
import os
import jax
import jax.numpy as jnp
import numpy as np
import csv

from src.utils import mvn_score, mvn_logpdf, kl_gaussian
from src.bat_match import bam
from src.advi import advi

def gaussian_target(mu_star, Sigma_star):
    """Return target score and log‑pdf for a Gaussian."""
    invS = jnp.linalg.inv(Sigma_star)
    @jax.jit
    def target_score(z):
        return invS @ (mu_star - z)
    @jax.jit
    def target_logp(z):
        return mvn_logpdf(z, mu_star, Sigma_star)
    return target_score, target_logp

def run_experiment(D, seed=42):
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    # Random Gaussian target
    A = jax.random.normal(subkey, (D, D))
    Sigma_star = A @ A.T + 0.1 * jnp.eye(D)
    mu_star = jax.random.normal(subkey, (D,)) * 0.5

    target_score, target_logp = gaussian_target(mu_star, Sigma_star)

    # Initial variational parameters
    mu0 = jnp.zeros(D)
    Sigma0 = jnp.eye(D)

    # Hyper‑parameters
    B = 20
    T = 2000
    lambda_reg = B * D          # as used in the paper
    lr = 0.01

    # Run BaM
    key, subkey = jax.random.split(key)
    mu_bam, Sigma_bam = bam(mu0, Sigma0, subkey, B, lambda_reg, target_score, T)

    # Run ADVI
    key, subkey = jax.random.split(key)
    mu_advi, Sigma_advi = advi(mu0, Sigma0, subkey, B, lr, T, target_logp)

    # Compute KL divergences
    kl_bam = kl_gaussian(mu_bam, Sigma_bam, mu_star, Sigma_star)
    kl_advi = kl_gaussian(mu_advi, Sigma_advi, mu_star, Sigma_star)

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    with open(f"results/gaussian_D{D}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["D", "mu_star", "Sigma_star", "kl_bam", "kl_advi"])
        writer.writerow([D, mu_star.tolist(), Sigma_star.tolist(), float(kl_bam), float(kl_advi)])

    print(f"D={D} | KL(BaM)={kl_bam:.4f} | KL(ADVI)={kl_advi:.4f}")

if __name__ == "__main__":
    for D in [4, 16, 64, 256]:
        run_experiment(D)