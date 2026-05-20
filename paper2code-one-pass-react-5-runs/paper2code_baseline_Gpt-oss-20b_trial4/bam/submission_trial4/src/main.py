import numpy as np
import csv
from utils import log_pdf_gaussian, kl_gaussian, forward_kl, score_gaussian
from bam import bam_step
from advi import advi_step

def run_bam(mu0, Sigma0, B, lam, T, mu_target, Sigma_target, rng):
    mu, Sigma = mu0.copy(), Sigma0.copy()
    for t in range(T):
        # sample from current q
        eps = rng.randn(B, mu.shape[0])
        L = np.linalg.cholesky(Sigma)
        samples = mu + eps @ L.T
        # compute scores of target at samples
        scores = np.array([score_gaussian(z, mu_target, Sigma_target) for z in samples])
        mu, Sigma = bam_step(mu, Sigma, samples, scores, lam)
    return mu, Sigma

def run_advi(mu0, Sigma0, lr, T, mu_target, Sigma_target):
    mu, Sigma = mu0.copy(), Sigma0.copy()
    for t in range(T):
        mu, Sigma = advi_step(mu, Sigma, mu_target, Sigma_target, lr)
    return mu, Sigma

def main():
    rng = np.random.default_rng(seed=42)

    # Target Gaussian
    mu_target = np.array([1.0, 2.0, 3.0])
    Sigma_target = np.array([[2.0, 0.5, 0.3],
                             [0.5, 1.0, 0.2],
                             [0.3, 0.2, 1.5]])

    # Initial variational parameters
    mu0 = np.zeros_like(mu_target)
    Sigma0 = np.eye(len(mu_target))

    # Settings
    B = 20
    lam = 1.0
    T = 200
    lr = 0.01

    # Run BaM
    mu_bam, Sigma_bam = run_bam(mu0, Sigma0, B, lam, T, mu_target, Sigma_target, rng)

    # Run ADVI
    mu_advi, Sigma_advi = run_advi(mu0, Sigma0, lr, T, mu_target, Sigma_target)

    # Compute divergences
    kl_bam = kl_gaussian(mu_bam, Sigma_bam, mu_target, Sigma_target)
    kl_advi = kl_gaussian(mu_advi, Sigma_advi, mu_target, Sigma_target)
    fkl_bam = forward_kl(mu_bam, Sigma_bam, mu_target, Sigma_target)
    fkl_advi = forward_kl(mu_advi, Sigma_advi, mu_target, Sigma_target)

    # Print summary
    print(f"Iteration {T}:")
    print(f"  BaM:  μ = {mu_bam}, Σ =\n{Sigma_bam}")
    print(f"        forward KL = {fkl_bam:.4f}, reverse KL = {kl_bam:.4f}")
    print(f"  ADVI: μ = {mu_advi}, Σ =\n{Sigma_advi}")
    print(f"        forward KL = {fkl_advi:.4f}, reverse KL = {kl_advi:.4f}")

    # Write to CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Mu", "Sigma", "Forward KL", "Reverse KL"])
        writer.writerow(["BaM", mu_bam.tolist(), Sigma_bam.tolist(), fkl_bam, kl_bam])
        writer.writerow(["ADVI", mu_advi.tolist(), Sigma_advi.tolist(), fkl_advi, kl_advi])

if __name__ == "__main__":
    main()