import argparse
import os
import numpy as np
import torch
from scipy.stats import norm

def compute_metrics(samples, true_mean, true_var):
    """
    Compute simple metrics: mean error, KL divergence, coverage.
    """
    samples = np.asarray(samples)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples, ddof=1)

    # RMSE of mean
    rmse_mean = abs(sample_mean - true_mean)

    # KL divergence between Gaussian(sample_mean, sample_var) and Gaussian(true_mean, true_var)
    # KL(P||Q) where P ~ N(mu_p, sigma_p^2), Q ~ N(mu_q, sigma_q^2)
    sigma_p2 = sample_var
    sigma_q2 = true_var
    mu_p = sample_mean
    mu_q = true_mean
    kl = 0.5 * ( (sigma_q2 / sigma_p2) + ((mu_p - mu_q)**2) / sigma_p2 - 1 + np.log(sigma_p2 / sigma_q2) )

    # 95% coverage
    ci_low = true_mean - 1.96 * np.sqrt(true_var)
    ci_high = true_mean + 1.96 * np.sqrt(true_var)
    coverage = np.mean((samples >= ci_low) & (samples <= ci_high))

    return {
        "sample_mean": sample_mean,
        "sample_var": sample_var,
        "rmse_mean": rmse_mean,
        "kl_gaussian": kl,
        "coverage_95": coverage
    }

def main(args):
    out_dir = args.out_dir
    # Load posterior samples
    posterior_path = os.path.join(out_dir, "posterior_samples.npy")
    posterior_samples = np.load(posterior_path)

    # True posterior parameters for toy Gaussian
    x_obs = 0.5
    prior_prec = 1.0
    likelihood_prec = 1.0 / 0.1  # 10
    post_prec = prior_prec + likelihood_prec
    true_var = 1.0 / post_prec
    true_mean = true_var * (x_obs / 0.1)

    # Compute metrics for posterior samples
    post_metrics = compute_metrics(posterior_samples, true_mean, true_var)

    # Baseline: prior samples
    prior_samples = np.random.randn(5000, 1).squeeze()
    prior_metrics = compute_metrics(prior_samples, 0.0, 1.0)

    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== Posterior Metrics ===\n")
        for k, v in post_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\n=== Prior Baseline Metrics ===\n")
        for k, v in prior_metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"Metrics written to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TSNPSE posterior")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory containing samples")
    args = parser.parse_args()
    main(args)