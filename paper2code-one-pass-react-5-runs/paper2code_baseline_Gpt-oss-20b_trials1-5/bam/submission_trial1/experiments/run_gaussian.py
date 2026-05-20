#!/usr/bin/env python
"""Toy experiment: BaM on a Gaussian target."""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from src.bam import bam
from src.metrics import forward_kl, reverse_kl, kl_kl_divergence

def parse_args():
    parser = argparse.ArgumentParser(description="BaM Gaussian toy experiment.")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to store figures and logs.")
    parser.add_argument("--D", type=int, default=4,
                        help="Dimension of the Gaussian target.")
    parser.add_argument("--B", type=int, default=20,
                        help="Batch size for BaM.")
    parser.add_argument("--lambda", type=float, default=1.0,
                        help="Inverse regularization (learning rate).")
    parser.add_argument("--T", type=int, default=30,
                        help="Number of BaM iterations.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Target Gaussian
    D = args.D
    A = np.random.randn(D, D)
    target_Sigma = A @ A.T + 0.1 * np.eye(D)  # ensure PD
    target_mu = np.random.randn(D)

    # 2. Initial variational parameters
    mu0 = np.random.randn(D) * 0.1
    Sigma0 = np.eye(D)

    # 3. Run BaM
    kl_forward = []
    kl_reverse = []
    kl_kl = []

    def record(t, mu, Sigma):
        kl_forward.append(forward_kl(mu, Sigma, target_mu, target_Sigma))
        kl_reverse.append(reverse_kl(mu, Sigma, target_mu, target_Sigma))
        kl_kl.append(kl_kl_divergence(mu, Sigma, target_mu, target_Sigma))

    bam(
        target_mu,
        target_Sigma,
        mu0,
        Sigma0,
        B=args.B,
        lambda_t=args.lambda,
        T=args.T,
        verbose=False,
        callback=record,
    )

    # 4. Plot convergence
    plt.figure(figsize=(8, 5))
    plt.semilogy(kl_forward, label="KL(p||q)  (forward)")
    plt.semilogy(kl_reverse, label="KL(q||p)  (reverse)")
    plt.semilogy(kl_kl, label="Score‑based div")
    plt.xlabel("Iteration")
    plt.ylabel("KL / Divergence (log scale)")
    plt.title(f"BaM convergence (D={D}, B={args.B}, λ={args.lambda})")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, "kl_convergence.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Result saved to {out_png}")

if __name__ == "__main__":
    main()