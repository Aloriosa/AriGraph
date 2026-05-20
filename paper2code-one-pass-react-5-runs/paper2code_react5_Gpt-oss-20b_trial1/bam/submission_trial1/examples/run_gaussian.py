"""
Minimal reproducible experiment for the BaM algorithm on a synthetic Gaussian target.
The script runs BaM, computes forward and reverse KL divergences, and plots the
convergence trajectory.  The results are saved in the `results/` directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.baum import BaM
from src.utils import kl_gaussian, kl_gaussian_reverse

# Ensure reproducibility
np.random.seed(0)

# Create output directory
os.makedirs("results", exist_ok=True)

# Define target Gaussian (randomly generated)
dim = 10
A = np.random.randn(dim, dim)
target_cov = A @ A.T + 0.1 * np.eye(dim)   # make it strictly positive definite
target_mean = np.random.randn(dim)

# Initial variational parameters
init_mean = np.random.randn(dim) * 0.1
init_cov = np.eye(dim)

# BaM hyperparameters
batch_size = 200
lambda_reg = 50.0   # large λ corresponds to aggressive matching
num_iter = 200

print("Running BaM on a synthetic Gaussian target...")
baum = BaM(target_mean=target_mean,
           target_cov=target_cov,
           dim=dim,
           init_mean=init_mean,
           init_cov=init_cov,
           batch_size=batch_size,
           lambda_reg=lambda_reg,
           num_iter=num_iter,
           seed=123)

history = baum.run(verbose=True)

# Compute final KL values
final_mu = history['mu'][-1]
final_Sigma = history['Sigma'][-1]
kl_forward_final = kl_gaussian_reverse(final_mu, final_Sigma, target_mean, target_cov)
kl_reverse_final = kl_gaussian(final_mu, final_Sigma, target_mean, target_cov)

print("\nFinal KL divergences:")
print(f"KL(p || q) (forward)  = {kl_forward_final:.6f}")
print(f"KL(q || p) (reverse) = {kl_reverse_final:.6f}")

# Plot convergence
iterations = np.arange(1, num_iter + 1)

plt.figure(figsize=(8, 5))
plt.semilogy(iterations, history['kl_forward'], label='KL(p || q)  (forward)')
plt.semilogy(iterations, history['kl_reverse'], label='KL(q || p)  (reverse)')
plt.xlabel('Iteration')
plt.ylabel('KL divergence (log scale)')
plt.title('BaM convergence on synthetic Gaussian target')
plt.legend()
plt.tight_layout()
plot_path = os.path.join("results", "baum_convergence.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")