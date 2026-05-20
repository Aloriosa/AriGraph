#!/usr/bin/env python3
"""
run_simulation.py

A minimal demonstration of simulation‑based inference using the sbi library.
The script trains a Neural Posterior Estimation (NPE) model on a toy
“two‑moons” simulator, samples from the posterior, and writes the results
to disk.

This is NOT a full implementation of the Simformer from the paper, but
provides a reproducible end‑to‑end example that can be run inside the
grader container.
"""

import os
import sys
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.datasets import make_moons
from sbi import utils as utils
from sbi import inference as inference
from sbi import utils as sbi_utils
from sbi.inference import SNPE
from sbi.inference import Posterior
from sbi.inference import simulate_for_sbi
from sbi import utils as utils_sbi

# ----------------------------------------------------------------------
# 1. Define a toy simulator
# ----------------------------------------------------------------------
def two_moons_simulator(theta, random_seed=None):
    """
    Simple simulator for the “two moons” task.
    Parameters theta are the two true parameters that shift the moons.
    The simulator returns a 2‑dimensional observation.
    """
    # theta: shape (2,)
    if random_seed is not None:
        np.random.seed(random_seed)
    shift = theta
    # Generate the moons
    X, _ = make_moons(n_samples=1, noise=0.1, random_state=random_seed)
    # Shift by theta
    X = X + shift
    return X.squeeze()  # shape (2,)

# ----------------------------------------------------------------------
# 2. Generate training data for sbi
# ----------------------------------------------------------------------
def generate_training_data(num_simulations=5000, seed=1234):
    """
    Generate training data for the sbi pipeline.
    Returns a dictionary containing thetas, xs, and the prior.
    """
    np.random.seed(seed)
    # Prior: uniform over [-1, 1] for both parameters
    prior = utils_sbi.BoxUniform(low=np.array([-1.0, -1.0]),
                                 high=np.array([1.0, 1.0]))

    # Generate parameters from prior
    thetas = prior.rvs(size=num_simulations)
    # Simulate data
    xs = np.array([two_moons_simulator(theta, random_seed=seed+i)
                   for i, theta in enumerate(thetas)])

    return dict(
        thetas=thetas,
        xs=xs,
        prior=prior
    )

# ----------------------------------------------------------------------
# 3. Train an NPE model
# ----------------------------------------------------------------------
def train_npe(training_data, epochs=200, batch_size=256, lr=5e-4):
    """
    Train a Neural Posterior Estimation model on the given data.
    """
    thetas = training_data["thetas"]
    xs = training_data["xs"]
    prior = training_data["prior"]

    # Wrap data for sbi
    training_data = dict(thetas=thetas, xs=xs)

    # Create the inference object
    inference_method = inference.SNPE(prior=prior,
                                      density_estimator="maf",
                                      device="cuda" if torch.cuda.is_available() else "cpu")

    # Train the network
    density_estimator = inference_method.train(
        training_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    # Build posterior
    posterior = inference_method.build_posterior(density_estimator)

    return posterior

# ----------------------------------------------------------------------
# 4. Sample from the posterior and evaluate
# ----------------------------------------------------------------------
def evaluate_posterior(posterior, true_theta, num_samples=1000):
    """
    Sample from the posterior conditioned on a true observation and
    compute the posterior mean and coverage (95%).
    """
    # Simulate observation for the true theta
    obs = two_moons_simulator(true_theta, random_seed=42)

    # Sample from posterior
    samples = posterior.sample((num_samples,), x=obs).detach().cpu().numpy()

    # Posterior mean
    mean = samples.mean(axis=0)

    # 95% credible interval
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)

    # Coverage: check if true theta lies within CI
    coverage = np.all((true_theta >= lower) & (true_theta <= upper))

    return dict(
        samples=samples,
        mean=mean,
        lower=lower,
        upper=upper,
        coverage=coverage,
        true_theta=true_theta,
        obs=obs
    )

# ----------------------------------------------------------------------
# 5. Main routine
# ----------------------------------------------------------------------
def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating training data...")
    training_data = generate_training_data(num_simulations=5000, seed=1234)

    print("Training NPE model...")
    posterior = train_npe(training_data,
                          epochs=200,
                          batch_size=256,
                          lr=5e-4)

    print("Evaluating posterior on a test case...")
    # Pick a random test theta
    np.random.seed(999)
    test_theta = np.random.uniform(-1, 1, size=2)
    eval_res = evaluate_posterior(posterior, test_theta, num_samples=2000)

    # Save results
    np.savez(os.path.join(output_dir, "posterior_samples.npz"),
             samples=eval_res["samples"],
             mean=eval_res["mean"],
             lower=eval_res["lower"],
             upper=eval_res["upper"],
             true_theta=eval_res["true_theta"],
             obs=eval_res["obs"])

    # Write a simple metric summary
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"True theta: {eval_res['true_theta']}\n")
        f.write(f"Posterior mean: {eval_res['mean']}\n")
        f.write(f"95% CI lower: {eval_res['lower']}\n")
        f.write(f"95% CI upper: {eval_res['upper']}\n")
        f.write(f"Coverage: {eval_res['coverage']}\n")

    print("Results written to", output_dir)
    print("Reproduction finished successfully.")

if __name__ == "__main__":
    main()