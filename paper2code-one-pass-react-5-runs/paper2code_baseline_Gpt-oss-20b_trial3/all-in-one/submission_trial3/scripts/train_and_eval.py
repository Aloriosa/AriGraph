#!/usr/bin/env python3
"""Script that trains the toy Simformer and samples from the posterior."""

import pickle
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from tqdm import tqdm
from simformer.trainer import train
from simformer.utils import two_moons_sampler, sigma_t, perturb
from simformer.model import ScoreModel
import argparse

# --------------------------------------------------------------
# Sampling from the learned posterior
# --------------------------------------------------------------
def sample_posterior(params,
                     rng: jax.random.PRNGKey,
                     x_obs: jnp.ndarray,
                     n_samples: int = 5000,
                     n_steps: int = 200,
                     learning_rate: float = 1e-3,
                     seed: int = 123) -> jnp.ndarray:
    """
    Sample from the posterior p(θ | x_obs) using reverse diffusion.
    """
    rng, sample_key = jax.random.split(rng)
    # Initialise latent variables with Gaussian noise at terminal time T
    T = 1.0
    sigma_T = sigma_t(T)
    θ_dim = x_obs.shape[0]
    latent = jax.random.normal(sample_key, shape=(n_samples, θ_dim)) * sigma_T

    # Build full joint variable array: [θ, x]
    # Observed x is fixed; θ latent
    def step_fn(state, _):
        rng, = state
        rng, key = jax.random.split(rng)
        # Compute current t
        t = T - (_ / n_steps) * T
        # Build joint
        joint = jnp.concatenate([state[2], jnp.tile(x_obs, (state[2].shape[0], 1))], axis=1)
        # Condition mask: observed x are 1, θ latent 0
        cond_mask = jnp.concatenate([jnp.zeros((state[2].shape[0], θ_dim)),
                                     jnp.ones((state[2].shape[0], x_obs.shape[0]))], axis=1)
        # Predict score
        pred_score = ScoreModel(n_vars=joint.shape[1]).apply(params, key, joint, cond_mask)
        # Gradient step: reverse SDE
        σ = sigma_t(t)
        # For VESDE: f=0, g=σ
        drift = -σ ** 2 * pred_score
        noise = jax.random.normal(key, shape=state[2].shape) * σ * jnp.sqrt(T / n_steps)
        new_latent = state[2] + drift * (T / n_steps) + noise
        return (rng, key, new_latent), None

    # Initial state
    init_state = (rng, sample_key, latent)

    # Run reverse diffusion
    state, _ = jax.lax.scan(step_fn, init_state, None, length=n_steps)
    final_latent = state[2]
    return final_latent


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train_steps", type=int, default=3000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Number of posterior samples")
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Number of reverse diffusion steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Train
    print("Training model...")
    params = train(num_steps=args.n_train_steps,
                   batch_size=args.batch_size,
                   learning_rate=args.learning_rate,
                   seed=args.seed)

    # Save parameters
    with open("model_params.pkl", "wb") as f:
        pickle.dump(params, f)

    # Generate a synthetic observation for inference
    rng = jax.random.PRNGKey(args.seed + 1)
    rng, sim_key = jax.random.split(rng)
    θ_true, x_true = two_moons_sampler(sim_key, n_samples=1)
    x_obs = x_true[0]
    print(f"True θ: {θ_true[0]}")
    print(f"Observation x: {x_obs}")

    # Sample posterior
    print("Sampling posterior...")
    posterior_samples = sample_posterior(params,
                                        rng,
                                        x_obs,
                                        n_samples=args.n_samples,
                                        n_steps=args.n_steps,
                                        seed=args.seed + 2)

    # Save to CSV
    np.savetxt("posterior_samples.csv",
               posterior_samples,
               delimiter=",",
               header="theta_0,theta_1",
               comments="")

    print(f"Saved posterior samples to posterior_samples.csv ({posterior_samples.shape[0]} samples)")


if __name__ == "__main__":
    main()