"""
Posterior sampling using the trained Simformer via guided reverse diffusion.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
from tqdm import trange

from .simulator import sample_prior_two_moons, sample_data_two_moons
from .tokenizer import Tokenizer
from .simformer import Simformer
from .sde import reverse_sde_step, sigma
from .utils import set_seed

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyper‑parameters
NUM_SAMPLES = 2000
NUM_STEPS = 100
D_THETA = 2
D_X = 2
SEQ_LEN = D_THETA + D_X
HIDDEN_DIM = 64

def load_checkpoint():
    ckpt_path = os.path.join(CHECKPOINT_DIR, "simformer.pkl")
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    return data["params"]

def posterior_sample(params, rng, x_obs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Generate posterior samples for θ given observed x using reverse diffusion.
    """
    tokenizer = Tokenizer(hidden_dim=HIDDEN_DIM)
    model = Simformer(hidden_dim=HIDDEN_DIM)

    batch_size = x_obs.shape[0]

    # Initialize from pure noise at t=1
    rng, subkey = jax.random.split(rng)
    noise = jax.random.normal(subkey, (batch_size, SEQ_LEN))
    x_t = noise * sigma(jnp.array([1.0]))  # at t=1

    # Reverse diffusion
    for i in trange(NUM_STEPS, desc="Reverse SDE"):
        t = 1.0 - (i + 1) / NUM_STEPS
        rng, subkey = jax.random.split(rng)
        # Compute score
        theta_t = x_t[:, :D_THETA]
        x_t_part = x_t[:, D_THETA:]
        token_embeds_t = tokenizer(theta_t, x_t_part, mask)
        score = model.apply({"params": params}, token_embeds_t, jnp.array([t]*batch_size), deterministic=True)
        # Zero out score for observed variables
        mask_expanded = mask[:, :, None]  # (B, seq_len, 1)
        score = score * (1.0 - mask_expanded)

        x_t = reverse_sde_step(x_t, score, t, dt=1.0/NUM_STEPS, rng=subkey)

    # After reverse diffusion, x_t contains sampled joint (θ, x)
    # Return θ samples
    return x_t[:, :D_THETA]

def evaluate():
    set_seed(42)
    rng = jax.random.PRNGKey(42)
    params = load_checkpoint()

    # Generate a synthetic observation
    rng, subkey = jax.random.split(rng)
    theta_true = sample_prior_two_moons(subkey, 1)
    x_obs = sample_data_two_moons(subkey, theta_true)

    # Observation mask: first D_THETA columns (θ) unobserved -> 0, last D_X columns (x) observed -> 1
    mask = jnp.concatenate([jnp.zeros((1, D_THETA), dtype=jnp.int32),
                            jnp.ones((1, D_X), dtype=jnp.int32)], axis=1)

    # Posterior sampling
    samples = posterior_sample(params, rng, x_obs, mask)
    samples_np = np.asarray(samples)

    # Save samples
    out_path = os.path.join(RESULTS_DIR, "posterior_samples.npy")
    np.save(out_path, samples_np)
    print(f"Posterior samples saved to {out_path}")

    # Simple evaluation: mean‑squared error between true θ and posterior mean
    posterior_mean = np.mean(samples_np, axis=0)
    mse = np.mean((posterior_mean - np.asarray(theta_true))**2)
    print(f"Mean‑squared‑error of posterior mean vs true θ: {mse:.4f}")

if __name__ == "__main__":
    evaluate()