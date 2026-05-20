"""Training and inference utilities for the toy Simformer."""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from typing import Tuple
from .model import ScoreModel
from .utils import perturb, sigma_t, two_moons_sampler

# --------------------------------------------------------------
# Loss and gradient functions
# --------------------------------------------------------------
def loss_fn(params: hk.Params,
            rng: jax.random.PRNGKey,
            batch: Tuple[jnp.ndarray, jnp.ndarray],
            cond_mask: jnp.ndarray,
            t: float,
            sigma_min: float = 0.0001,
            sigma_max: float = 15.0) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute score matching loss for a single batch.
    """
    θ, x_clean = batch  # (batch, n_vars)
    # Concatenate to get joint vector
    values = jnp.concatenate([θ, x_clean], axis=1)  # (batch, 2n)

    # Generate noisy observation at time t
    rng, noise_key = jax.random.split(rng)
    noisy_values, eps = perturb(values, t, noise_key, sigma_min, sigma_max)

    # Build mask: observed variables are 1, latent are 0
    # For training we mask a random subset of the joint
    # Here we simply use the provided cond_mask
    pred_scores = ScoreModel(n_vars=values.shape[1]).apply(params, rng, noisy_values, cond_mask)
    # Target score: (clean - noisy) / sigma_t^2
    σ = sigma_t(t, sigma_min, sigma_max)
    true_scores = (values - noisy_values) / (σ ** 2)

    # Loss only on latent variables
    loss = jnp.mean(((pred_scores - true_scores) ** 2) * (1 - cond_mask))

    return loss, (pred_scores, true_scores)


@jax.jit
def update(state, rng, batch, cond_mask, t):
    """Single gradient step."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params, rng, batch, cond_mask, t)
    updates, new_opt_state = state.opt.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state._replace(params=new_params, opt_state=new_opt_state)
    return new_state, loss


# --------------------------------------------------------------
# Training loop
# --------------------------------------------------------------
class TrainerState:
    def __init__(self, params, opt_state, opt, rng):
        self.params = params
        self.opt_state = opt_state
        self.opt = opt
        self.rng = rng

    def _replace(self, **kwargs):
        return TrainerState(
            params=kwargs.get('params', self.params),
            opt_state=kwargs.get('opt_state', self.opt_state),
            opt=kwargs.get('opt', self.opt),
            rng=kwargs.get('rng', self.rng),
        )


def train(num_steps: int = 3000,
          batch_size: int = 128,
          learning_rate: float = 1e-3,
          seed: int = 42) -> hk.Params:
    """
    Train the score model on the Two‑Moons toy dataset.
    Returns trained parameters.
    """
    rng = jax.random.PRNGKey(seed)
    rng, data_key, model_key = jax.random.split(rng, 3)

    # Generate full dataset
    θ, x = two_moons_sampler(data_key, n_samples=20000)
    dataset = (θ, x)

    # Prepare model
    def forward(values, cond_mask):
        return ScoreModel(n_vars=values.shape[1]).apply(params, None, values, cond_mask)

    # Initialize parameters
    dummy_values = jnp.zeros((1, 4))
    dummy_cond = jnp.zeros((1, 4))
    params = hk.transform(forward).init(model_key, dummy_values, dummy_cond)

    # Optimizer
    opt = optax.adamw(learning_rate)
    opt_state = opt.init(params)

    state = TrainerState(params, opt_state, opt, rng)

    # Precompute condition mask distribution
    # During training we sample a random mask for each batch:
    #  - 50% chance to mask each variable independently
    def sample_mask(rng, shape):
        return jax.random.bernoulli(rng, p=0.5, shape=shape).astype(jnp.float32)

    # Training loop
    for step in range(num_steps):
        rng, batch_key, mask_key, t_key = jax.random.split(state.rng, 4)

        # Sample batch
        idx = jax.random.randint(batch_key, shape=(batch_size,), minval=0, maxval=dataset[0].shape[0])
        batch = (dataset[0][idx], dataset[1][idx])

        # Sample condition mask
        mask = sample_mask(mask_key, (batch_size, 4))

        # Sample diffusion time t ∈ [0,1]
        t = jax.random.uniform(t_key, minval=0.0, maxval=1.0)

        state, loss = update(state, batch_key, batch, mask, t)

        if step % 500 == 0 or step == num_steps - 1:
            print(f"[{step:5d}] loss: {loss:.6f}")

    return state.params