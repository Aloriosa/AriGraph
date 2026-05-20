"""
Training script for the Simformer on the Two‑Moons simulator.
"""

import jax
import jax.numpy as jnp
import flax
import optax
import numpy as np
import os
import pickle
from tqdm import trange

from .simulator import batch_simulate_two_moons
from .tokenizer import Tokenizer
from .simformer import Simformer
from .sde import forward_sample
from .utils import set_seed

# Hyper‑parameters
BATCH_SIZE = 512
NUM_EPOCHS = 20
LR = 1e-3
SEQ_LEN = 4  # 2 θ + 2 x
HIDDEN_DIM = 64
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Random mask settings
MASK_TYPES = ["joint", "posterior", "likelihood", "random"]
RAND_PROB = 0.3  # probability for random mask

def sample_mask(rng, batch_size, seq_len, d_theta, d_x, mask_type):
    if mask_type == "joint":
        mask = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    elif mask_type == "posterior":
        mask = jnp.concatenate(
            [jnp.zeros((batch_size, d_theta), dtype=jnp.int32),
             jnp.ones((batch_size, d_x), dtype=jnp.int32)], axis=1)
    elif mask_type == "likelihood":
        mask = jnp.concatenate(
            [jnp.ones((batch_size, d_theta), dtype=jnp.int32),
             jnp.zeros((batch_size, d_x), dtype=jnp.int32)], axis=1)
    else:  # random
        mask = jax.random.bernoulli(rng, p=RAND_PROB,
                                    shape=(batch_size, seq_len)).astype(jnp.int32)
    return mask

def init_model(rng, seq_len):
    """
    Initialise model parameters.
    """
    model = Simformer(hidden_dim=HIDDEN_DIM)
    dummy_token = jnp.zeros((BATCH_SIZE, seq_len, HIDDEN_DIM))
    dummy_t = jnp.ones((BATCH_SIZE,), dtype=jnp.float32)
    variables = model.init(rng, dummy_token, dummy_t, deterministic=True)
    return variables["params"], model

@jax.jit
def compute_loss(params, model, batch, rng):
    """
    Forward pass and loss computation.
    batch: dict with keys 'theta', 'x', 'mask', 't'
    """
    theta, x, mask, t = batch["theta"], batch["x"], batch["mask"], batch["t"]
    d_theta = theta.shape[1]
    d_x = x.shape[1]
    seq_len = d_theta + d_x

    tokenizer = Tokenizer(hidden_dim=HIDDEN_DIM)
    # Token embeddings for current batch
    token_embeds = tokenizer(theta, x, mask)

    # Sample forward diffusion
    rng, subkey = jax.random.split(rng)
    x0 = jnp.concatenate([theta, x], axis=1)
    x_t, target_score = forward_sample(subkey, x0, t)

    # Token embeddings for noisy sample
    token_embeds_t = tokenizer(x_t[:, :d_theta], x_t[:, d_theta:], mask)

    # Predict score
    pred_score = model.apply({"params": params}, token_embeds_t, t, deterministic=True)

    # Expand target_score to hidden_dim
    target_score_expanded = target_score[:, :, None] * jnp.ones((1, 1, HIDDEN_DIM))

    loss = jnp.mean((pred_score - target_score_expanded) ** 2)
    return loss, rng

def train():
    rng = jax.random.PRNGKey(0)
    set_seed(0)
    # Initialise model
    params, model = init_model(rng, SEQ_LEN)
    opt_state = optax.adam(LR).init(params)

    @jax.jit
    def step(params, opt_state, batch, rng):
        (loss, rng), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params, model, batch, rng)
        updates, opt_state = optax.adam(LR).update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, rng

    for epoch in trange(NUM_EPOCHS, desc="Epoch"):
        rng, subkey = jax.random.split(rng)
        theta, x = batch_simulate_two_moons(subkey, BATCH_SIZE)
        # Randomly choose mask type
        rng, mask_key = jax.random.split(rng)
        mask_type = jax.random.choice(mask_key,
                                      jnp.array(MASK_TYPES), p=None).item()
        mask = sample_mask(mask_key, BATCH_SIZE, SEQ_LEN, 2, 2, mask_type)
        t = jax.random.uniform(subkey, (BATCH_SIZE,), minval=0.0, maxval=1.0)
        batch = {"theta": theta, "x": x, "mask": mask, "t": t}
        params, opt_state, loss, rng = step(params, opt_state, batch, rng)
        if epoch % 5 == 0:
            print(f"Epoch {epoch} loss: {loss:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "simformer.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"params": params, "rng": rng}, f)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()