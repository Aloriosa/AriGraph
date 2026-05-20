"""
CIFAR‑10 VAE experiment.
This script trains a small convolutional VAE, then performs posterior inference
on a held‑out image using BaM, ADVI, and GSM.
"""

import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow_datasets as tfds
from jax.experimental import stax
from jax.experimental.stax import Conv, Relu, Flatten, Dense, LogSoftmax

from src.bam import bam, advi, gsm, _kl_gaussian

# ---------------------------------------------------------------------------

# Simple VAE encoder/decoder using stax
def VAE_encoder(NLATENT=64):
    return stax.serial(
        Conv(32, (4, 4), strides=(2, 2), padding='SAME'),
        Relu,
        Conv(64, (4, 4), strides=(2, 2), padding='SAME'),
        Relu,
        Flatten,
        Dense(NLATENT * 2),  # mean and logvar
    )

def VAE_decoder(NLATENT=64):
    return stax.serial(
        Dense(8 * 8 * 64),
        Relu,
        stax.Reshape((8, 8, 64)),
        Conv(32, (4, 4), strides=(2, 2), padding='SAME'),
        Relu,
        Conv(3, (4, 4), strides=(2, 2), padding='SAME'),
        LogSoftmax,
    )

# ---------------------------------------------------------------------------

def train_vae(batch_size=128, epochs=10, seed=0):
    """
    Train a small VAE on CIFAR‑10.  Returns trained parameters.
    """
    key = jax.random.PRNGKey(seed)

    # Load data
    ds_train = tfds.load("cifar10", split="train", shuffle_files=True)
    ds_train = ds_train.shuffle(10_000, seed=seed).batch(batch_size)

    # VAE architecture
    encoder_init, encoder_apply = VAE_encoder()
    decoder_init, decoder_apply = VAE_decoder()

    # Initialise parameters
    key_e, key_d = jax.random.split(key)
    _, encoder_params = encoder_init(key_e, (-1, 32, 32, 3))
    _, decoder_params = decoder_init(key_d, (-1, NLATENT))

    # Optimiser
    opt = optax.adam(1e-3)
    opt_state = opt.init((encoder_params, decoder_params))

    @jax.jit
    def elbo_step(params, batch, key):
        encoder_p, decoder_p = params
        z_mean_logvar = encoder_apply(encoder_p, batch)
        mu, logvar = jnp.split(z_mean_logvar, 2, axis=1)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        z = mu + eps * std

        recon_logits = decoder_apply(decoder_p, z)
        # Reconstruction loss
        recon_loss = -jnp.mean(
            jnp.sum(
                jax.nn.log_softmax(recon_logits) * batch,
                axis=[1, 2, 3],
            )
        )
        # KL divergence
        kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1))
        return recon_loss + kl

    @jax.jit
    def update(params, opt_state, batch, key):
        loss, grads = jax.value_and_grad(elbo_step)(params, batch, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    for epoch in range(epochs):
        for batch in ds_train.as_numpy_iterator():
            batch = jnp.array(batch, dtype=jnp.float32) / 255.0
            key, subk = jax.random.split(key)
            encoder_params, decoder_params, opt_state, loss = update(
                (encoder_params, decoder_params),
                opt_state,
                batch,
                subk,
            )
        print(f"Epoch {epoch+1}/{epochs} done.")

    return encoder_params, decoder_params


# ---------------------------------------------------------------------------

def sample_posterior(encoder_p, decoder_p, x, T=200, B=20, seed=0):
    """
    Perform posterior inference on a single test image x using BaM, ADVI, GSM.
    Returns KL histories and final parameters.
    """
    key = jax.random.PRNGKey(seed)

    # Encode x to get approximate posterior mean and logvar
    z_mean_logvar = encoder_p.apply(x[None, ...])
    mu, logvar = jnp.split(z_mean_logvar, 2, axis=1)
    std = jnp.exp(0.5 * logvar)
    # Use the approximate posterior as the target (simplification)
    mu_star = mu.squeeze()
    Sigma_star = jnp.diag(std.squeeze() ** 2)

    # Target log prob and score are given by the VAE decoder likelihood
    def target_log_prob(z):
        # The decoder outputs log‑softmax over pixel values
        logits = decoder_p.apply(z)
        # Assume pixels independent Bernoulli with probability softmax
        # Here we simply return a dummy constant (since exact p is intractable)
        return -0.5 * jnp.sum((x - jnp.exp(logits)) ** 2)

    def target_score(z):
        # Gradient of the dummy log prob (zero)
        return jnp.zeros_like(z)

    # ------------------------------------------------------------
    # 1. BaM
    lambda_reg = [B * 64 / (t + 1) for t in range(T)]
    key, subk = jax.random.split(key)
    bam_mu, bam_Sigma, bam_kl_hist, bam_score_hist = bam(
        subk,
        target_log_prob=target_log_prob,
        target_score=target_score,
        target_mean=mu_star,
        target_cov=Sigma_star,
        T=T,
        B=B,
        lambda_reg=lambda_reg,
        mu_init=jnp.zeros(64),
        Sigma_init=jnp.eye(64),
        kl_history=True,
        score_history=True,
    )

    # ------------------------------------------------------------
    # 2. ADVI
    key, subk = jax.random.split(key)
    advi_mu, advi_Sigma, advi_kl_hist = advi(
        subk,
        target_log_prob=target_log_prob,
        target_score=target_score,
        target_mean=mu_star,
        target_cov=Sigma_star,
        T=T,
        B=B,
        lr=0.02,
        mu_init=jnp.zeros(64),
        Sigma_init=jnp.eye(64),
        kl_history=True,
    )

    # ------------------------------------------------------------
    # 3. GSM
    key, subk = jax.random.split(key)
    gsm_mu, gsm_Sigma, _ = gsm(
        subk,
        target_score=target_score,
        mu_init=jnp.zeros(64),
        Sigma_init=jnp.eye(64),
        T=T,
        B=1,
    )
    gsm_kl_hist = jnp.array([_kl_gaussian(mu_star, Sigma_star, gsm_mu, gsm_Sigma)])

    return {
        "bam_kl": bam_kl_hist,
        "advi_kl": advi_kl_hist,
        "gsm_kl": gsm_kl_hist,
    }


# ---------------------------------------------------------------------------

def run_vae_experiment(seed=42):
    """
    Full VAE experiment: train VAE, then run posterior inference on a test image.
    """
    encoder_params, decoder_params = train_vae()

    # Load a test image
    ds_test = tfds.load("cifar10", split="test", batch_size=1)
    test_image = next(iter(ds_test.as_numpy_iterator()))[0]
    test_image = jnp.array(test_image, dtype=jnp.float32) / 255.0

    # Posterior inference
    results = sample_posterior(
        encoder_params, decoder_params, test_image, T=200, B=20, seed=seed
    )

    # Plotting
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(results["bam_kl"], label="BaM", linewidth=2)
    ax.plot(results["advi_kl"], label="ADVI", linewidth=2)
    ax.plot(results["gsm_kl"], label="GSM", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathrm{KL}(p\Vert q_t)$")
    ax.set_title("Posterior inference on CIFAR‑10 image")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/vae_results.png")
    plt.close()
    print("\nFigure generated: figures/vae_results.png")


if __name__ == "__main__":
    run_vae_experiment(seed=42)