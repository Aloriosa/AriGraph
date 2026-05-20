"""
Simple 2‑level normal hierarchical Bayesian experiment.
The model:
    z0 ~ N(0, 1)
    z1 | z0 ~ N(z0, 1)
    x   | z1 ~ N(z1, 1)
We observe a single data point x=0 and perform posterior inference over
(z0, z1).  The true posterior is a 2‑dimensional Gaussian with
known mean and covariance.
"""

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from src.bam import bam, advi, gsm, _kl_gaussian, _score_divergence

# ---------------------------------------------------------------------------

def hierarchical_log_prob(z: jnp.ndarray, x: float = 0.0) -> jnp.ndarray:
    """
    Unnormalized log density of the posterior p(z0, z1 | x).
    z : shape (2,)
    """
    z0, z1 = z
    # Prior: z0 ~ N(0,1), z1|z0 ~ N(z0,1)
    log_prior = -0.5 * (z0 ** 2 + (z1 - z0) ** 2)
    # Likelihood: x | z1 ~ N(z1,1)
    log_lik = -0.5 * (x - z1) ** 2
    return log_prior + log_lik


def hierarchical_score(z: jnp.ndarray, x: float = 0.0) -> jnp.ndarray:
    """
    Gradient of the log density w.r.t. z.
    """
    z0, z1 = z
    # Derivatives
    d_logp_dz0 = -z0 + (z1 - z0)
    d_logp_dz1 = -(z1 - z0) - (x - z1)
    return jnp.array([d_logp_dz0, d_logp_dz1])


# ---------------------------------------------------------------------------

def run_hierarchical(T: int = 200,
                     B: int = 20,
                     seed: int = 42) -> None:
    """
    Run BaM, ADVI, GSM on the hierarchical target and plot KL & score divergence.
    """
    key = jax.random.PRNGKey(seed)

    D = 2
    mu_star = jnp.array([0.0, 0.0])  # true posterior mean (derived analytically)
    # true posterior covariance:
    Sigma_star = jnp.array([[0.66666667, 0.33333333],
                            [0.33333333, 0.66666667]])

    # ------------------------------------------------------------
    # 1. BaM
    lambda_reg = [B * D / (t + 1) for t in range(T)]
    key, subk = jax.random.split(key)
    bam_mu, bam_Sigma, bam_kl_hist, bam_score_hist = bam(
        subk,
        target_log_prob=hierarchical_log_prob,
        target_score=hierarchical_score,
        target_mean=mu_star,
        target_cov=Sigma_star,
        T=T,
        B=B,
        lambda_reg=lambda_reg,
        mu_init=jnp.zeros(D),
        Sigma_init=jnp.eye(D),
        kl_history=True,
        score_history=True,
    )

    # ------------------------------------------------------------
    # 2. ADVI
    key, subk = jax.random.split(key)
    advi_mu, advi_Sigma, advi_kl_hist = advi(
        subk,
        target_log_prob=hierarchical_log_prob,
        target_score=hierarchical_score,
        target_mean=mu_star,
        target_cov=Sigma_star,
        T=T,
        B=B,
        lr=0.01,
        mu_init=jnp.zeros(D),
        Sigma_init=jnp.eye(D),
        kl_history=True,
    )

    # ------------------------------------------------------------
    # 3. GSM
    key, subk = jax.random.split(key)
    gsm_mu, gsm_Sigma, _ = gsm(
        subk,
        target_score=hierarchical_score,
        mu_init=jnp.zeros(D),
        Sigma_init=jnp.eye(D),
        T=T,
        B=1,
    )
    gsm_kl_hist = jnp.array([_kl_gaussian(mu_star, Sigma_star, gsm_mu, gsm_Sigma)])

    # ------------------------------------------------------------
    # Plotting
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(bam_kl_hist, label="BaM", linewidth=2)
    ax[0].plot(advi_kl_hist, label="ADVI", linewidth=2)
    ax[0].plot(gsm_kl_hist, label="GSM", linewidth=2)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel(r"$\mathrm{KL}(p\Vert q_t)$")
    ax[0].set_title("KL divergence")
    ax[0].legend()

    ax[1].plot(bam_score_hist, label="BaM", linewidth=2)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel(r"$\mathscr{D}(q_t;p)$")
    ax[1].set_title("Score‑based divergence (BaM only)")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("figures/hierarchical_results.png")
    plt.close()
    print("\nFigure generated: figures/hierarchical_results.png")


if __name__ == "__main__":
    run_hierarchical(T=200, B=20, seed=42)