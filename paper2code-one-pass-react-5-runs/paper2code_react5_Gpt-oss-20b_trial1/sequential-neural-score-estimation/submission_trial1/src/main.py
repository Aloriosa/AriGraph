"""
Minimal reproducible implementation of TSNPSE on the Two‑Moons benchmark.
The code follows the algorithm described in the paper:
  1. Simulate data from the prior.
  2. Train a conditional score network.
  3. Sample from the learned posterior using the probability‑flow ODE.
  4. Use the posterior samples as a truncated proposal for the next round.
  5. Repeat for a fixed number of rounds.
"""

import os
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.simulator import (
    sample_prior_uniform,
    simulate_two_moons,
    X_OBS,
)
from src.model import ScoreMLP
from src.utils import (
    ensure_dir,
    save_numpy,
    SimulatorDataset,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Simulation budget and sequential settings
N_SIMULATIONS = 10_000   # total simulated pairs
N_ROUNDS = 10
M_PER_ROUND = N_SIMULATIONS // N_ROUNDS  # simulations per round

# Training settings
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3

# Diffusion settings (VE SDE)
SIGMA_MIN = 0.01
SIGMA_MAX = 2.0
T_MAX = 1.0

# Sampling settings
N_POSTERIOR_SAMPLES = 5_000
ODE_DT = 1e-3

# Results directory
RESULTS_DIR = "results"
ensure_dir(RESULTS_DIR)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    σ_t for the VE SDE: σ_t = σ_min · (σ_max/σ_min)^t
    """
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t


def forward_diffusion(theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Sample θ_t ~ N(θ, σ_t² I)
    """
    sigma_t = noise_schedule(t)
    return theta + torch.randn_like(theta) * sigma_t


def backward_ode(theta_t: torch.Tensor,
                 x_obs: torch.Tensor,
                 score_net: torch.nn.Module,
                 dt: float = ODE_DT) -> torch.Tensor:
    """
    Integrate the probability‑flow ODE backwards from t = T to t = 0.

    dθ/dt = -0.5 σ_t² ⋅ score(θ_t, x_obs, t)
    """
    theta = theta_t.clone()
    t = torch.full((theta.shape[0],), T_MAX, device=theta.device)

    while t.max() > 0:
        sigma_t = noise_schedule(t)
        # Predict score
        score = score_net(theta, x_obs, t)
        drift = -0.5 * sigma_t ** 2 * score
        theta = theta + drift * dt
        t = t - dt
        t = torch.clamp(t, min=0.0)

    return theta


def train_score_network(dataset: SimulatorDataset) -> ScoreMLP:
    """
    Train the score network on the entire dataset.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ScoreMLP(dim_theta=2, dim_x=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for theta, x in loader:
            theta = theta.to(DEVICE)
            x = x.to(DEVICE)

            # Sample time t uniformly from (0, 1]
            t = torch.rand(theta.shape[0], device=DEVICE) * T_MAX
            t = t.clamp(min=1e-6)  # avoid t=0

            # Forward diffusion
            theta_t = forward_diffusion(theta, t)

            # Target score: ∇θ log p_{t|0}(θ_t|θ) = (θ − θ_t)/σ_t²
            sigma_t = noise_schedule(t)
            target_score = (theta - theta_t) / sigma_t ** 2

            # Predict score
            pred_score = model(theta_t, x, t)

            loss = criterion(pred_score, target_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * theta.shape[0]

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} – Loss: {epoch_loss:.6f}")

    return model


def sample_from_prior(n: int, prior_sampler) -> np.ndarray:
    """
    Sample n parameters from the given prior sampler.
    prior_sampler: function returning np.ndarray of shape (n, 2)
    """
    return prior_sampler(n)


def sample_posterior(score_net: torch.nn.Module,
                     n_samples: int) -> np.ndarray:
    """
    Sample from the learned posterior using the probability‑flow ODE.
    """
    # Start from reference distribution N(0, σ_T² I)
    theta = torch.randn(n_samples, 2, device=DEVICE) * SIGMA_MAX
    x_obs = torch.from_numpy(X_OBS).unsqueeze(0).repeat(n_samples, 1).to(DEVICE)

    theta = backward_ode(theta, x_obs, score_net, dt=ODE_DT)

    return theta.cpu().numpy()


def posterior_predictive_stats(theta_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predictive mean and variance of the simulator output
    given posterior samples of θ.
    """
    xs = simulate_two_moons(theta_samples)
    pred_mean = xs.mean(axis=0)
    pred_var = xs.var(axis=0)
    return pred_mean, pred_var


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #

def main():
    # Store all simulated data across rounds
    all_thetas = []
    all_xs = []

    # Posterior samples from the previous round (used as truncated prior)
    prev_posterior_samples = None

    for r in range(1, N_ROUNDS + 1):
        print(f"\n=== Round {r} ===")

        # 1. Sample parameters from the prior (or truncated prior)
        if r == 1:
            thetas = sample_from_prior(M_PER_ROUND, sample_prior_uniform)
        else:
            # Use posterior samples from previous round as prior (with replacement)
            idx = np.random.choice(prev_posterior_samples.shape[0],
                                   size=M_PER_ROUND,
                                   replace=True)
            thetas = prev_posterior_samples[idx]

        # 2. Simulate data
        xs = simulate_two_moons(thetas)

        # 3. Append to the global dataset
        all_thetas.append(thetas)
        all_xs.append(xs)

        # 4. Create dataset and train score network
        dataset = SimulatorDataset(
            thetas=np.concatenate(all_thetas, axis=0),
            xs=np.concatenate(all_xs, axis=0)
        )
        print("  Training score network on accumulated data...")
        score_net = train_score_network(dataset)

        # 5. Sample from the learned posterior
        print("  Sampling from posterior...")
        posterior_samples = sample_posterior(score_net, N_POSTERIOR_SAMPLES)
        prev_posterior_samples = posterior_samples  # for next round

        # 6. Save intermediate posterior samples
        np.save(os.path.join(RESULTS_DIR, f"posterior_round_{r}.npy"),
                posterior_samples)

        # 7. Compute predictive statistics
        pred_mean, pred_var = posterior_predictive_stats(posterior_samples)
        print(f"  Predictive mean: {pred_mean}")
        print(f"  Predictive var : {pred_var}")

    # ----------------------------------------------------------------------- #
    # Final evaluation
    # ----------------------------------------------------------------------- #
    final_posterior_samples = prev_posterior_samples
    np.save(os.path.join(RESULTS_DIR, "posterior_samples.npy"),
            final_posterior_samples)

    # Compute summary statistics
    mean = final_posterior_samples.mean(axis=0)
    var = final_posterior_samples.var(axis=0)
    pred_mean, pred_var = posterior_predictive_stats(final_posterior_samples)

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Posterior mean (approx.): {mean}\n")
        f.write(f"Posterior variance (approx.): {var}\n")
        f.write(f"Predictive mean : {pred_mean}\n")
        f.write(f"Predictive var  : {pred_var}\n")
        f.write(f"Number of posterior samples: {final_posterior_samples.shape[0]}\n")
        f.write(f"Observed data x_obs: {X_OBS}\n")

    print(f"\n=== Results written to {RESULTS_DIR} ===")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()