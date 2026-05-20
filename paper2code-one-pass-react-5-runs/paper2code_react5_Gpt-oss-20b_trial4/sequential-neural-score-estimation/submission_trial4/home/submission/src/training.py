import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import GaussianLinearDataset
from .model import ScoreNetwork
from .utils import log_posterior_gaussian, hpr_threshold

def train_round(dataset, model, cfg, device):
    """
    Train the score network on the provided dataset for one round.
    """
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    # Simple constant weighting; the paper uses λ_t but a constant suffices for the toy benchmark
    lambda_t = 1.0
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        for theta_t, x_obs, t, target in loader:
            theta_t = theta_t.to(device)
            x_obs = x_obs.to(device)
            t = t.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(theta_t, x_obs, t)
            loss = ((pred - target) ** 2).mean() * lambda_t
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * theta_t.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {epoch_loss:.6f}")
    return model

def build_truncated_prior(samples, log_probs, epsilon):
    """
    Build a truncated prior by selecting samples whose log‑posterior
    is above the (1‑ε) quantile.
    """
    threshold = hpr_threshold(log_probs, epsilon)
    mask = log_probs >= threshold
    return samples[mask]

def generate_dataset_from_prior(prior_samples, x_obs, cfg, device):
    """
    Draw θ₀ from prior_samples and generate a dataset for training.
    """
    # Uniformly sample diffusion times
    t = torch.rand(prior_samples.shape[0], device=device)
    dataset = GaussianLinearDataset(prior_samples, x_obs, t, cfg)
    return dataset

def sequential_training(cfg):
    """
    Main sequential loop implementing TSNPSE for the Gaussian‑linear benchmark.
    """
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Initial prior samples
    d = 1  # dimensionality
    prior_std = cfg["prior_std"]
    prior_samples = torch.randn(cfg["sim_per_round"], d, device=device) * prior_std
    x_obs = torch.tensor([cfg["x_observation"]], device=device)

    # Model
    model = ScoreNetwork(d, d).to(device)

    # Sequential loop
    for r in range(1, cfg["rounds"] + 1):
        print(f"\n=== Round {r} ===")
        # Build dataset from current prior
        dataset = generate_dataset_from_prior(prior_samples, x_obs, cfg, device)
        # Train
        model = train_round(dataset, model, cfg, device)

        # After training, sample from approximate posterior to update prior
        # 1. Draw many samples from posterior via probability‑flow ODE
        from .sampling import sample_posterior
        posterior_samples = sample_posterior(model, x_obs, cfg, device,
                                            num_samples=20000,
                                            sample_steps=cfg["sample_steps"])
        # 2. Compute log‑posterior for these samples
        with torch.no_grad():
            logp = log_posterior_gaussian(posterior_samples,
                                          cfg["x_observation"],
                                          cfg["prior_std"],
                                          cfg["simulator_std"],
                                          cfg.get("prior_mean", 0.0))
        # 3. Build truncated prior for next round
        if r < cfg["rounds"]:
            prior_samples = build_truncated_prior(posterior_samples, logp, cfg["epsilon_trunc"])
            print(f"Truncated prior size: {prior_samples.shape[0]}")
            # if too few samples, fall back to drawing from prior again
            if prior_samples.shape[0] < cfg["sim_per_round"] // 2:
                print("Too few samples in truncated prior, resampling from full prior.")
                prior_samples = torch.randn(cfg["sim_per_round"], d, device=device) * prior_std

    return model, x_obs