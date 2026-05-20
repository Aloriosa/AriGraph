#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .utils import simulator, forward_noising, compute_analytical_score, sample_posterior
from .model import ScoreMLP
from .diffusion import sigma_t

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_EPOCHS = 5   # small for toy demo
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load observed data
obs_path = os.path.join('data', 'observed_data.npy')
x_obs = torch.from_numpy(np.load(obs_path)).float().to(DEVICE)  # shape (2,)
x_obs = x_obs.unsqueeze(0)  # shape (1, 2)

# Hyperparameters
dim = 2
num_rounds = 2
samples_per_round = 2000
num_noise_steps = 10

# Model
model = ScoreMLP(dim_theta=dim, dim_x=dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_round(round_idx, prior_samples):
    """
    Train one sequential round.
    Args:
        prior_samples: Tensor of shape (N, dim) sampled from the current proposal prior.
    Returns:
        Updated model (in place).
    """
    # Generate synthetic data for each prior sample
    x_batch = simulator(prior_samples)  # shape (N, dim)

    # For each sample, pick a random time t in [0,1]
    t_batch = torch.rand(prior_samples.shape[0], device=DEVICE)

    # Forward noising
    theta_t, sigma = forward_noising(prior_samples, t_batch)
    # Compute analytical score
    target_score = compute_analytical_score(prior_samples, theta_t, sigma)

    # Prepare dataset
    dataset = TensorDataset(theta_t, x_batch, t_batch, target_score)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for theta_t_batch, x_batch_batch, t_batch_batch, target_batch in tqdm(loader, desc=f'Round {round_idx+1} Epoch {epoch+1}', leave=False):
            theta_t_batch = theta_t_batch.to(DEVICE)
            x_batch_batch = x_batch_batch.to(DEVICE)
            t_batch_batch = t_batch_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            pred_score = model(theta_t_batch, x_batch_batch, t_batch_batch)
            loss = F.mse_loss(pred_score, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * theta_t_batch.size(0)

        epoch_loss /= len(loader.dataset)
        print(f'Round {round_idx+1} Epoch {epoch+1} Loss: {epoch_loss:.4f}')
    return

def sample_from_prior(round_idx, num_samples):
    """
    Sample from the current truncated prior.
    For the toy demo we simply sample from the base prior (standard normal).
    """
    return torch.randn(num_samples, dim, device=DEVICE)

def main():
    # Initial prior is standard normal
    prior = torch.randn(samples_per_round, dim, device=DEVICE)
    for r in range(num_rounds):
        print(f'=== Starting round {r+1} ===')
        train_round(r, prior)
        # After training, use the model to sample from posterior
        posterior_samples = sample_posterior(model, num_samples=1000, num_steps=num_noise_steps, device=DEVICE)
        # Use these samples as the new prior for next round (truncated by simple rejection)
        # Here we just keep them (no actual truncation for simplicity)
        prior = posterior_samples

    # Save final posterior samples
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/posterior_samples.npy', posterior_samples.cpu().numpy())
    print('Posterior samples saved to outputs/posterior_samples.npy')

if __name__ == '__main__':
    main()