# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm
import numpy as np

from tokenizer import Tokenizer
from simformer import Simformer
from utils import sample_forward, target_score, reverse_step

# --------------------------------------------------------------------------- #
#  Hyper‑parameters
# --------------------------------------------------------------------------- #
BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-4
EMBED_DIM = 64
NUM_LAYERS = 4
NHEAD = 4
DROPOUT = 0.1
SVD_STEPS = 50  # number of reverse steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
#  Data generation (Gaussian‑linear toy)
# --------------------------------------------------------------------------- #
def generate_gaussian_linear(batch_size: int):
    """
    Sample θ ~ N(0, 0.01 I), x ~ N(θ, 0.01 I)
    Returns tensors (θ, x) each of shape (B, 1)
    """
    theta = torch.randn(batch_size, 1) * math.sqrt(0.01)
    noise = torch.randn(batch_size, 1) * math.sqrt(0.01)
    x = theta + noise
    return theta, x

# --------------------------------------------------------------------------- #
#  Prepare training dataset
# --------------------------------------------------------------------------- #
NUM_SAMPLES = 100_000
theta_samples, x_samples = generate_gaussian_linear(NUM_SAMPLES)

# Tokenizer expects values of shape (B,N,1)
theta_vals = theta_samples.unsqueeze(1)  # (B,1,1)
x_vals = x_samples.unsqueeze(1)          # (B,1,1)

# identifiers: 0 -> θ, 1 -> x
identifiers = torch.cat([torch.zeros(NUM_SAMPLES, 1, dtype=torch.long),
                         torch.ones(NUM_SAMPLES, 1, dtype=torch.long)], dim=1)

# condition state will be set during training
dataset = TensorDataset(theta_vals, x_vals, identifiers)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------------------------------------------------------- #
#  Model setup
# --------------------------------------------------------------------------- #
tokenizer = Tokenizer(n_identifiers=2, embed_dim=EMBED_DIM).to(DEVICE)
simformer = Simformer(embed_dim=EMBED_DIM,
                      num_layers=NUM_LAYERS,
                      nhead=NHEAD,
                      dim_feedforward=EMBED_DIM * 2,
                      dropout=DROPOUT).to(DEVICE)

optimizer = optim.Adam(list(tokenizer.parameters()) + list(simformer.parameters()), lr=LR)
criterion = nn.MSELoss()

# --------------------------------------------------------------------------- #
#  Mask sampling utilities
# --------------------------------------------------------------------------- #
def sample_condition_mask(batch_size: int, N: int):
    """
    Randomly sample one of the following masks:
        - joint: all zeros
        - posterior: x conditioned (token 1)
        - likelihood: θ conditioned (token 0)
        - random: Bernoulli(0.3)
    Returns: (batch_size, N) bool tensor
    """
    choice = torch.randint(0, 4, (), device=DEVICE)
    if choice == 0:  # joint
        return torch.zeros(batch_size, N, dtype=torch.bool, device=DEVICE)
    elif choice == 1:  # posterior
        mask = torch.zeros(batch_size, N, dtype=torch.bool, device=DEVICE)
        mask[:, 1] = True
        return mask
    elif choice == 2:  # likelihood
        mask = torch.zeros(batch_size, N, dtype=torch.bool, device=DEVICE)
        mask[:, 0] = True
        return mask
    else:  # random
        return torch.rand(batch_size, N, device=DEVICE) < 0.3

# --------------------------------------------------------------------------- #
#  Training loop
# --------------------------------------------------------------------------- #
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for theta_batch, x_batch, id_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        theta_batch = theta_batch.to(DEVICE)
        x_batch = x_batch.to(DEVICE)
        id_batch = id_batch.to(DEVICE)

        # Stack into (B,N,1)
        values = torch.cat([theta_batch, x_batch], dim=1)  # (B,2,1)
        # Random condition states
        cond_state = torch.zeros_like(id_batch, dtype=torch.long).to(DEVICE)

        # Sample noise level t in [0,1]
        t = torch.rand(theta_batch.size(0), device=DEVICE)
        # Forward diffusion
        x_t, eps, sigma = sample_forward(values, t, DEVICE)

        # Build tokens for x_t
        tokens = tokenizer(id_batch, x_t, cond_state)

        # Predict scores
        scores = simformer(tokens)  # (B,N,embed)

        # Compute analytical score
        tgt = target_score(values, x_t, sigma)  # (B,N,embed)

        # Sample condition mask
        mask = sample_condition_mask(theta_batch.size(0), 2)  # (B,2)

        # Compute loss only on latent tokens
        loss = criterion(scores * (~mask).unsqueeze(-1), tgt * (~mask).unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * theta_batch.size(0)

    epoch_loss /= len(loader.dataset)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1} – Training loss: {epoch_loss:.4f}")

# --------------------------------------------------------------------------- #
#  Posterior sampling for a random observation
# --------------------------------------------------------------------------- #
# Generate a single random observation x_obs
theta_true, x_obs = generate_gaussian_linear(1)
theta_true = theta_true.to(DEVICE)
x_obs = x_obs.to(DEVICE)

# Condition mask: x conditioned
cond_mask = torch.tensor([[False, True]], device=DEVICE)  # θ unconditioned, x conditioned
cond_values = torch.cat([torch.zeros_like(theta_true), x_obs], dim=1)  # (1,2,1)

# Reverse diffusion
x_t = torch.randn(1, 2, EMBED_DIM, device=DEVICE) * sigma_t(torch.tensor(1.0, device=DEVICE))
t = torch.tensor([1.0], device=DEVICE)
dt = 1.0 / SVD_STEPS

samples = []
for _ in range(SVD_STEPS):
    x_t = reverse_step(
        model=simformer,
        x_t=x_t,
        t=t,
        dt=dt,
        cond_mask=cond_mask,
        cond_values=cond_values,
        tokenizer=tokenizer,
        device=DEVICE,
    )
    t = t - dt
    samples.append(x_t.detach().cpu())

# Take last state as sample
posterior_samples = samples[-1].cpu().numpy()  # shape (1,2,embed_dim)

# We only keep the first token (θ)
theta_samples = posterior_samples[0, 0, :]  # (embed_dim,)

# For evaluation, we compare the mean and covariance of the predicted θ
# to the analytical posterior (mean=0.5*x_obs, var=0.005*I)
pred_mean = theta_samples.mean()
pred_std = theta_samples.std()
analytical_mean = 0.5 * x_obs.item()
analytical_std = math.sqrt(0.005)

print("\n=== Posterior sampling result ===")
print(f"Analytical mean : {analytical_mean:.4f}")
print(f"Predicted mean  : {pred_mean:.4f}")
print(f"Analytical std  : {analytical_std:.4f}")
print(f"Predicted std   : {pred_std:.4f}")