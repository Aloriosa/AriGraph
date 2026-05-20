# Minimal training and sampling script for the stochastic interpolant model.

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from utils import alpha_beta
from models import VelocityNet
from torchdiffeq import odeint

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MASK_PROB = 0.3        # probability that a pixel is kept from the target
EMB_DIM = 256
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. Data
# --------------------------------------------------------------------------- #
transform = T.Compose([
    T.ToTensor(),          # [0,1]
    T.Normalize((0.5,), (0.5,)),  # [-1,1]
])

train_dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

# --------------------------------------------------------------------------- #
# 3. Model, optimizer, loss
# --------------------------------------------------------------------------- #
img_size = 28 * 28
velocity_net = VelocityNet(img_dim=img_size, emb_dim=EMB_DIM).to(DEVICE)
optimizer = optim.Adam(velocity_net.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss()

# --------------------------------------------------------------------------- #
# 4. Training loop
# --------------------------------------------------------------------------- #
print("Starting training on", DEVICE)
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, (x1, _) in enumerate(pbar):
        x1 = x1.to(DEVICE)          # [B,1,28,28]
        B = x1.shape[0]

        # Random binary mask
        mask = (torch.rand_like(x1) < MASK_PROB).float()
        # Base image: keep masked pixels from x1, fill rest with noise
        noise = torch.randn_like(x1)
        x0 = mask * x1 + (1 - mask) * noise

        # Sample times t ~ U(0,1)
        t = torch.rand(B, device=DEVICE)

        # Interpolant coefficients
        alpha, beta, dot_alpha, dot_beta = alpha_beta(t)

        # Compute I_t and dot_I_t
        I_t = alpha[:, None, None, None] * x0 + beta[:, None, None, None] * x1
        dot_I_t = dot_alpha[:, None, None, None] * x0 + dot_beta[:, None, None, None] * x1

        # Flatten
        I_t_flat = I_t.view(B, -1)
        dot_I_t_flat = dot_I_t.view(B, -1)

        # Forward pass
        pred = velocity_net(I_t_flat, t)

        loss = mse_loss(pred, dot_I_t_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

# --------------------------------------------------------------------------- #
# 5. Sampling
# --------------------------------------------------------------------------- #
print("\nSampling 5 images...")

def sample(num_samples=5, steps=100):
    """
    Sample images using probability‑flow ODE integration.
    """
    # Start from pure noise
    x0 = torch.randn(num_samples, 1, 28, 28, device=DEVICE)
    y0 = x0.view(num_samples, -1)

    # Define ODE function
    class ODEFunc(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, t, y):
            # y: [B, img_dim]
            t_tensor = torch.full((y.shape[0],), t, device=y.device)
            return self.net(y, t_tensor)

    ode_func = ODEFunc(velocity_net)

    t_span = torch.linspace(0.0, 1.0, steps, device=DEVICE)
    # DOPRI solver
    sol = odeint(ode_func, y0, t_span, rtol=1e-5, atol=1e-5, method="dopri5")
    y_T = sol[-1]
    x_T = y_T.view(num_samples, 1, 28, 28)

    # Clamp to [-1,1] and save as PNG
    x_T = torch.clamp(x_T, -1, 1)
    inv_norm = torchvision.transforms.Normalize((-1,), (2,))  # to [0,1]
    for i in range(num_samples):
        img = inv_norm(x_T[i]).detach().cpu()
        torchvision.utils.save_image(img, os.path.join(OUTPUT_DIR, f"sample_{i}.png"))
    print(f"Saved {num_samples} samples to {OUTPUT_DIR}/")

sample(num_samples=5, steps=100)

print("\nAll done. See the generated images in", OUTPUT_DIR)