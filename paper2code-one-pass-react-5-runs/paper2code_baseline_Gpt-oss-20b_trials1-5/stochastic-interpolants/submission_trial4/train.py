import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SimpleUNet
from utils import alpha_beta, dot_alpha_beta

# ------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 128
EPOCHS = 3          # keep training short for demo
LR = 2e-4
SIGMA = 0.1         # noise level for base density

# ------------------------------------------------------------------
# 2. Data
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),          # 0‑1 tensor
    transforms.Normalize((0.5,), (0.5,))  # mean‑zero, std‑half
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------------------------------------------
# 3. Model
# ------------------------------------------------------------------
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------------------------
# 4. Training loop (Algorithm 1)
# ------------------------------------------------------------------
def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0.0
    for xb, _ in train_loader:
        xb = xb.to(device)          # target samples x1
        B = xb.shape[0]

        # Sample noise z and time t
        z = torch.randn(B, 1, 28, 28, device=device)
        t = torch.rand(B, device=device)

        # Data‑dependent coupling: base x0 = x1 + sigma * z
        x0 = xb + SIGMA * z

        # Interpolant coefficients
        alpha_t, beta_t = alpha_beta(t)          # (B,)
        dot_alpha_t, dot_beta_t = dot_alpha_beta(t)

        # Expand for broadcasting
        alpha_t = alpha_t[:, None, None, None]
        beta_t = beta_t[:, None, None, None]
        dot_alpha_t = dot_alpha_t[:, None, None, None]
        dot_beta_t = dot_beta_t[:, None, None, None]

        # Interpolant I_t
        I_t = alpha_t * x0 + beta_t * xb

        # Derivative of I_t
        dotI_t = dot_alpha_t * x0 + dot_beta_t * xb

        # Predict velocity
        b_pred = model(I_t, t)

        # Loss: L_b = E[|b|^2 - 2 dotI · b]
        loss = torch.mean(
            (b_pred ** 2).sum([1,2,3]) - 2 * (dotI_t * b_pred).sum([1,2,3])
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * B

    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}")

for epoch in range(EPOCHS):
    train_one_epoch(epoch)

# Save model
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/velocity.pth')
print("Model saved to checkpoints/velocity.pth")