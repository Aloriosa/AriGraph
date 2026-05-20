#!/usr/bin/env python3
"""
train.py – fine‑tune a pretrained DDPM on a 10‑shot target dataset
using similarity‑guided loss and adversarial noise selection
(DPMs‑ANT toy implementation).

The script:
  1. Loads CIFAR‑10 and selects a target class (default 'cat').
  2. Builds a 10‑shot target set and a large source set.
  3. Trains a binary classifier that distinguishes noised
     target vs. source images.
  4. Fine‑tunes the entire UNet of the pre‑trained DDPM
     using the similarity‑guided loss and an inner
     adversarial‑noise maximisation loop.
  5. Saves the fine‑tuned UNet to ./checkpoints/unet.pt and
     the classifier to ./checkpoints/classifier.pt.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Diffusers imports
from diffusers import UNet2DModel, DDPMScheduler

# ------------------------------------------------------------- #
# Reproducibility
# ------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------- #
# Data
# ------------------------------------------------------------- #
def load_cifar10():
    """Return train and test splits."""
    transform = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    return train, test

def get_target_indices(dataset, target_class=3, n=10):
    """Return indices of `n` images belonging to target_class."""
    indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
    random.shuffle(indices)
    return indices[:n]

# ------------------------------------------------------------- #
# Classifier
# ------------------------------------------------------------- #
class DomainClassifier(nn.Module):
    """
    Small CNN that predicts whether a 32×32 RGB image is from the
    target domain (label 1) or from the source domain (label 0).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),        # 1x1
            nn.Flatten(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)

def train_classifier(classifier, source_loader, target_loader, scheduler, device,
                     epochs=5, lr=1e-3, weight_decay=0.0):
    """Train classifier on noised images."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    classifier.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        # Mix batches from source and target
        for (src_imgs, _), (tgt_imgs, _) in zip(source_loader, target_loader):
            src_imgs = src_imgs.to(device)
            tgt_imgs = tgt_imgs.to(device)
            B = src_imgs.size(0)

            # Sample random timesteps
            t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device)

            # Add noise
            src_noisy = scheduler.add_noise(src_imgs, torch.randn_like(src_imgs), t)
            tgt_noisy = scheduler.add_noise(tgt_imgs, torch.randn_like(tgt_imgs), t)

            # Labels
            labels = torch.cat([torch.zeros(B, dtype=torch.long, device=device),
                                torch.ones(B, dtype=torch.long, device=device)], dim=0)
            imgs = torch.cat([src_noisy, tgt_noisy], dim=0)

            logits = classifier(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        epoch_loss /= len(source_loader.dataset) + len(target_loader.dataset)
        print(f"Classifier epoch {epoch+1:02d} loss: {epoch_loss:.4f}")

    classifier.eval()
    return classifier

# ------------------------------------------------------------- #
# Helper functions
# ------------------------------------------------------------- #
def get_sigma2(scheduler, t):
    """Return sigma^2 for a batch of timesteps."""
    return scheduler.sigmas[t] ** 2

# ------------------------------------------------------------- #
# Main training
# ------------------------------------------------------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CIFAR10
    train_ds, test_ds = load_cifar10()

    # Select target class (cat=3)
    target_class = 3
    target_idx = get_target_indices(train_ds, target_class=target_class, n=10)
    target_ds = Subset(train_ds, target_idx)

    # Create source dataset (exclude target indices to avoid leakage)
    source_idx = [i for i in range(len(train_ds)) if i not in target_idx]
    source_ds = Subset(train_ds, source_idx)

    # Dataloaders
    batch_size = 32
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Load pre‑trained DDPM (CIFAR10)
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

    # Normalise data to [-1,1] as expected by the UNet
    norm = T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    for d in [train_ds, target_ds, source_ds]:
        d.transform = T.Compose([d.transform, norm])

    # Train classifier
    classifier = DomainClassifier().to(device)
    classifier = train_classifier(classifier, source_loader, target_loader, scheduler, device)

    # Optimiser for UNet
    optimizer = optim.AdamW(unet.parameters(), lr=1e-5, weight_decay=0.0)

    # Hyper‑parameters
    epochs = 20
    gamma = 5.0          # similarity‑guidance weight
    J = 10               # inner adversarial steps
    omega = 0.02         # inner step size

    unet.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(target_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_imgs, _ in pbar:
            B = batch_imgs.size(0)
            batch_imgs = batch_imgs.to(device)

            # Sample random timesteps
            t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device)

            # Initialise adversarial noise
            eps = torch.randn_like(batch_imgs, requires_grad=True)

            # Inner maximisation
            for _ in range(J):
                xt = scheduler.add_noise(batch_imgs, eps, t)
                eps_pred = unet(xt, t).sample
                loss_noise = ((eps - eps_pred) ** 2).mean()
                grad_eps = torch.autograd.grad(loss_noise, eps, retain_graph=True)[0]
                eps = eps + omega * grad_eps
                eps = (eps - eps.mean()) / (eps.std() + 1e-6)

            # Final loss w/ similarity guidance
            xt_star = scheduler.add_noise(batch_imgs, eps, t)
            eps_pred = unet(xt_star, t).sample

            # Gradient of classifier log‑prob wrt xt_star
            xt_star.requires_grad_(True)
            logits = classifier(xt_star)
            # Target label = 1
            log_prob = F.log_softmax(logits, dim=1)[range(B), torch.ones(B, dtype=torch.long, device=device)]
            grad_logp = torch.autograd.grad(log_prob.sum(), xt_star, create_graph=True)[0]

            sigma2 = get_sigma2(scheduler, t).reshape(-1, 1, 1, 1)
            loss = ((eps - eps_pred - gamma * sigma2 * grad_logp) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * B
            pbar.set_postfix(loss=loss.item())

        epoch_loss /= len(target_loader.dataset)
        print(f"Epoch {epoch+1} – loss {epoch_loss:.4f}")

    # Save checkpoints
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(unet.state_dict(), "checkpoints/unet.pt")
    torch.save(classifier.state_dict(), "checkpoints/classifier.pt")
    torch.save({"seed": SEED, "device": str(device)}, "checkpoints/meta.pt")
    print("Training finished – checkpoints saved.")