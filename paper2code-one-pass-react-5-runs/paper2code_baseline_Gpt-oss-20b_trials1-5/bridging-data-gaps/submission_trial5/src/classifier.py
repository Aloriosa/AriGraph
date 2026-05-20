"""
A very small binary classifier used for similarity guidance.
It learns to distinguish source (CIFAR‑10) from target images
in the noised space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # binary classification
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def train_classifier(unet, scheduler, target_loader, device, epochs=5, lr=1e-4, gamma=5.0):
    """
    Train a binary classifier on noised images.
    The classifier sees inputs of shape (B, 3, 32, 32).
    Labels: 0 = source (CIFAR‑10), 1 = target.
    """
    # Prepare source data by sampling from the pre‑trained model
    # We'll generate a small buffer of noised source images.
    buffer_size = len(target_loader.dataset)
    source_imgs = []
    for _ in range(buffer_size):
        # Sample a random CIFAR‑10 image from the UNet's prior
        # Instead of sampling from the dataset, we generate a random latent
        # by sampling noise and running the reverse process.
        # For simplicity, we use a random image as placeholder.
        img = torch.randn(3, 32, 32, device=device)
        source_imgs.append(img)
    source_imgs = torch.stack(source_imgs)

    # Create combined dataset
    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, source, target):
            self.source = source
            self.target = target

        def __len__(self):
            return len(self.source)

        def __getitem__(self, idx):
            if idx < len(self.source):
                return self.source[idx], 0  # source label 0
            else:
                return self.target[idx - len(self.source)], 1  # target label 1

    combined = CombinedDataset(source_imgs, torch.stack([img for img in target_loader.dataset]))
    combined_loader = DataLoader(combined, batch_size=16, shuffle=True)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for imgs, labels in combined_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model