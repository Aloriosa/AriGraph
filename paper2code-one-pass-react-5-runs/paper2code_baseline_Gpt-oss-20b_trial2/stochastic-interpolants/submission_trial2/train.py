#!/usr/bin/env python3
"""
Training script for the velocity model using the quadratic loss from the paper.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VelocityNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Model
    net = VelocityNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for xb, _ in train_dl:
            xb = xb.view(xb.size(0), -1).to(device)  # (B, 784)
            # Sample t ~ Uniform(0,1)
            t = torch.rand(xb.size(0), device=device)

            # Coupling: x0 = x1 + sigma * z
            sigma = 0.1
            z = torch.randn_like(xb)
            x0 = xb + sigma * z
            x1 = xb

            # Interpolant
            alpha = 1.0 - t
            beta = t
            I_t = alpha.unsqueeze(1) * x0 + beta.unsqueeze(1) * x1
            dotI = -x0 + x1  # derivative of I_t

            # Velocity prediction
            b_pred = net(I_t, t)

            # Loss: mean squared error to dotI
            loss = ((b_pred - dotI) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_dl.dataset)
        print(f"Epoch {epoch}/{args.epochs}  Loss={epoch_loss:.6f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(net.state_dict(), "checkpoints/velocity.pth")
    print("Model saved to checkpoints/velocity.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train velocity model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    train(args)