# train_inpainting.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import create_random_mask, sample_base
from model import VelocityMLP

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, _) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"):
        data = data.to(device)  # (B,1,28,28), values in [0,1]
        B = data.size(0)

        # Sample random mask for each image
        mask = create_random_mask((1, 28, 28)).to(device)  # (1,28,28)
        mask = mask.repeat(B, 1, 1, 1)  # (B,1,28,28)

        # Generate base samples
        x0 = sample_base(data, mask, noise_std=0.1)

        # Sample random time t in [0,1]
        t = torch.rand(B, device=device)

        # Interpolant coefficients: alpha = 1 - t, beta = t
        alpha = 1.0 - t
        beta = t

        # I_t = alpha * x0 + beta * x1 (gamma=0)
        I_t = alpha[:, None, None, None] * x0 + beta[:, None, None, None] * data

        # Derivative of I_t w.r.t t: dotI_t = -x0 + x1
        dotI_t = -x0 + data

        # Predict velocity b_t(I_t, t)
        b_hat = model(I_t, t)

        # Loss: E[|b_hat|^2 - 2 * dotI_t · b_hat]
        b_hat_sq = (b_hat ** 2).sum(dim=[1,2,3])  # (B,)
        prod = (dotI_t * b_hat).sum(dim=[1,2,3])  # (B,)
        loss = (b_hat_sq - 2 * prod).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B

    epoch_loss = total_loss / len(loader.dataset)
    return epoch_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),                     # [0,1]
    ])
    train_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=".", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model, optimizer
    model = VelocityMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch} – loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), args.output)
            print(f"  → new best model saved (loss={best_loss:.4f})")

    print(f"Training finished – best loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", type=str, default="model.pt")
    args = parser.parse_args()
    main(args)