#!/usr/bin/env python3
import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_seed, InpaintDataset, SuperResDataset, psnr
from model import SimpleUNet

# ------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------
def to_device(t, device):
    if isinstance(t, (list, tuple)):
        return [to_device(e, device) for e in t]
    return t.to(device)

def train_one_epoch(model, loader, optimizer, device, time_dim):
    model.train()
    epoch_loss = 0.0
    for x0, mask, x1 in tqdm(loader, desc="Batch", leave=False):
        x0, mask, x1 = to_device([x0, mask, x1], device)
        batch_size = x0.shape[0]

        # Random time in (0,1)
        t = torch.rand(batch_size, device=device)

        # Interpolant I_t = (1-t)*x0 + t*x1
        I_t = (1.0 - t).unsqueeze(1) * x0 + t.unsqueeze(1) * x1
        dotI_t = -x0 + x1  # derivative

        # Conditioning
        cond = mask if isinstance(loader.dataset, InpaintDataset) else x0[:, :3, :, :]  # low-res upsampled for superres

        # Forward pass
        b_pred = model(I_t, t, cond)
        loss = (b_pred.pow(2) - 2 * dotI_t * b_pred).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_size

    return epoch_loss / len(loader.dataset)

def evaluate_psnr(model, loader, device, time_dim):
    model.eval()
    psnr_vals = []
    with torch.no_grad():
        for x0, mask, x1 in tqdm(loader, desc="Eval", leave=False):
            x0, mask, x1 = to_device([x0, mask, x1], device)
            batch_size = x0.shape[0]
            # Euler integration with torchdiffeq
            from torchdiffeq import odeint

            def ode_func(t, y):
                cond = mask if isinstance(loader.dataset, InpaintDataset) else x0[:, :3, :, :]
                return model(y, t, cond)

            t_span = torch.linspace(0.0, 1.0, steps=100, device=device)
            y = odeint(ode_func, x0, t_span, method="dopri5")
            y_T = y[-1]
            y_T = torch.clamp(y_T, 0.0, 1.0)
            psnr_vals.append(psnr(y_T, x1))
    return float(np.mean(psnr_vals))

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train velocity model for conditional generation")
    parser.add_argument("--task", type=str, choices=["inpaint", "superres"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.task == "inpaint":
        train_ds = InpaintDataset(split="train", transform=None)
        val_ds = InpaintDataset(split="test", transform=None)
    else:
        train_ds = SuperResDataset(split="train", transform=None)
        val_ds = SuperResDataset(split="test", transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # Determine conditioning channels
    cond_ch = 1 if args.task == "inpaint" else 3
    model = SimpleUNet(in_channels=3, cond_channels=cond_ch, time_dim=64,
                       base_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr = -float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device, time_dim=64)
        elapsed = time.time() - start
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Time {elapsed:.1f}s")

        # Estimate PSNR on validation set
        val_psnr = evaluate_psnr(model, val_loader, device, time_dim=64)
        print(f"PSNR @ epoch {epoch}: {val_psnr:.2f} dB")

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pt")

    # Final PSNR
    final_psnr = evaluate_psnr(model, val_loader, device, time_dim=64)
    with open(Path(args.output_dir) / "psnr.txt", "w") as f:
        f.write(f"{final_psnr:.2f}\n")
    print(f"Final PSNR: {final_psnr:.2f} dB")

if __name__ == "__main__":
    main()