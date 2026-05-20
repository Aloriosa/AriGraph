#!/usr/bin/env python3
"""
Training script for the 1‑D wave PDE PINN.

The training proceeds in two phases:
  1. Adam optimizer for 5000 iterations.
  2. L‑BFGS optimizer for 2000 iterations.

At the end the script prints the final loss and the L2 relative error.
"""

import argparse
import os
import time
import numpy as np
import torch
from torch import optim

from src.pinn import MLP, loss_fn, l2re, build_points

torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)


def train(
    model,
    x_res,
    x_bc,
    x_ic,
    device,
    adam_lr=1e-3,
    adam_steps=5000,
    lbfgs_steps=2000,
):
    model.to(device)

    # Adam phase
    optimizer = optim.Adam(model.parameters(), lr=adam_lr)
    print("Starting Adam phase")
    t0 = time.time()
    for step in range(1, adam_steps + 1):
        optimizer.zero_grad()
        loss = loss_fn(model, x_res, x_bc, x_ic)
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == 1:
            print(f"  Adam step {step:5d}  loss={loss.item():.6e}")

    print(f"Adam phase finished in {time.time() - t0:.1f}s")

    # L‑BFGS phase
    # Note: LBFGS requires closure that returns loss.
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_steps, line_search_fn="strong_wolfe")
    print("Starting L‑BFGS phase")

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model, x_res, x_bc, x_ic)
        loss.backward()
        return loss

    t0 = time.time()
    optimizer.step(closure)
    print(f"L‑BFGS phase finished in {time.time() - t0:.1f}s")

    # Final loss
    final_loss = loss_fn(model, x_res, x_bc, x_ic).item()
    return final_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build data
    x_res, x_bc, x_ic = build_points(device=device)

    # Build model
    model = MLP(width=args.width)

    # Train
    final_loss = train(
        model,
        x_res,
        x_bc,
        x_ic,
        device,
        adam_lr=args.adam_lr,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
    )

    # Evaluate L2RE on full grid
    grid_x = torch.linspace(0, 1, 255, device=device)
    grid_t = torch.linspace(0, 1, 100, device=device)
    l2_error = l2re(model, grid_x, grid_t, device)

    print("\n=== Final Results ===")
    print(f"Final loss: {final_loss:.6e}")
    print(f"L2 relative error: {l2_error:.6e}")

    # Save results
    os.makedirs("output", exist_ok=True)
    with open("output/results.txt", "w") as f:
        f.write(f"final_loss={final_loss}\n")
        f.write(f"l2re={l2_error}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN on 1‑D wave equation")
    parser.add_argument("--width", type=int, default=50, help="hidden layer width")
    parser.add_argument("--adam_lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--adam_steps", type=int, default=5000, help="Adam iterations")
    parser.add_argument("--lbfgs_steps", type=int, default=2000, help="L‑BFGS iterations")
    args = parser.parse_args()
    main(args)