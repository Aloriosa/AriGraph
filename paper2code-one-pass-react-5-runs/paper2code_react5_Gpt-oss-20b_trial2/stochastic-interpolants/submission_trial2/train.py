import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import torchdiffeq

from model import UNet, TimeEmbedding


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loader(batch_size: int, train: bool = True) -> torch.utils.data.DataLoader:
    """CIFAR‑10 loader with standard normalisation."""
    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=train, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        drop_last=True,
    )
    return loader


# ----------------------------------------------------------------------
# Coupling helpers
# ----------------------------------------------------------------------
def downsample(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(img, size=size, mode="bilinear", align_corners=False)


def generate_mask(batch: int, H: int, W: int, patch_size: int = 8, p: float = 0.3) -> torch.Tensor:
    """
    Generate a random binary mask with square patches of size patch_size.
    1 = known pixel, 0 = missing pixel.
    """
    mask = torch.ones(batch, 1, H, W, device=img.device)
    grid_h = H // patch_size
    grid_w = W // patch_size
    for i in range(batch):
        for gh in range(grid_h):
            for gw in range(grid_w):
                if random.random() < p:
                    mask[i, 0,
                          gh * patch_size : (gh + 1) * patch_size,
                          gw * patch_size : (gw + 1) * patch_size] = 0
    return mask


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(
    task: str,
    batch_size: int,
    epochs: int,
    lr: float,
    sigma: float,
    save_dir: str,
    device: torch.device,
) -> None:
    loader = get_loader(batch_size, train=True)
    ckpt_dir = Path(save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Determine conditioning channels
    cond_channels = 0
    if task == "inpaint":
        cond_channels = 1
    elif task == "sr":
        cond_channels = 3

    model = UNet(base_channels=3, cond_channels=cond_channels, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
        for i, (x1, _) in pbar:
            x1 = x1.to(device)  # (B,3,32,32)
            B = x1.size(0)

            # ---------- conditioning and coupling ----------
            if task == "sr":
                low_res = downsample(x1, size=(8, 8))
                low_res_up = downsample(low_res, size=(32, 32))
                cond = low_res_up
                m_x1 = low_res_up
            elif task == "inpaint":
                mask = generate_mask(B, 32, 32, patch_size=8, p=0.3).to(device)
                cond = mask
                m_x1 = mask * x1
            else:  # baseline or other tasks
                cond = None
                m_x1 = None

            # ---------- stochastic interpolant ----------
            eps = torch.randn_like(x1).to(device)
            z = torch.randn_like(x1).to(device)

            if task == "baseline":
                x0 = torch.randn_like(x1).to(device)  # independent Gaussian
            else:
                x0 = m_x1 + sigma * eps

            # time samples
            t = torch.rand(B, device=device)  # (B,)
            eps_t = 1e-4
            t = torch.clamp(t, eps_t, 1.0 - eps_t)
            a = 1.0 - t
            b = t
            gamma = torch.sqrt(2 * t * (1 - t))
            dot_gamma = (1 - 2 * t) / torch.sqrt(2 * t * (1 - t))

            it = a[:, None, None, None] * x0 + b[:, None, None, None] * x1 + gamma[:, None, None, None] * z
            it_dot = -x0 + x1 + dot_gamma[:, None, None, None] * z

            # ---------- forward ----------
            if cond is not None:
                inp = torch.cat([it, cond], dim=1)
            else:
                inp = it
            with torch.cuda.amp.autocast():
                pred = model(inp, t)
                loss = ((pred - it_dot) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch}] Average loss: {avg_loss:.6f}")

        # checkpoint
        ckpt_path = ckpt_dir / f"model_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "sigma": sigma,
                "task": task,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")

        # keep best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "sigma": sigma,
                    "task": task,
                },
                ckpt_dir / "model_best.pt",
            )
    print("Training finished. Best model saved to model_best.pt")


# ----------------------------------------------------------------------
# Sampling (probability‑flow ODE)
# ----------------------------------------------------------------------
def sample(
    ckpt_path: str,
    samples: int,
    sample_batch: int,
    sigma: float,
    task: str,
    device: torch.device,
    out_dir: str,
) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    cond_channels = 0
    if task == "inpaint":
        cond_channels = 1
    elif task == "sr":
        cond_channels = 3

    model = UNet(base_channels=3, cond_channels=cond_channels, out_channels=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = get_loader(sample_batch, train=False)
    generated = []

    with torch.no_grad():
        for batch_idx, (x1, _) in enumerate(loader):
            x1 = x1.to(device)
            B = x1.size(0)

            # ---------- conditioning ----------
            if task == "sr":
                low_res = downsample(x1, size=(8, 8))
                low_res_up = downsample(low_res, size=(32, 32))
                cond = low_res_up
                m_x1 = low_res_up
            elif task == "inpaint":
                mask = generate_mask(B, 32, 32, patch_size=8, p=0.3).to(device)
                cond = mask
                m_x1 = mask * x1
            else:
                cond = None
                m_x1 = None

            # ---------- base sample ----------
            eps = torch.randn_like(x1).to(device)
            if task == "baseline":
                x0 = torch.randn_like(x1).to(device)
            else:
                x0 = m_x1 + sigma * eps

            # ---------- ODE integration ----------
            def ode_func(t, y):
                y_in = y
                if cond is not None:
                    y_in = torch.cat([y, cond], dim=1)
                return model(y_in, torch.full((B,), t, device=device))

            t_span = torch.tensor([0.0, 1.0], device=device)
            # 50 steps Euler is enough for a small toy example
            y1 = torchdiffeq.odeint(
                ode_func,
                x0,
                t_span,
                method="euler",
                rtol=1e-5,
                atol=1e-5,
            )[-1]  # final state

            generated.append(y1.cpu())

            if len(generated) * sample_batch >= samples:
                break

    generated = torch.cat(generated, dim=0)[:samples]  # (S,3,32,32)
    generated = generated.clamp(-1, 1)

    # Convert to [0,255] images and save
    inv_norm = T.Normalize(mean=(-1, -1, -1), std=(2, 2, 2))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(generated):
        img = inv_norm(img).permute(1, 2, 0)  # (H,W,3)
        torchvision.utils.save_image(
            img, out_path / f"sample_{i:04d}.png", normalize=True
        )

    print(f"Saved {len(generated)} samples to {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stochastic interpolants demo")
    parser.add_argument("--task", type=str, default="inpaint",
                        help="Task: 'baseline', 'inpaint', 'sr'")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--sigma_base", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=2)
    parser.add_argument("--sample", action="store_true",
                        help="Run sampling only (requires --ckpt)")
    parser.add_argument("--ckpt", type=str, default="model_best.pt")
    parser.add_argument("--sample_batch", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="model")
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.sample:
        sample(
            ckpt_path=args.ckpt,
            samples=args.num_samples,
            sample_batch=args.sample_batch,
            sigma=args.sigma,
            task=args.task,
            device=device,
            out_dir=args.out_dir,
        )
    else:
        train(
            task=args.task,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            sigma=args.sigma,
            save_dir=args.save_dir,
            device=device,
        )


if __name__ == "__main__":
    main()