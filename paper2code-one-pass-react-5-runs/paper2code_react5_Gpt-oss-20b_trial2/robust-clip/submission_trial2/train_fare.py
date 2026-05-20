#!/usr/bin/env python3
"""
FARE unsupervised adversarial fine‑tuning on an unlabeled dataset.
For demonstration it uses CIFAR‑10 training set (50000 images).
"""

import argparse
import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import clip
from tqdm import tqdm

from datasets import load_dataset

# ---------------------------------------------------------------------------

def parse_alpha(s):
    """Parse a fraction string like '2/255' into float."""
    if '/' in s:
        a, b = s.split('/')
        return float(a) / float(b)
    return float(s)

# ---------------------------------------------------------------------------

class FARELoss(nn.Module):
    """
    FARE loss: L2 distance between original and perturbed embeddings.
    """
    def __init__(self):
        super().__init__()

    def forward(self, orig, perturbed):
        return F.mse_loss(perturbed, orig)

# ---------------------------------------------------------------------------

def apgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
    half_precision: bool = True,
    device: torch.device = torch.device("cpu")
):
    """
    Approximate PGD (APGD) attack used in the paper.
    1. Half‑precision (float16) loop for fast iterations.
    2. Single‑precision refinement on the final perturbation.
    Returns the adversarial images.
    """
    model.eval()
    images = images.clone().detach().to(device)
    perturb = torch.zeros_like(images, device=device, dtype=torch.float32)

    # 1. Half‑precision loop
    if half_precision:
        images_half = images.half()
        perturb_half = torch.zeros_like(images_half)
        for _ in range(steps):
            perturb_half.requires_grad = True
            pert_images = torch.clamp(images_half + perturb_half, -1.0, 1.0)
            logits = model.encode_image(pert_images)
            loss = -logits.norm(dim=-1).mean()  # maximize norm (any reasonable loss)
            loss.backward()
            grad = perturb_half.grad
            perturb_half = perturb_half + step_size * grad.sign()
            perturb_half = torch.clamp(perturb_half, -epsilon, epsilon)
            perturb_half = perturb_half.detach()
        perturb = perturb_half.to(torch.float32)

    # 2. Single‑precision refinement
    for _ in range(10):
        perturb.requires_grad = True
        pert_images = torch.clamp(images + perturb, -1.0, 1.0)
        logits = model.encode_image(pert_images)
        loss = -logits.norm(dim=-1).mean()
        loss.backward()
        grad = perturb.grad
        perturb = perturb + step_size * grad.sign()
        perturb = torch.clamp(perturb, -epsilon, epsilon)
        perturb = perturb.detach()

    adv_images = torch.clamp(images + perturb, -1.0, 1.0)
    return adv_images

# ---------------------------------------------------------------------------

def get_dataloader(batch_size, split="train"):
    """
    Return a DataLoader for CIFAR‑10 (unlabeled) training set.
    """
    dataset = load_dataset("cifar10", split=split)
    # We only need images; no labels.
    def collate_fn(batch):
        imgs = [torch.tensor(item["image"]).permute(2, 0, 1).float() / 255.0
                for item in batch]
        imgs = torch.stack(imgs)
        return imgs
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=4)
    return loader

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FARE training")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=parse_alpha, default="2/255")
    parser.add_argument("--adv_steps", type=int, default=10)
    parser.add_argument("--step_size", type=parse_alpha, default="1/255")
    parser.add_argument("--output", type=str, default="checkpoints/fare_clip.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root=".")
    clip_model.eval()
    # We only fine‑tune the vision encoder
    vision_encoder = clip_model.visual
    vision_encoder.train()

    # ------------------------------------------------------------
    # DataLoader
    loader = get_dataloader(args.batch_size, split="train")

    # ------------------------------------------------------------
    # Optimizer
    optimizer = optim.AdamW(vision_encoder.parameters(),
                            lr=args.lr, weight_decay=args.wd)

    # Learning‑rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=args.epochs * len(loader))

    # Loss
    loss_fn = FARELoss()

    # ------------------------------------------------------------
    print("Starting training...")
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs in pbar:
            imgs = imgs.to(device)
            # Original embeddings (detached)
            with torch.no_grad():
                orig_emb = vision_encoder(imgs)

            # Generate adversarial examples
            adv_imgs = apgd_attack(
                vision_encoder,
                imgs,
                epsilon=args.epsilon,
                steps=args.adv_steps,
                step_size=args.step_size,
                device=device
            )
            # Perturbed embeddings
            pert_emb = vision_encoder(adv_imgs)

            # Loss: keep embeddings of perturbed close to original
            loss = loss_fn(orig_emb, pert_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=loss.item())

    # ------------------------------------------------------------
    # Save fine‑tuned vision encoder (state_dict)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(vision_encoder.state_dict(), args.output)
    print(f"Saved FARE checkpoint to {args.output}")

if __name__ == "__main__":
    main()