#!/usr/bin/env python3
"""
FARE (Unsupervised Adversarial Fine‑Tuning) of CLIP vision encoder
on a small dataset (CIFAR‑10 by default).

Usage:
    python src/clip_finetune.py \
        --dataset cifar10 \
        --epochs 2 \
        --batch-size 128 \
        --lr 1e-4 \
        --wd 1e-4 \
        --adv-steps 10 \
        --eps 4/255
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import clip
import numpy as np

def parse_eps(eps_str):
    """Parse string like '4/255' to float."""
    if '/' in eps_str:
        num, den = eps_str.split('/')
        return float(num) / float(den)
    return float(eps_str)

def get_dataloaders(dataset_name, batch_size, val_split=0.1):
    """Return training and test dataloaders."""
    if dataset_name.lower() == 'cifar10':
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def pgd_attack(model, images, labels, eps, alpha, iters, device):
    """Standard PGD attack on the image.  We do not use the loss
    because the goal is just to perturb the image to maximize
    the embedding distance from the clean embedding."""
    images_adv = images.clone().detach().to(device)
    images_adv.requires_grad = True

    # Pre‑compute clean embeddings
    with torch.no_grad():
        clean_emb = model.encode_image(images.to(device))

    for _ in range(iters):
        # Forward
        adv_emb = model.encode_image(images_adv)
        # Distance loss (we try to maximize this, so we use negative)
        loss = -torch.mean((adv_emb - clean_emb)**2)
        loss.backward()

        # Update adversarial image
        grad_sign = images_adv.grad.sign()
        images_adv = images_adv + alpha * grad_sign
        # Projection
        eta = torch.clamp(images_adv - images.to(device), min=-eps, max=eps)
        images_adv = torch.clamp(images.to(device) + eta, min=0, max=1).detach()
        images_adv.requires_grad = True

    return images_adv

def main():
    parser = argparse.ArgumentParser(description="FARE CLIP finetune")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--adv-steps", type=int, default=10)
    parser.add_argument("--eps", type=str, default="4/255")
    parser.add_argument("--alpha", type=float, default=1/255)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    train_loader, _ = get_dataloaders(args.dataset, args.batch_size)

    # Load CLIP vision model (freeze text encoder)
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()  # we will fine‑tune only the vision head
    for p in model.text.parameters():
        p.requires_grad = False

    # Only fine‑tune the vision layers
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )

    eps = parse_eps(args.eps)
    alpha = args.alpha

    print("Starting FARE training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)

            # Clean embeddings
            with torch.no_grad():
                clean_emb = model.encode_image(imgs)

            # Generate adversarial images
            imgs_adv = pgd_attack(
                model, imgs, None, eps, alpha, args.adv_steps, device
            )

            # Adversarial embeddings
            adv_emb = model.encode_image(imgs_adv)

            # FARE loss: keep adv_emb close to clean_emb
            loss = torch.mean((adv_emb - clean_emb) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"=== Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

    # Save the fine‑tuned vision encoder
    Path("finetuned_clip.pt").write_bytes(
        torch.save(model.state_dict(), "finetuned_clip.pt")
    )
    print("Finetuned model saved to finetuned_clip.pt")

if __name__ == "__main__":
    main()