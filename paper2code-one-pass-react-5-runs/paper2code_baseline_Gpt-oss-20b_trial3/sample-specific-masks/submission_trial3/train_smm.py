#!/usr/bin/env python3
"""
Training script for Sample‑Specific Masks for Visual Reprogramming (SMM).

Usage:
    python train_smm.py [--dataset {cifar10,cifar100,svhn}]
                       [--backbone {resnet18,resnet50,vit}]
                       [--epochs N] [--batch_size B]
                       [--lr LR] [--patch_size P]
"""
import argparse
import random
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from torch.utils.data import DataLoader
from torch.optim import Adam

# Local imports
from utils.data_loader import get_dataloaders
from models.mask_generator import ResNetMaskGenerator, ViTMaskGenerator
from utils.patch_interpolation import patch_interpolate
from utils.mapping import ILMMapper

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_img(img, size):
    """Resize PIL image or tensor to (size, size)."""
    if isinstance(img, torch.Tensor):
        return F.interpolate(img.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    else:
        return torchvision.transforms.functional.resize(img, (size, size))


# ----------------------------------------------------------------------
# Main training routine
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SMM training script")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "svhn"],
                        help="Dataset to train on.")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit"],
                        help="Pretrained backbone.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Patch size (power of two).")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay.")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size, augment=True
    )
    print(f"Dataset {args.dataset} loaded, num_classes={num_classes}")

    # ------------------------------------------------------------------
    # 2. Backbone
    # ------------------------------------------------------------------
    if args.backbone in ["resnet18", "resnet50"]:
        model = getattr(torchvision.models, args.backbone)(pretrained=True)
        input_size = 224
    else:  # vit
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        input_size = 384  # ViT‑B32 expects 384×384

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    # ------------------------------------------------------------------
    # 3. SMM components
    # ------------------------------------------------------------------
    if args.backbone in ["resnet18", "resnet50"]:
        mask_gen = ResNetMaskGenerator(l=args.patch_size).to(device)
    else:
        mask_gen = ViTMaskGenerator(l=args.patch_size).to(device)

    # Learnable shared pattern δ
    delta = nn.Parameter(torch.zeros(3, input_size, input_size, device=device))

    # Optimizers
    optimizer = Adam(
        list(mask_gen.parameters()) + [delta],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Label mapper (ILM)
    mapper = ILMMapper(num_pretrained=model.fc.out_features,
                      num_target=num_classes,
                      device=device)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        # Update label mapping at the start of the epoch
        mapper.update(train_loader, model, mask_gen, delta, device)

        model.train()
        mask_gen.train()
        delta.requires_grad_(True)

        epoch_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)  # (B,3,H,W)
            labels = labels.to(device)

            # Resize to backbone input size
            imgs_resized = F.interpolate(imgs, size=input_size, mode="bilinear", align_corners=False)

            # Generate mask
            mask = mask_gen(imgs_resized)  # (B,3,h',w')
            mask = patch_interpolate(mask, target_size=input_size, patch_size=args.patch_size)

            # Visual reprogramming
            prog_imgs = imgs_resized + delta * mask
            prog_imgs = torch.clamp(prog_imgs, 0.0, 1.0)

            # Forward through frozen backbone
            with torch.no_grad():
                logits_pretrained = model(prog_imgs)  # (B,1000)

            # Map logits to target space using ILM
            logits_target = mapper.apply(logits_pretrained, labels)

            loss = criterion(logits_target, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            _, preds = logits_target.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)

        train_acc = 100.0 * correct / total
        train_loss = epoch_loss / total

        # ------------------------------------------------------------------
        # 5. Evaluation
        # ------------------------------------------------------------------
        model.eval()
        mask_gen.eval()
        delta.requires_grad_(False)

        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                imgs_resized = F.interpolate(imgs, size=input_size, mode="bilinear", align_corners=False)
                mask = mask_gen(imgs_resized)
                mask = patch_interpolate(mask, target_size=input_size, patch_size=args.patch_size)
                prog_imgs = imgs_resized + delta * mask
                prog_imgs = torch.clamp(prog_imgs, 0.0, 1.0)

                logits_pretrained = model(prog_imgs)
                logits_target = mapper.apply(logits_pretrained, labels)

                _, preds = logits_target.max(1)
                test_correct += preds.eq(labels).sum().item()
                test_total += imgs.size(0)

        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    print("Training finished.")


if __name__ == "__main__":
    main()