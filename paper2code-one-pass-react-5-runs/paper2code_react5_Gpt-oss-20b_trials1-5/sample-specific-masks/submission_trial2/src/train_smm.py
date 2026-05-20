#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SMM (Sample‑specific Multi‑channel Masks) implementation for multiple target datasets.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src import utils

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Mask generator – lightweight 5‑layer CNN producing a 3‑channel mask
# ----------------------------------------------------------------------
class MaskGenerator(nn.Module):
    """
    5‑layer CNN:
        Conv3x3 -> ReLU -> MaxPool2d(2)
        Conv3x3 -> ReLU -> MaxPool2d(2)
        Conv3x3 -> ReLU
        Conv3x3 -> ReLU
        Conv3x3 (output 3 channels)
    Input: (3, H, W) where H=W=224 (after resizing)
    Output: (3, 224, 224) – after nearest‑neighbour up‑sampling.
    """

    def __init__(self, in_channels=3, out_channels=3, num_pool=2):
        super().__init__()
        self.num_pool = num_pool
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.features(x)  # shape (N, 3, 56, 56)
        # Patch‑wise interpolation: repeat each pixel to 2^num_pool × 2^num_pool block
        scale = 2 ** self.num_pool
        out = out.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
        # After repeat, shape is (N, 3, 224, 224)
        # Clamp to [0,1] as in the paper
        return torch.sigmoid(out)


# ----------------------------------------------------------------------
# Iterative Label Mapping (Ilm) – compute mapping from ImageNet to target
# ----------------------------------------------------------------------
def compute_ilm_mapping(
    model: nn.Module,
    mask_gen: nn.Module,
    delta: torch.Tensor,
    train_loader: DataLoader,
    device: torch.device,
    num_target_classes: int,
) -> torch.Tensor:
    """
    Returns a tensor of shape (num_target_classes,) where each entry
    is the ImageNet class index mapped to that target class.
    The mapping is injective (one‑to‑one).
    """
    model.eval()
    mask_gen.eval()
    mapping_counts = np.zeros((1000, num_target_classes), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)          # (B,3,224,224)
            labels = labels.to(device)      # (B,)
            # generate mask
            mask = mask_gen(imgs)           # (B,3,224,224)
            # reprogrammed image
            inp = imgs + delta * mask
            logits = model(inp)             # (B,1000)
            preds = logits.argmax(dim=1)    # (B,)
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            for p, t in zip(preds_cpu, labels_cpu):
                mapping_counts[p, t] += 1

    # Greedy injective mapping
    mapping = np.full(num_target_classes, -1, dtype=np.int64)
    used_imagenet = set()
    for t in range(num_target_classes):
        best_p, best_c = -1, -1
        for p in range(1000):
            if p in used_imagenet:
                continue
            c = mapping_counts[p, t]
            if c > best_c:
                best_c = c
                best_p = p
        if best_p == -1:
            break
        mapping[t] = best_p
        used_imagenet.add(best_p)

    return torch.from_numpy(mapping).to(device)


# ----------------------------------------------------------------------
# Baseline mask functions
# ----------------------------------------------------------------------
def get_baseline_mask(mask_name: str, img_size: int) -> torch.Tensor:
    if mask_name == "pad":
        return utils.create_pad_mask(img_size, pad=28)
    if mask_name == "narrow":
        return utils.create_narrow_mask(img_size, width=28)
    if mask_name == "medium":
        return utils.create_medium_mask(img_size, width=56)
    if mask_name == "full":
        return utils.create_full_mask(img_size)
    raise ValueError(f"Unknown baseline mask: {mask_name}")


# ----------------------------------------------------------------------
# Training / Evaluation
# ----------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    mask_gen: nn.Module,
    delta: torch.Tensor,
    train_loader: DataLoader,
    optimizer_mask: torch.optim.Optimizer,
    optimizer_delta: torch.optim.Optimizer,
    mapping: torch.Tensor,
    device: torch.device,
    baseline_mask: torch.Tensor | None,
):
    model.train()
    mask_gen.train()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # generate mask
        if baseline_mask is None:
            mask = mask_gen(imgs)  # (B,3,224,224)
        else:
            mask = baseline_mask.unsqueeze(0).repeat(imgs.size(0), 1, 1, 1).to(device)

        inp = imgs + delta * mask
        logits = model(inp)  # (B,1000)

        # aggregate logits according to mapping
        target_logits = torch.zeros(imgs.size(0), len(mapping), device=device)
        for t, im_idx in enumerate(mapping):
            if im_idx < 0:
                continue
            target_logits[:, t] = logits[:, im_idx]

        loss = criterion(target_logits, labels)
        optimizer_mask.zero_grad()
        optimizer_delta.zero_grad()
        loss.backward()
        optimizer_mask.step()
        optimizer_delta.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = target_logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    mask_gen: nn.Module,
    delta: torch.Tensor,
    test_loader: DataLoader,
    mapping: torch.Tensor,
    device: torch.device,
    baseline_mask: torch.Tensor | None,
):
    model.eval()
    mask_gen.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            if baseline_mask is None:
                mask = mask_gen(imgs)
            else:
                mask = baseline_mask.unsqueeze(0).repeat(imgs.size(0), 1, 1, 1).to(device)

            inp = imgs + delta * mask
            logits = model(inp)
            target_logits = torch.zeros(imgs.size(0), len(mapping), device=device)
            for t, im_idx in enumerate(mapping):
                if im_idx < 0:
                    continue
                target_logits[:, t] = logits[:, im_idx]

            _, preds = target_logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
    acc = correct / total
    return acc


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SMM training script")
    parser.add_argument("--dataset", default="cifar10", choices=[
        "cifar10", "cifar100", "svhn", "gtsrb",
        "flowers102", "dtd", "ucf101", "food101",
        "eurosat", "oxfordpets", "sun397"
    ], help="Target dataset")
    parser.add_argument("--backbone", default="resnet18", choices=[
        "resnet18", "vit_b32"
    ], help="Pre‑trained backbone")
    parser.add_argument("--baseline", default="smm",
                        choices=["smm", "pad", "narrow", "medium", "full"],
                        help="Baseline mask or 'smm' for sample‑specific mask")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ---------- Data ----------
    if args.dataset == "cifar10":
        num_classes = 10
        root = "./data"
        img_size = 224
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train_set = dset.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_set = dset.CIFAR10(root=root, train=False, download=True, transform=transform)
    elif args.dataset == "cifar100":
        num_classes = 100
        root = "./data"
        img_size = 224
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train_set = dset.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_set = dset.CIFAR100(root=root, train=False, download=True, transform=transform)
    elif args.dataset == "svhn":
        num_classes = 10
        root = "./data"
        img_size = 224
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train_set = dset.SVHN(root=root, download=True, split="train", transform=transform)
        test_set = dset.SVHN(root=root, download=True, split="test", transform=transform)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented in this minimal repo")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ---------- Model ----------
    if args.backbone == "resnet18":
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:  # vit_b32
        backbone = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.to(device)

    # ---------- SMM components ----------
    mask_gen = MaskGenerator().to(device)
    delta = nn.Parameter(torch.zeros(3, img_size, img_size, device=device))

    # ---------- Optimizers ----------
    optimizer_mask = torch.optim.Adam(mask_gen.parameters(), lr=args.lr)
    optimizer_delta = torch.optim.Adam([delta], lr=args.lr)

    # ---------- Scheduler ----------
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer_mask, milestones=[100, 145], gamma=0.1
    )

    # ---------- Baseline mask ----------
    baseline_mask = None
    if args.baseline != "smm":
        baseline_mask = get_baseline_mask(args.baseline, img_size)

    # ---------- Checkpointing ----------
    best_acc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # ---------- Training ----------
    for epoch in range(1, args.epochs + 1):
        # Update mapping each epoch
        mapping = compute_ilm_mapping(backbone, mask_gen, delta, train_loader,
                                      device, num_classes)

        train_loss, train_acc = train_epoch(
            backbone, mask_gen, delta, train_loader,
            optimizer_mask, optimizer_delta, mapping, device, baseline_mask
        )
        test_acc = evaluate(
            backbone, mask_gen, delta, test_loader,
            mapping, device, baseline_mask
        )
        scheduler.step()

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "backbone_state": backbone.state_dict(),
                "mask_gen_state": mask_gen.state_dict(),
                "delta_state": delta.detach().cpu(),
                "mapping": mapping.cpu(),
            }, checkpoint_dir / f"best_{args.dataset}_{args.baseline}.pth")

    print(f"\nFinal Test Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()