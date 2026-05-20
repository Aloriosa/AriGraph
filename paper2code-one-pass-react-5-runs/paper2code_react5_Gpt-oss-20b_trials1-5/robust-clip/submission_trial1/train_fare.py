#!/usr/bin/env python
"""
Unsupervised adversarial fine‑tuning of CLIP (FARE).

The script follows the training protocol described in the paper:
  • 2 epochs on the full ImageNet training set
  • 10 PGD steps per batch
  • ε = 2/255
  • AdamW optimizer, lr = 1e‑5, weight decay = 1e‑4
  • MSE loss between clean and perturbed image embeddings
  • ViT‑B/32 CLIP model from OpenAI

The trained checkpoint is saved to `checkpoints/clip_fare.pt`.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPModel

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pgd_adversarial(
    model: CLIPModel,
    images: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    device: torch.device,
):
    """
    Generate adversarial perturbations using PGD with respect to the
    squared L2 loss between clean and perturbed embeddings.
    """
    images_adv = images.clone().detach().requires_grad_(True)
    for _ in range(steps):
        # forward clean
        with torch.no_grad():
            feat_clean = model.get_image_features(images).detach()
        # forward perturbed
        feat_adv = model.get_image_features(images_adv)
        loss = nn.functional.mse_loss(feat_adv, feat_clean)
        loss.backward()

        # gradient sign
        grad_sign = images_adv.grad.sign()
        images_adv = images_adv.detach() + step_size * grad_sign
        # project onto ε‑ball
        perturbation = torch.clamp(images_adv - images, min=-eps, max=eps)
        images_adv = torch.clamp(images + perturbation, min=0.0, max=1.0).detach()
        images_adv.requires_grad_(True)

    return images_adv.detach()

# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained CLIP (ViT‑B/32)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip.to(device)

    # Freeze all parameters except the vision encoder
    for name, param in clip.named_parameters():
        if "visual" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Data transforms
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load the full ImageNet training set
    # (This will download the dataset if not present)
    train_ds = load_dataset("imagenet", split="train")
    val_ds   = load_dataset("imagenet", split="validation")

    def collate_fn(batch):
        images = [preprocess(item["image"]) for item in batch]
        images = torch.stack(images)
        return images

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        clip.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    clip.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images in pbar:
            images = images.to(device)

            # Generate adversarial images
            images_adv = pgd_adversarial(
                model=clip,
                images=images,
                eps=args.eps,
                steps=args.pgd_steps,
                step_size=args.step_size,
                device=device,
            ).to(device)

            # Forward passes
            feat_clean = clip.get_image_features(images)
            feat_adv   = clip.get_image_features(images_adv)

            # Loss: mean squared difference (unsupervised)
            loss = nn.functional.mse_loss(feat_adv, feat_clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    # Save checkpoint
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "clip_fare.pt")
    torch.save(clip.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FARE unsupervised adversarial fine‑tuning of CLIP."
    )
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="Weight decay.")
    parser.add_argument("--eps", type=float, default=2/255.0,
                        help="PGD ε.")
    parser.add_argument("--pgd-steps", type=int, default=10,
                        help="Number of PGD steps.")
    parser.add_argument("--step-size", type=float, default=1/255.0,
                        help="PGD step size.")
    parser.add_argument("--out-dir", type=str, default="checkpoints",
                        help="Directory to store the checkpoint.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()
    main(args)