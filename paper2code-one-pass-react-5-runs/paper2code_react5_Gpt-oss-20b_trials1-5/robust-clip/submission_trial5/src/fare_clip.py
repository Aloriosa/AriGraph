#!/usr/bin/env python3
"""
FARE‑CLIP training and evaluation on CIFAR‑10.
Author: OpenAI ChatGPT (adapted for reproducibility)
"""

import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from transformers import CLIPProcessor, CLIPModel

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(tensor, device):
    return tensor.to(device, non_blocking=True)

# --------------------------------------------------------------------------- #
# PGD adversarial attack on image embeddings
# --------------------------------------------------------------------------- #
def pgd_adversarial(
    model: CLIPModel,
    x_clean: torch.Tensor,
    eps: float = 8.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    num_steps: int = 10,
):
    """
    Generate an adversarial perturbation that maximises the squared L2 distance
    between the clean and perturbed image embeddings.
    """
    x_adv = x_clean.clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        # Forward pass
        emb_clean = model.get_image_features(x_clean)
        emb_adv = model.get_image_features(x_adv)
        loss = ((emb_clean - emb_adv) ** 2).mean()
        loss.backward()
        # Gradient sign step
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv + step_size * grad_sign
        # Projection onto epsilon ball and valid pixel range
        x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)  # CLIP normalised pixel range
        x_adv = x_adv.detach().requires_grad_(True)
    return x_adv

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_fare(
    model: CLIPModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 2,
    eps: float = 8.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    num_steps: int = 10,
):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch["pixel_values"].to(device)
            # PGD attack on the batch (maximising embedding distance)
            x_adv = pgd_adversarial(
                model, images, eps, step_size, num_steps
            )
            # Compute embeddings
            emb_clean = model.get_image_features(images)
            emb_adv = model.get_image_features(x_adv)
            # FARE loss: minimise squared L2 distance
            loss = ((emb_clean - emb_adv) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

# --------------------------------------------------------------------------- #
# Zero‑shot classification on CIFAR‑10
# --------------------------------------------------------------------------- #
def zero_shot_accuracy(
    model: CLIPModel,
    dataloader: torch.utils.data.DataLoader,
    text_tokens: torch.Tensor,
    device: torch.device,
):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            images = batch["pixel_values"].to(device)
            outputs = model.get_image_features(images)  # (B, D)
            # Cosine similarity with text tokens
            logits = (outputs @ text_tokens.t()).softmax(-1)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += images.size(0)
    return correct / total

# --------------------------------------------------------------------------- #
# Main script
# --------------------------------------------------------------------------- #
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP‑ViT‑B/32
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()  # for evaluation; will be set to train() during training

    # Prepare CIFAR‑10 dataset (torchvision does not provide pixel_values, so we use processor)
    transform = processor.feature_extractor
    train_set = torchvision.datasets.CIFAR10(
        root=".", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=".", train=False, download=True, transform=transform
    )
    # Wrap into dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )

    # Prepare text embeddings for class names (zero‑shot)
    class_names = [
        "a photo of a airplane",
        "a photo of a automobile",
        "a photo of a bird",
        "a photo of a cat",
        "a photo of a deer",
        "a photo of a dog",
        "a photo of a frog",
        "a photo of a horse",
        "a photo of a ship",
        "a photo of a truck",
    ]
    text_inputs = processor(
        text=class_names, padding=True, return_tensors="pt"
    )
    text_tokens = model.get_text_features(**text_inputs).to(device)

    # Optimiser
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-5, weight_decay=1e-4
    )

    # -------------------------------------------------------------------- #
    # 1. Unsupervised adversarial fine‑tuning (FARE)
    # -------------------------------------------------------------------- #
    print("Starting FARE training...")
    train_fare(
        model,
        train_loader,
        optimizer,
        device,
        epochs=2,
        eps=8.0 / 255.0,
        step_size=1.0 / 255.0,
        num_steps=10,
    )

    # Save the fine‑tuned model
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")

    # -------------------------------------------------------------------- #
    # 2. Clean accuracy
    # -------------------------------------------------------------------- #
    clean_acc = zero_shot_accuracy(
        model, test_loader, text_tokens, device
    )
    print(f"Clean accuracy: {clean_acc*100:.2f}%")

    # -------------------------------------------------------------------- #
    # 3. Robust accuracy (PGD ε=8/255)
    # -------------------------------------------------------------------- #
    print("Evaluating robust accuracy (PGD ε=8/255)...")
    robust_correct = 0
    robust_total = 0
    model.eval()
    for batch in tqdm(test_loader, desc="Robust Eval"):
        images = batch["pixel_values"].to(device)
        # Generate adversarial examples
        x_adv = pgd_adversarial(
            model, images, eps=8.0 / 255.0, step_size=1.0 / 255.0, num_steps=10
        )
        # Forward
        outputs = model.get_image_features(x_adv)
        logits = (outputs @ text_tokens.t()).softmax(-1)
        preds = logits.argmax(-1)
        robust_correct += (preds == batch["labels"]).sum().item()
        robust_total += images.size(0)
    robust_acc = robust_correct / robust_total
    print(f"Robust accuracy (PGD ε=8/255): {robust_acc*100:.2f}%")

    # -------------------------------------------------------------------- #
    # 4. Save results
    # -------------------------------------------------------------------- #
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"Clean accuracy: {clean_acc*100:.2f}%\n")
        f.write(f"Robust accuracy (PGD ε=8/255): {robust_acc*100:.2f}%\n")

    print(f"Results written to {output_dir}")

if __name__ == "__main__":
    main()