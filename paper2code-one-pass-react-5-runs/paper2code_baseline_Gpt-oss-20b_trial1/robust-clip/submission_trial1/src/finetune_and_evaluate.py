#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FARE – Unsupervised Adversarial Fine‑Tuning of CLIP (Demo)

This script performs a lightweight adversarial fine‑tuning of the visual encoder of
CLIP on a small subset of CIFAR‑10 and evaluates zero‑shot classification
accuracy on clean and adversarial test images.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import os

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TRAIN_STEPS = 1  # number of epochs on the small subset
ATTACK_EPS = 4.0 / 255.0  # epsilon in pixel range [-1, 1] (CLIP normalizes to [-1,1])
ATTACK_STEPS = 5
STEP_SIZE = ATTACK_EPS / 10.0
LR = 1e-5
NUM_CLASSES = 10
TRAIN_SUBSET = 10000  # we use a small subset for speed

# set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def compute_logits(image_emb, text_emb):
    """
    Compute similarity logits between image embeddings and text embeddings.
    Both inputs are assumed to be L2‑normalized.
    """
    return image_emb @ text_emb.t()  # (B, C)

def adversarial_batch(model, processor, images, labels, text_emb, device):
    """
    For a batch of clean images, generate adversarial examples that
    maximize the cross‑entropy loss, then return the adversarial images.
    """
    adv = images.clone().detach().requires_grad_(True)
    for _ in range(ATTACK_STEPS):
        # Forward
        img_emb = model.get_image_features(pixel_values=adv)
        img_emb = F.normalize(img_emb, dim=1)
        logits = compute_logits(img_emb, text_emb)
        loss = F.cross_entropy(logits, labels)
        # Backward
        loss.backward()
        # PGD update
        adv = adv + STEP_SIZE * adv.grad.sign()
        adv = torch.clamp(adv, -1.0, 1.0).detach().requires_grad_(True)
    return adv

def evaluate(model, processor, dataset, text_emb, device, adversarial=False):
    """
    Evaluate zero‑shot classification accuracy on a dataset.
    If adversarial=True, adversarial examples are generated on the fly.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = processor(
            images=batch["image"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).pixel_values.to(device)
        labels = torch.tensor(batch["label"], device=device)

        if adversarial:
            images = adversarial_batch(
                model, processor, images, labels, text_emb, device
            )

        img_emb = model.get_image_features(pixel_values=images)
        img_emb = F.normalize(img_emb, dim=1)
        logits = compute_logits(img_emb, text_emb)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    model.train()

    # Freeze text encoder – only fine‑tune the visual encoder
    for p in model.text_model.parameters():
        p.requires_grad = False
    for p in model.text_projection.parameters():
        p.requires_grad = False

    # Prepare class‑text embeddings (CIFAR‑10 class names)
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    text_inputs = processor(
        text=class_names,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = F.normalize(text_emb, dim=1)

    # Load CIFAR‑10 dataset
    full_train = load_dataset("cifar10", split="train", cache_dir="cache")
    full_test = load_dataset("cifar10", split="test", cache_dir="cache")

    # Subsample training set for speed
    train_ds = full_train.shuffle(seed=SEED).select(range(TRAIN_SUBSET))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer for visual encoder
    optimizer = torch.optim.AdamW(model.visual.parameters(), lr=LR)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(TRAIN_STEPS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images = processor(
                images=batch["image"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).pixel_values.to(DEVICE)
            labels = torch.tensor(batch["label"], device=DEVICE)

            # Generate adversarial examples
            adv_images = adversarial_batch(
                model, processor, images, labels, text_emb, DEVICE
            )

            # Forward on clean and adv images
            img_emb_clean = model.get_image_features(pixel_values=images)
            img_emb_clean = F.normalize(img_emb_clean, dim=1)
            logits_clean = compute_logits(img_emb_clean, text_emb)

            img_emb_adv = model.get_image_features(pixel_values=adv_images)
            img_emb_adv = F.normalize(img_emb_adv, dim=1)
            logits_adv = compute_logits(img_emb_adv, text_emb)

            # Loss: cross‑entropy on clean + adv (adversarial training)
            loss_clean = F.cross_entropy(logits_clean, labels)
            loss_adv = F.cross_entropy(logits_adv, labels)
            loss = loss_clean + loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    model.eval()
    clean_acc = evaluate(model, processor, full_test, text_emb, DEVICE, adversarial=False)
    adv_acc = evaluate(model, processor, full_test, text_emb, DEVICE, adversarial=True)

    # ------------------------------------------------------------------
    # Baseline with original CLIP (no fine‑tuning)
    # ------------------------------------------------------------------
    baseline_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    baseline_model.eval()
    baseline_clean_acc = evaluate(baseline_model, processor, full_test, text_emb, DEVICE, adversarial=False)
    baseline_adv_acc = evaluate(baseline_model, processor, full_test, text_emb, DEVICE, adversarial=True)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs("output", exist_ok=True)
    with open("output/results.txt", "w") as f:
        f.write("Method,Clean Accuracy,Adversarial Accuracy\n")
        f.write(f"Original CLIP,{baseline_clean_acc:.4f},{baseline_adv_acc:.4f}\n")
        f.write(f"FARE,{clean_acc:.4f},{adv_acc:.4f}\n")

    # Print to console
    print("\n=== Final Results ===")
    print(f"Original CLIP - Clean: {baseline_clean_acc:.2%}, Adv: {baseline_adv_acc:.2%}")
    print(f"FARE         - Clean: {clean_acc:.2%}, Adv: {adv_acc:.2%}")

if __name__ == "__main__":
    main()