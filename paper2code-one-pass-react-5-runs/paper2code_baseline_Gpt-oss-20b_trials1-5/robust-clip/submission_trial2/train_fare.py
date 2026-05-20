#!/usr/bin/env python3
"""
train_fare.py – Train an unsupervised adversarially fine‑tuned CLIP encoder (FARE)
on a small subset of ImageNet (CIFAR‑10 as a stand‑in) for demonstration purposes.
"""

import os
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPImageProcessor

# ---------- Configuration ----------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 2
NUM_STEPS = 10          # PGD steps
EPS = 4 / 255.0
STEP_SIZE = 1 / 255.0
LR = 1e-5
WD = 1e-4
MODEL_OUT = Path("trained_fare_clip.pt")
# ------------------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(SEED)

# ---------- Data ----------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

train_dataset = torchvision.datasets.CIFAR10(root="data", train=True,
                                            download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)

# ---------- Model ----------
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip.requires_grad_(True)          # fine‑tune the visual encoder
optimizer = torch.optim.AdamW(clip.parameters(), lr=LR, weight_decay=WD)

# ---------- PGD utility ----------
@torch.no_grad()
def pgd_attack(img, clean_emb, target_emb, eps, alpha, steps):
    """Generate adversarial image that maximises L2 distance to clean embedding."""
    img_adv = img.clone().detach()
    img_adv.requires_grad = True

    for _ in range(steps):
        # forward
        emb = clip.get_image_features(img_adv)
        loss = F.mse_loss(emb, target_emb)
        loss.backward()

        # gradient step
        grad = img_adv.grad.data
        img_adv = img_adv + alpha * grad.sign()
        # clip to epsilon ball around original image
        img_adv = torch.max(torch.min(img_adv, img + eps), img - eps)
        # clip to valid image range
        img_adv = torch.clamp(img_adv, 0.0, 1.0)
        img_adv = img_adv.detach()
        img_adv.requires_grad = True

    return img_adv

# ---------- Training ----------
print("Training FARE‑CLIP on CIFAR‑10 (≈2 epochs)…")
clip.train()
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for imgs, _ in pbar:
        imgs = imgs.to(DEVICE)

        # Clean image embeddings
        clean_emb = clip.get_image_features(imgs)

        # Generate adversarial images
        adv_imgs = pgd_attack(imgs, clean_emb, clean_emb, EPS, STEP_SIZE, NUM_STEPS)

        # Adversarial embeddings
        adv_emb = clip.get_image_features(adv_imgs)

        # Loss encourages embeddings of adversarial to stay close to clean
        loss = F.mse_loss(adv_emb, clean_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

torch.save(clip.state_dict(), MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")