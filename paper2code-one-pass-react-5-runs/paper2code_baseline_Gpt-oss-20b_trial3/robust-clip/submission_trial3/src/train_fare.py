#!/usr/bin/env python3
# ------------------------------------------------------------------
# Fine‑tune CLIP image encoder on CIFAR‑10 using the FARE loss
# ------------------------------------------------------------------
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration
BATCH_SIZE = 128
NUM_EPOCHS = 2
LR = 1e-5
WEIGHT_DECAY = 0.0
PGD_STEPS = 10
EPSILON = 8 / 255.0          # adversarial perturbation bound
PGD_STEP_SIZE = 2 / 255.0    # step size for PGD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------------------------------------------
# 1. Load pre‑trained CLIP ViT‑B/32
# ------------------------------------------------------------------
print("Loading pre‑trained CLIP ViT‑B/32...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(DEVICE)
clip_model.eval()  # keep text model frozen

# Freeze the text encoder
for p in clip_model.text_model.parameters():
    p.requires_grad = False

# Only fine‑tune the image encoder
for p in clip_model.vision_model.parameters():
    p.requires_grad = True

# Create an optimizer for the vision encoder only
optimizer = optim.AdamW(
    clip_model.vision_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# ------------------------------------------------------------------
# 2. Prepare CIFAR‑10 dataset
# ------------------------------------------------------------------
print("Preparing CIFAR‑10 dataset...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
@torch.no_grad()
def get_clean_embedding(batch_images):
    """Return clean image embeddings from the vision encoder."""
    return clip_model.vision_model(batch_images).last_hidden_state[:, 0]  # class token

def pgd_attack(clean_images, clean_embeds):
    """
    PGD attack that maximises the L2 distance between clean and perturbed embeddings.
    Returns perturbed images and their embeddings.
    """
    # initialise perturbation uniformly in [-eps, eps]
    delta = torch.empty_like(clean_images).uniform_(-EPSILON, EPSILON).to(DEVICE)
    delta.requires_grad = True

    for _ in range(PGD_STEPS):
        perturbed = torch.clamp(clean_images + delta, -1.0, 1.0)
        pert_embeds = clip_model.vision_model(perturbed).last_hidden_state[:, 0]
        loss = -torch.norm(pert_embeds - clean_embeds, p=2, dim=1).mean()
        loss.backward()
        # update perturbation
        grad_sign = delta.grad.sign()
        delta.data = (delta + PGD_STEP_SIZE * grad_sign).clamp_(-EPSILON, EPSILON)
        delta.grad.zero_()

    perturbed = torch.clamp(clean_images + delta, -1.0, 1.0).detach()
    pert_embeds = clip_model.vision_model(perturbed).last_hidden_state[:, 0]
    return perturbed, pert_embeds

# ------------------------------------------------------------------
# 3. Training loop
# ------------------------------------------------------------------
print("Starting FARE training...")
clip_model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for imgs, _ in pbar:
        imgs = imgs.to(DEVICE)

        # Compute clean embeddings
        clean_embeds = get_clean_embedding(imgs)

        # PGD attack to generate adversarial embeddings
        adv_imgs, adv_embeds = pgd_attack(imgs, clean_embeds)

        # FARE loss: L2 distance between clean and adv embeddings
        loss = torch.norm(adv_embeds - clean_embeds, p=2, dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch} finished. Avg loss: {avg_loss:.4f}")

# ------------------------------------------------------------------
# 4. Save the fine‑tuned model
# ------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
model_path = "outputs/fare_clip.pth"
torch.save(clip_model.state_dict(), model_path)
print(f"Fine‑tuned model saved to {model_path}")

# ------------------------------------------------------------------
# 5. Save a small checkpoint for evaluation
# ------------------------------------------------------------------
torch.save(
    {
        "model_state_dict": clip_model.state_dict(),
        "config": {
            "batch_size": BATCH_SIZE,
            "eps": EPSILON,
            "pgd_steps": PGD_STEPS,
            "step_size": PGD_STEP_SIZE,
        },
    },
    "outputs/fare_clip_checkpoint.pt",
)
print("Checkpoint created for evaluation.")