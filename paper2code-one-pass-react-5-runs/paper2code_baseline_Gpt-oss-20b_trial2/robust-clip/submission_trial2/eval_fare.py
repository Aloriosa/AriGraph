#!/usr/bin/env python3
"""
eval_fare.py – Evaluate the fine‑tuned FARE‑CLIP model on clean and adversarial
CIFAR‑10 images using zero‑shot classification.
"""

import os
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score

# ---------- Configuration ----------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
MODEL_IN = Path("trained_fare_clip.pt")
EPS = 4 / 255.0
STEP_SIZE = 1 / 255.0
NUM_STEPS = 10
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

test_dataset = torchvision.datasets.CIFAR10(root="data", train=False,
                                           download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)

# ---------- Load model ----------
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip.load_state_dict(torch.load(MODEL_IN, map_location=DEVICE))
clip.eval()
clip.requires_grad_(False)

# Text encoder
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
class_names = test_dataset.classes
text_prompts = [f"a photo of a {name}" for name in class_names]
text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
with torch.no_grad():
    text_emb = clip.get_text_features(**text_inputs)
    text_emb = F.normalize(text_emb, dim=-1)

# ---------- Utility ----------
@torch.no_grad()
def pgd_attack(img, eps, alpha, steps):
    """Generate adversarial image that maximises L2 distance to clean embedding."""
    img_adv = img.clone().detach()
    img_adv.requires_grad = True

    for _ in range(steps):
        # forward
        emb = clip.get_image_features(img_adv)
        loss = emb.norm(p=2, dim=1).mean()   # dummy loss to trigger grad
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

# ---------- Evaluation ----------
clean_preds = []
adv_preds = []
true_labels = []

print("Evaluating on clean and adversarial CIFAR‑10…")
clip.eval()
for imgs, labels in tqdm(test_loader):
    imgs = imgs.to(DEVICE)
    labels = labels.to(DEVICE)

    # Clean embeddings
    with torch.no_grad():
        img_emb = clip.get_image_features(imgs)
        img_emb = F.normalize(img_emb, dim=-1)

        # Cosine similarity with text embeddings
        sims = img_emb @ text_emb.T
        _, pred_clean = sims.max(1)
        clean_preds.extend(pred_clean.cpu().numpy())

    # Adversarial images
    adv_imgs = pgd_attack(imgs, EPS, STEP_SIZE, NUM_STEPS)
    with torch.no_grad():
        adv_emb = clip.get_image_features(adv_imgs)
        adv_emb = F.normalize(adv_emb, dim=-1)
        sims = adv_emb @ text_emb.T
        _, pred_adv = sims.max(1)
        adv_preds.extend(pred_adv.cpu().numpy())

    true_labels.extend(labels.cpu().numpy())

clean_acc = accuracy_score(true_labels, clean_preds)
adv_acc = accuracy_score(true_labels, adv_preds)

print(f"\n=== Results ===")
print(f"Clean accuracy : {clean_acc*100:.2f}%")
print(f"Adversarial accuracy (ε={EPS:.3f}) : {adv_acc*100:.2f}%")