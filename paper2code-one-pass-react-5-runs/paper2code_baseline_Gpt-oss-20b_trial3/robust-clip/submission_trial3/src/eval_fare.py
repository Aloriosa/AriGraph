#!/usr/bin/env python3
# ------------------------------------------------------------------
# Evaluate the fine‑tuned CLIP (FARE) on CIFAR‑10 test set
# ------------------------------------------------------------------
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Configuration (must match training)
BATCH_SIZE = 128
EPSILON = 8 / 255.0
PGD_STEPS = 10
PGD_STEP_SIZE = 2 / 255.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Load the fine‑tuned model
# ------------------------------------------------------------------
model_path = "outputs/fare_clip.pth"
print(f"Loading fine‑tuned model from {model_path}...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.vision_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
clip_model.to(DEVICE)
clip_model.eval()

# ------------------------------------------------------------------
# Prepare CIFAR‑10 test set
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ------------------------------------------------------------------
# Text prompts for zero‑shot classification
# ------------------------------------------------------------------
labels = test_dataset.classes  # ['airplane', 'automobile', 'bird', ...]
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_inputs = processor(text=labels, return_tensors="pt", padding=True).to(DEVICE)

with torch.no_grad():
    text_embeds = clip_model.text_model(**text_inputs).last_hidden_state[:, 0]
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

# ------------------------------------------------------------------
# Helper: PGD adversarial attack on images
# ------------------------------------------------------------------
def pgd_attack(imgs):
    delta = torch.empty_like(imgs).uniform_(-EPSILON, EPSILON).to(DEVICE)
    delta.requires_grad = True
    for _ in range(PGD_STEPS):
        pert = torch.clamp(imgs + delta, -1.0, 1.0)
        emb = clip_model.vision_model(pert).last_hidden_state[:, 0]
        emb = emb / emb.norm(dim=1, keepdim=True)
        logits = (emb @ text_embeds.T).softmax(dim=-1)
        loss = -logits.max(dim=1).values.mean()  # maximize the highest class prob
        loss.backward()
        grad_sign = delta.grad.sign()
        delta.data = (delta + PGD_STEP_SIZE * grad_sign).clamp_(-EPSILON, EPSILON)
        delta.grad.zero_()
    return torch.clamp(imgs + delta, -1.0, 1.0).detach()

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
clean_correct = 0
robust_correct = 0
total = 0

print("Starting evaluation...")
with torch.no_grad():
    for imgs, targets in tqdm(test_loader):
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Clean predictions
        clean_emb = clip_model.vision_model(imgs).last_hidden_state[:, 0]
        clean_emb = clean_emb / clean_emb.norm(dim=1, keepdim=True)
        logits_clean = (clean_emb @ text_embeds.T)
        preds_clean = logits_clean.argmax(dim=1)
        clean_correct += (preds_clean == targets).sum().item()

        # Robust predictions
        adv_imgs = pgd_attack(imgs)
        adv_emb = clip_model.vision_model(adv_imgs).last_hidden_state[:, 0]
        adv_emb = adv_emb / adv_emb.norm(dim=1, keepdim=True)
        logits_adv = (adv_emb @ text_embeds.T)
        preds_adv = logits_adv.argmax(dim=1)
        robust_correct += (preds_adv == targets).sum().item()

        total += imgs.size(0)

clean_acc = 100.0 * clean_correct / total
robust_acc = 100.0 * robust_correct / total

print(f"Clean accuracy: {clean_acc:.2f}%")
print(f"Robust accuracy (ε={EPSILON:.3f}): {robust_acc:.2f}%")

# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
with open("outputs/results.txt", "w") as f:
    f.write(f"Clean accuracy: {clean_acc:.4f}\n")
    f.write(f"Robust accuracy (ε={EPSILON:.3f}): {robust_acc:.4f}\n")

print("Results written to outputs/results.txt")