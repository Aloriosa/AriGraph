#!/usr/bin/env python
"""
Training script for the DPMs‑ANT method (simplified implementation).
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import numpy as np

# ------------------------------------------------------------------
# 0. Settings & random seed
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# 1. Dataset helpers
# ------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a folder."""
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png','.jpg','.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = transforms.functional.to_tensor(transforms.functional.resize(
            torch.load(self.files[idx]) if self.files[idx].endswith('.pt') else self._load_image(self.files[idx]),
            (32, 32)
        ))
        if self.transform:
            img = self.transform(img)
        return img

    @staticmethod
    def _load_image(path):
        from PIL import Image
        return Image.open(path).convert('RGB')

# ------------------------------------------------------------------
# 2. Simple classifier (binary: source vs target)
# ------------------------------------------------------------------
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------
# 3. Prepare data
# ------------------------------------------------------------------
# Source domain: all CIFAR-10 images
source_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
source_dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=False, transform=source_transform)

# Target domain: 10-shot images
target_dataset = ImageFolderDataset('dataset/targets', transform=source_transform)

# Build combined dataset for classifier training
source_subset = torch.utils.data.Subset(source_dataset, list(range(10, 20)))  # 10 source samples
train_classifier_ds = ConcatDataset([source_subset, target_dataset])  # 20 samples

train_loader_cls = DataLoader(train_classifier_ds, batch_size=4, shuffle=True)

# ------------------------------------------------------------------
# 4. Train classifier
# ------------------------------------------------------------------
print("Training simple classifier...")
classifier = SimpleClassifier().to(device)
optimizer_cls = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    classifier.train()
    epoch_loss = 0.0
    for batch in train_loader_cls:
        batch = batch.to(device)
        # Labels: 0 for source (first 10), 1 for target (last 10)
        labels = torch.arange(10, 20, device=device)  # placeholder to match batch size
        # In practice we create labels per sample
        labels = torch.tensor([0 if i < 10 else 1 for i in range(batch.size(0))], device=device)
        logits = classifier(batch)
        loss = criterion(logits, labels)
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={epoch_loss/len(train_loader_cls):.4f}")

# Save classifier
os.makedirs('output', exist_ok=True)
torch.save(classifier.state_dict(), 'output/classifier.pth')
print("Classifier saved to output/classifier.pth")

# ------------------------------------------------------------------
# 5. Load diffusion model
# ------------------------------------------------------------------
print("Loading pre-trained DDPM (CIFAR‑10‑32)...")
unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
unet.train()

# ------------------------------------------------------------------
# 6. Prepare target loader for fine‑tuning
# ------------------------------------------------------------------
target_loader = DataLoader(target_dataset, batch_size=4, shuffle=True)

# ------------------------------------------------------------------
# 7. Training hyper‑parameters
# ------------------------------------------------------------------
num_epochs = 5
gamma = 5.0          # similarity guidance scale
J = 10               # steps for adversarial noise
omega = 0.02         # learning rate for noise ascent
lr = 1e-4
optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

# ------------------------------------------------------------------
# 8. Fine‑tuning loop
# ------------------------------------------------------------------
print("Fine‑tuning diffusion model with DPMs‑ANT...")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(target_loader, desc=f"Epoch {epoch+1}"):
        batch = batch.to(device)

        # Sample random timesteps
        t = torch.randint(0, scheduler.num_train_timesteps, (batch.size(0),), device=device).long()

        # Sample Gaussian noise
        noise = torch.randn_like(batch)

        # Compute x_t
        sqrt_alpha_prod = scheduler.alphas_cumprod[t].sqrt().view(-1,1,1,1)
        sqrt_one_minus_alpha = (1 - scheduler.alphas_cumprod[t]).sqrt().view(-1,1,1,1)
        x_t = sqrt_alpha_prod * batch + sqrt_one_minus_alpha * noise

        # -----------------------------------------
        # 8.1 Adversarial noise selection
        # -----------------------------------------
        epsilon_adv = noise.clone().detach().requires_grad_(True)
        for _ in range(J):
            # Predict noise for current x_t
            pred_noise = unet(x_t, t).sample
            loss_adv = ((pred_noise - epsilon_adv)**2).mean()
            loss_adv.backward()
            with torch.no_grad():
                epsilon_adv += omega * epsilon_adv.grad
                # Normalize to zero mean & unit std
                epsilon_adv = (epsilon_adv - epsilon_adv.mean()) / (epsilon_adv.std() + 1e-6)
            epsilon_adv.grad.zero_()

        # -----------------------------------------
        # 8.2 Similarity‑guided term
        # -----------------------------------------
        x_t.requires_grad_(True)
        logits = classifier(x_t)
        # Target label = 1 (since x_t is from target domain)
        loss_cls = nn.CrossEntropyLoss()(logits, torch.ones(batch.size(0), dtype=torch.long, device=device))
        loss_cls.backward()
        grad_classifier = x_t.grad.clone()
        x_t.grad.zero_()

        # -----------------------------------------
        # 8.3 Final loss for UNet
        # -----------------------------------------
        # Re‑compute predicted noise with updated epsilon_adv
        # (use the same x_t, t)
        pred_noise = unet(x_t, t).sample
        loss = ((epsilon_adv - pred_noise - gamma * grad_classifier)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} finished, avg loss: {epoch_loss/len(target_loader):.4f}")

# Save fine‑tuned checkpoint
torch.save(unet.state_dict(), 'output/ckpt.pth')
print("Fine‑tuned model saved to output/ckpt.pth")