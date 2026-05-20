#!/usr/bin/env python3
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler

from adapter import Adapter
from classifier import SourceTargetClassifier
from utils import NoisyImageDataset, compute_fid, compute_lpips

# ---------------------------------------------------------
#  Hyper‑parameters (match the paper)
# ---------------------------------------------------------
LR = 5e-5
BATCH_SIZE = 10
STEPS = 300
GAMMA = 5.0
OMEGA = 0.02
J = 10
EVAL_SAMPLES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------
#  Directories
# ---------------------------------------------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
GEN_DIR = RESULTS_DIR / "generated"
GEN_DIR.mkdir(exist_ok=True)
REAL_DIR = RESULTS_DIR / "real_svhn"   # target 10‑shot images

# ---------------------------------------------------------
#  1. Load datasets
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Source: CIFAR‑10 (used only for classifier training)
source_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
# Target: 10‑shot SVHN (we keep the first 10 training images)
full_svhn_ds = datasets.SVHN(root=DATA_DIR, split="train", download=True, transform=transform)
target_10shot_ds = torch.utils.data.Subset(full_svhn_ds, list(range(10)))

# Save target images for metric computation
REAL_DIR.mkdir(exist_ok=True)
for idx, (img, _) in enumerate(target_10shot_ds):
    img_path = REAL_DIR / f"svhn_{idx:03d}.png"
    torchvision.utils.save_image(img, img_path)

# ---------------------------------------------------------
#  2. Load pre‑trained DDPM for CIFAR‑10
# ---------------------------------------------------------
print("Loading pre‑trained DDPM (CIFAR‑10)...")
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10")
pipe = pipe.to(DEVICE)
pipe.eval()  # we will set requires_grad=False for the UNet

# Freeze UNet parameters
for p in pipe.unet.parameters():
    p.requires_grad = False

# Unet output channels (C=3 for CIFAR‑10)
unet_channels = 3
adapter = Adapter(channels=unet_channels).to(DEVICE)
adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=LR)

# Scheduler for noise schedule
scheduler = pipe.scheduler  # DDPMScheduler

# ---------------------------------------------------------
#  3. Train the binary classifier
# ---------------------------------------------------------
print("Training source/target classifier...")
classifier = SourceTargetClassifier(in_channels=3).to(DEVICE)
clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
clf_criterion = nn.CrossEntropyLoss()

# Build NoisyImageDataset for source (label 0) and target (label 1)
source_noisy_ds = NoisyImageDataset(source_ds, scheduler, label=0)
target_noisy_ds = NoisyImageDataset(target_10shot_ds, scheduler, label=1)
combined_ds = torch.utils.data.ConcatDataset([source_noisy_ds, target_noisy_ds])
clf_loader = DataLoader(combined_ds, batch_size=64, shuffle=True, num_workers=2)

# Train for a few epochs
clf_epochs = 5
classifier.train()
for epoch in range(clf_epochs):
    pbar = tqdm(clf_loader, desc=f"Classifier epoch {epoch+1}")
    for x_t, t, y in pbar:
        x_t, t, y = x_t.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
        clf_optimizer.zero_grad()
        logits = classifier(x_t, t)
        loss = clf_criterion(logits, y)
        loss.backward()
        clf_optimizer.step()
        pbar.set_postfix(loss=loss.item())
classifier.eval()

# ---------------------------------------------------------
#  4. Fine‑tune diffusion model on 10‑shot target
# ---------------------------------------------------------
print("Fine‑tuning diffusion model on 10‑shot SVHN...")

# Prepare target DataLoader
target_loader = DataLoader(target_10shot_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)

# Helper to compute sigma_hat_t^2 (approximate)
def sigma_hat_t_squared(t: torch.Tensor) -> torch.Tensor:
    # scheduler.alphas_cumprod: shape (T,)
    alpha_bar_t = scheduler.alphas_cumprod[t]               # (B,)
    alpha_bar_prev = torch.where(t > 0, scheduler.alphas_cumprod[t - 1], torch.ones_like(alpha_bar_t))
    alpha_t = scheduler.alphas_cumprod[t] / alpha_bar_prev  # α_t = ᾱ_t / ᾱ_{t-1}
    # σ̂_t^2 from paper: (1 - ᾱ_{t-1}) * sqrt(α_t / (1 - ᾱ_t))
    sigma_hat_sq = (1 - alpha_bar_prev) * torch.sqrt(alpha_t / (1 - alpha_bar_t))
    return sigma_hat_sq

# Training loop
adapter.train()
for step in tqdm(range(STEPS), desc="Fine‑tune steps"):
    # Sample a batch of target images and repeat to fill BATCH_SIZE
    imgs = next(iter(target_loader))  # (B, 3, 32, 32)
    imgs = imgs.to(DEVICE)

    # Sample timesteps
    t = torch.randint(0, scheduler.num_train_timesteps, (imgs.size(0),), device=DEVICE)
    t = t.long()

    # Sample noise and compute x_t
    eps = torch.randn_like(imgs)
    x_t = scheduler.add_noise(imgs, eps, t)

    # -------------------------------------------------------------
    # 4a. Adversarial noise selection (inner loop)
    # -------------------------------------------------------------
    eps_adv = torch.randn_like(imgs)  # start from fresh noise
    for _ in range(J):
        # Predict noise with current model (UNet + adapter)
        with torch.no_grad():
            eps_pred = pipe.unet(x_t, t).sample + adapter(x_t)
        # Loss for adversarial step
        loss_adv = ((eps_adv - eps_pred) ** 2).mean()
        # Grad w.r.t eps_adv
        grad = torch.autograd.grad(loss_adv, eps_adv, create_graph=False)[0]
        eps_adv = eps_adv + OMEGA * grad
        # Normalise to unit std
        eps_adv = (eps_adv - eps_adv.mean()) / (eps_adv.std() + 1e-6)

    # Final adversarial noise
    eps_star = eps_adv.detach()

    # Recompute noised image with adversarial noise
    x_t_star = scheduler.add_noise(imgs, eps_star, t)

    # Predict noise with model
    eps_pred_star = pipe.unet(x_t_star, t).sample + adapter(x_t_star)

    # -------------------------------------------------------------
    # 4b. Similarity‑guided loss
    # -------------------------------------------------------------
    # Classifier gradient: grad log p_T(y=1 | x_t_star)
    logits = classifier(x_t_star, t)
    # target label = 1 (target)
    tgt_label = torch.ones_like(t, dtype=torch.long, device=DEVICE)
    log_probs = F.log_softmax(logits, dim=-1)
    log_p_target = log_probs.gather(1, tgt_label.unsqueeze(1)).squeeze(1)  # (B,)
    # Gradient of log_p_target w.r.t x_t_star
    grad_logp = torch.autograd.grad(
        outputs=log_p_target.sum(),
        inputs=x_t_star,
        create_graph=True
    )[0]  # (B, C, H, W)

    # Compute sigma_hat^2
    sigma_sq = sigma_hat_t_squared(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    similarity_term = GAMMA * sigma_sq * grad_logp

    # Final loss
    loss = ((eps_star - eps_pred_star - similarity_term) ** 2).mean()

    # Backprop through adapter only
    adapter_optimizer.zero_grad()
    loss.backward()
    adapter_optimizer.step()

# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------
print("Generating samples for evaluation...")
# Generate EVAL_SAMPLES images
generator = pipe.unet
generator.eval()
generated_samples = []

with torch.no_grad():
    for i in tqdm(range(EVAL_SAMPLES), desc="Generating"):
        # Sample random noise
        noise = torch.randn(1, 3, 32, 32, device=DEVICE)
        x = noise
        # Reverse diffusion
        for t in reversed(range(scheduler.num_train_timesteps)):
            t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
            eps_pred = generator(x, t_tensor).sample + adapter(x)
            x = scheduler.step(eps_pred, t, x).prev_sample
        # x is final image
        generated_samples.append(x.cpu())

# Save generated images
for idx, img in enumerate(generated_samples):
    torchvision.utils.save_image(img.squeeze(0), GEN_DIR / f"gen_{idx:04d}.png")

# Compute metrics
print("Computing FID...")
fid_val = compute_fid(str(GEN_DIR), str(REAL_DIR), DEVICE)
print(f"FID: {fid_val:.2f}")

print("Computing LPIPS...")
lpips_val = compute_lpips(str(GEN_DIR), str(REAL_DIR), DEVICE)
print(f"LPIPS (avg min): {lpips_val:.4f}")

# Save results
with open(RESULTS_DIR / "metrics.txt", "w") as f:
    f.write(f"FID: {fid_val:.2f}\n")
    f.write(f"LPIPS: {lpips_val:.4f}\n")

print("All done. Results are in", RESULTS_DIR)