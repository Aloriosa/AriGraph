#!/usr/bin/env python
"""
Main training script implementing DPMs‑ANT.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils import randn_tensor
from torchvision import transforms
import lpips

from utils import normalize_noise, get_alpha_cumprod, get_beta
from dataset import TargetDataset, collate_fn
from classifier import train_classifier

# ====================== Hyper‑parameters ======================
BATCH_SIZE = 8
NUM_STEPS = 300          # fine‑tune for 300 steps
LEARNING_RATE = 5e-5
GAMMA = 5.0              # similarity guidance weight
J_ADV = 5                # adversarial noise steps
OMEGA = 0.02             # adversarial noise learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ====================== Prepare data ======================
target_dataset = TargetDataset()
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)

# ====================== Load pre‑trained DDPM ======================
print("Loading pre‑trained DDPM (google/ddpm-cifar10-32)...")
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
pipe = pipe.to(DEVICE)
unet = pipe.unet  # UNet2DModel
scheduler = pipe.scheduler  # DDPMScheduler

# ====================== Prepare classifier for similarity guidance ======================
print("Training similarity classifier...")
classifier = train_classifier(unet, scheduler,
                              target_loader, DEVICE,
                              epochs=4, lr=1e-4, gamma=GAMMA)
classifier.eval()
classifier.to(DEVICE)

# ====================== Optimizer ======================
optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)

# ====================== Training loop ======================
alpha_cumprod = get_alpha_cumprod(scheduler)

log_path = "output/log.txt"
os.makedirs("output/generated_samples", exist_ok=True)

with open(log_path, "w") as log_file:
    for step in tqdm(range(NUM_STEPS)):
        # ---- Sample a batch of target images ----
        imgs = next(iter(target_loader)).to(DEVICE)  # (B, 3, 32, 32)

        # ---- Sample random timesteps and noise ----
        t = torch.randint(0, scheduler.num_train_timesteps, (imgs.shape[0],), device=DEVICE)
        noise = torch.randn_like(imgs, device=DEVICE)

        # ---- Compute noisy images x_t ----
        alpha_t = alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        x_t = sqrt_alpha_t * imgs + sqrt_one_minus_alpha_t * noise

        # ---- Adversarial noise selection ----
        # Start from random noise and ascend on the loss w.r.t. noise
        adv_noise = torch.randn_like(imgs, device=DEVICE)
        adv_noise.requires_grad_(True)
        for _ in range(J_ADV):
            # Compute predicted noise on current noisy image
            adv_x_t = sqrt_alpha_t * imgs + sqrt_one_minus_alpha_t * adv_noise
            eps_pred = unet(adv_x_t, t).sample
            # Loss is MSE between eps_pred and true noise (ignoring classifier for speed)
            loss_adv = F.mse_loss(eps_pred, noise)
            loss_adv.backward()
            # Gradient ascent step
            adv_noise.data = adv_noise.data + OMEGA * adv_noise.grad.data
            adv_noise.data = normalize_noise(adv_noise.data)
            adv_noise.grad.zero_()

        # ---- Use adversarial noise to compute final loss ----
        final_x_t = sqrt_alpha_t * imgs + sqrt_one_minus_alpha_t * adv_noise
        eps_pred = unet(final_x_t, t).sample

        # ---- Similarity‑guided term ----
        with torch.no_grad():
            logits = classifier(final_x_t)
            probs = F.softmax(logits, dim=-1)
            # Target class is 1
            log_prob_target = torch.log(probs[:, 1] + 1e-12)
            # Gradient of log prob w.r.t. input
            grad_logp = torch.autograd.grad(log_prob_target.sum(),
                                            final_x_t,
                                            retain_graph=True,
                                            create_graph=True)[0]
        sim_guided = gamm = GAMMA * torch.sum((eps_pred - noise) * grad_logp, dim=[1,2,3])

        # ---- Total loss ----
        loss = F.mse_loss(eps_pred, noise) - sim_guided.mean()

        # ---- Backprop and update ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_file.write(f"Step {step:04d} | Loss: {loss.item():.4f}\n")
        if step % 50 == 0:
            print(f"Step {step:04d} | Loss: {loss.item():.4f}")

# ====================== Save fine‑tuned checkpoint ======================
torch.save(unet.state_dict(), "output/fine_tuned_model.pth")
print("Fine‑tuned model saved to output/fine_tuned_model.pth")

# ====================== Generate samples ======================
print("Generating samples...")
# Use the fine‑tuned UNet in a new pipeline
pipe_finetuned = DDPMPipeline(unet=unet, scheduler=scheduler)
pipe_finetuned = pipe_finetuned.to(DEVICE)

with torch.no_grad():
    for i in range(5):
        sample = pipe_finetuned(num_inference_steps=50, guidance_scale=0.0).images[0]
        sample.save(f"output/generated_samples/sample_{i}.png")
print("Generated 5 samples in output/generated_samples/")