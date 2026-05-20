#!/usr/bin/env python3
"""
APGD attack implementation used by train_fare.py and demo.py.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def apgd_attack(
    model,
    images,
    epsilon,
    steps,
    step_size,
    half_precision=True,
    device=None,
):
    """
    Approximate PGD (APGD) attack.
    Arguments
    ---------
    model : nn.Module
        Vision encoder with `encode_image` method.
    images : torch.Tensor
        Batch of clean images in range [-1, 1].
    epsilon : float
        Perturbation bound (ℓ∞).
    steps : int
        Number of iterations.
    step_size : float
        Step size for each iteration.
    half_precision : bool
        Run the fast half‑precision loop first.
    device : torch.device | None
        Device to run the attack on.
    Returns
    -------
    torch.Tensor
        Adversarial images in range [-1, 1].
    """
    model.eval()
    device = device or torch.device("cpu")
    images = images.to(device, dtype=torch.float32).detach()

    # 1. Half‑precision loop for speed
    if half_precision:
        images_hp = images.half()
        perturb_hp = torch.zeros_like(images_hp, device=device)
        for _ in range(steps):
            perturb_hp.requires_grad = True
            pert_imgs = torch.clamp(images_hp + perturb_hp, -1.0, 1.0)
            out = model.encode_image(pert_imgs)
            loss = -out.norm(dim=-1).mean()
            loss.backward()
            grad = perturb_hp.grad
            perturb_hp = perturb_hp + step_size * grad.sign()
            perturb_hp = torch.clamp(perturb_hp, -epsilon, epsilon)
            perturb_hp = perturb_hp.detach()
        perturb = perturb_hp.to(torch.float32)
    else:
        perturb = torch.zeros_like(images, device=device)

    # 2. Single‑precision refinement
    for _ in range(10):
        perturb.requires_grad = True
        pert_imgs = torch.clamp(images + perturb, -1.0, 1.0)
        out = model.encode_image(pert_imgs)
        loss = -out.norm(dim=-1).mean()
        loss.backward()
        grad = perturb.grad
        perturb = perturb + step_size * grad.sign()
        perturb = torch.clamp(perturb, -epsilon, epsilon)
        perturb = perturb.detach()

    adv_imgs = torch.clamp(images + perturb, -1.0, 1.0)
    return adv_imgs