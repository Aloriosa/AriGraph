"""
FARE: Unsupervised Adversarial Fine‑Tuning loss for CLIP vision encoder.
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def fare_loss(encoder, images, eps=4 / 255, pgd_steps=10, step_size=1 / 255):
    """
    Compute the FARE loss for a batch of images.
    Args:
        encoder: CLIP vision encoder (nn.Module)
        images: input images, shape (B, 3, H, W), pixel values in [0,1]
        eps: adversarial perturbation radius (float)
        pgd_steps: number of PGD iterations
        step_size: PGD step size
    Returns:
        loss: scalar tensor
    """
    # Ensure encoder is in training mode
    encoder.train()
    # Compute *original* embeddings (no grad)
    with torch.no_grad():
        orig_emb = encoder(images).last_hidden_state[:, 0, :]  # CLS token

    # Initialise adversarial images
    adv = images.clone().detach()
    adv.requires_grad_(True)

    for _ in range(pgd_steps):
        # Forward pass
        adv_emb = encoder(adv).last_hidden_state[:, 0, :]

        # Loss: L2 between adv and orig embeddings
        loss = F.mse_loss(adv_emb, orig_emb)

        # Backward
        loss.backward()

        # PGD update
        grad = adv.grad.data
        adv.data = adv.data + step_size * grad.sign()
        # Project back to epsilon ball
        adv.data = torch.max(torch.min(adv.data, images + eps), images - eps)
        adv.data = torch.clamp(adv.data, 0.0, 1.0)

        # Zero gradients for next step
        adv.grad.zero_()

    # After PGD, compute final loss
    final_adv_emb = encoder(adv).last_hidden_state[:, 0, :]
    final_loss = F.mse_loss(final_adv_emb, orig_emb)
    return final_loss