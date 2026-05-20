#!/usr/bin/env python
"""
Evaluation of the fine‑tuned CLIP model on clean and adversarial ImageNet
validation images.  Accuracy is computed using the 1000 ImageNet class
names and the cosine similarity between image and text embeddings.

The script outputs a plain text file `results.txt` containing:
  clean_accuracy
  robust_accuracy_eps2
  robust_accuracy_eps4
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apgd_attack(
    model: CLIPModel,
    images: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    device: torch.device,
    dtype=torch.float32,
):
    """
    APGD attack with respect to the squared L2 loss between clean and perturbed
    embeddings.  The attack can be run in half‑precision (dtype=torch.float16)
    or single‑precision (dtype=torch.float32).
    """
    images_adv = images.clone().detach().requires_grad_(True).to(dtype)
    for _ in range(steps):
        # forward clean
        with torch.no_grad():
            feat_clean = model.get_image_features(images).to(dtype)
        # forward perturbed
        feat_adv = model.get_image_features(images_adv)
        loss = nn.functional.mse_loss(feat_adv, feat_clean)
        loss.backward()

        # gradient sign
        grad_sign = images_adv.grad.sign()
        images_adv = images_adv.detach() + step_size * grad_sign
        # project onto ε‑ball
        perturbation = torch.clamp(images_adv - images.to(dtype), min=-eps, max=eps)
        images_adv = torch.clamp(images.to(dtype) + perturbation, min=0.0, max=1.0).detach()
        images_adv.requires_grad_(True)

    return images_adv.detach()

def single_precision_refine(model, images_adv, eps, steps, step_size, device):
    """Refine an existing adversarial image with single‑precision PGD."""
    return apgd_attack(model, images_adv, eps, steps, step_size, device, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if os.path.exists(args.ckpt):
        clip.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint from {args.ckpt}")
    clip.to(device)
    clip.eval()

    # Data transforms
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Validation split (full ImageNet validation set)
    val_ds = load_dataset("imagenet", split="validation")

    def collate_fn(batch):
        images = [preprocess(item["image"]) for item in batch]
        images = torch.stack(images)
        return images

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load ImageNet class names (1000 official names)
    with open("imagenet_classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Encode class names once
    processor = CLIPProcessor()
    text_inputs = processor(text=class_names, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embeds = clip.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Evaluation
    correct_clean = 0
    correct_eps2 = 0
    correct_eps4 = 0
    total = 0

    for images in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        batch_size = images.size(0)

        # Clean accuracy
        with torch.no_grad():
            feats_clean = clip.get_image_features(images)
            feats_clean = feats_clean / feats_clean.norm(dim=-1, keepdim=True)
        logits_clean = torch.matmul(feats_clean, text_embeds.t())
        preds_clean = logits_clean.argmax(dim=-1)
        correct_clean += (preds_clean == torch.arange(batch_size, device=device)).sum().item()

        # ε = 2/255 adversarial
        # Half‑precision APGD (100 steps)
        adv2_half = apgd_attack(clip, images, eps=2/255.0, steps=100,
                                step_size=2/255.0/100, device=device,
                                dtype=torch.float16)
        # Single‑precision refinement (10 steps)
        adv2 = single_precision_refine(clip, adv2_half, eps=2/255.0,
                                       steps=10,
                                       step_size=2/255.0/10,
                                       device=device)
        feats_adv2 = clip.get_image_features(adv2)
        feats_adv2 = feats_adv2 / feats_adv2.norm(dim=-1, keepdim=True)
        logits_adv2 = torch.matmul(feats_adv2, text_embeds.t())
        preds_adv2 = logits_adv2.argmax(dim=-1)
        correct_eps2 += (preds_adv2 == torch.arange(batch_size, device=device)).sum().item()

        # ε = 4/255 adversarial
        adv4_half = apgd_attack(clip, images, eps=4/255.0, steps=100,
                                step_size=4/255.0/100, device=device,
                                dtype=torch.float16)
        adv4 = single_precision_refine(clip, adv4_half, eps=4/255.0,
                                       steps=10,
                                       step_size=4/255.0/10,
                                       device=device)
        feats_adv4 = clip.get_image_features(adv4)
        feats_adv4 = feats_adv4 / feats_adv4.norm(dim=-1, keepdim=True)
        logits_adv4 = torch.matmul(feats_adv4, text_embeds.t())
        preds_adv4 = logits_adv4.argmax(dim=-1)
        correct_eps4 += (preds_adv4 == torch.arange(batch_size, device=device)).sum().item()

        total += batch_size

    clean_acc  = correct_clean / total
    robust_acc_eps2 = correct_eps2 / total
    robust_acc_eps4 = correct_eps4 / total

    print(f"Clean accuracy:      {clean_acc*100:.2f}%")
    print(f"Robust accuracy (ε=2/255):  {robust_acc_eps2*100:.2f}%")
    print(f"Robust accuracy (ε=4/255):  {robust_acc_eps4*100:.2f}%")

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write(f"clean_accuracy:      {clean_acc*100:.2f}%\n")
        f.write(f"robust_accuracy_eps2:  {robust_acc_eps2*100:.2f}%\n")
        f.write(f"robust_accuracy_eps4:  {robust_acc_eps4*100:.2f}%\n")
    print(f"Results written to {os.path.join(args.out_dir, 'results.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP and FARE‑CLIP on clean and adversarial accuracy."
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/clip_fare.pt",
                        help="Path to the fine‑tuned checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Directory to store the results.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()
    main(args)