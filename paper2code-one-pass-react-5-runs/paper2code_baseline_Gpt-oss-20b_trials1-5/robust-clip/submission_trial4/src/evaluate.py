#!/usr/bin/env python3
"""
Evaluate a CLIP vision encoder on clean and adversarial images.
Uses zero‑shot classification with CLIP text embeddings.

Usage:
    python src/evaluate.py \
        --model finetuned_clip.pt \
        --dataset cifar10 \
        --eps 4/255
"""
import argparse
import os
import sys
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

def parse_eps(eps_str):
    if '/' in eps_str:
        num, den = eps_str.split('/')
        return float(num) / float(den)
    return float(eps_str)

def get_dataloaders(dataset_name, batch_size=128):
    if dataset_name.lower() == 'cifar10':
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

def zero_shot_logits(images, model, text_tokens, device):
    """Return logits for zero‑shot classification."""
    with torch.no_grad():
        image_features = model.encode_image(images.to(device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(text_tokens.to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
    return logits

def pgd_attack(model, images, eps, alpha, iters, device):
    """PGD attack on the image.  We use the same procedure as in training."""
    images_adv = images.clone().detach().to(device)
    images_adv.requires_grad = True

    # Pre‑compute clean embeddings
    with torch.no_grad():
        clean_emb = model.encode_image(images.to(device))

    for _ in range(iters):
        adv_emb = model.encode_image(images_adv)
        loss = -torch.mean((adv_emb - clean_emb)**2)
        loss.backward()

        grad_sign = images_adv.grad.sign()
        images_adv = images_adv + alpha * grad_sign
        eta = torch.clamp(images_adv - images.to(device), min=-eps, max=eps)
        images_adv = torch.clamp(images.to(device) + eta, min=0, max=1).detach()
        images_adv.requires_grad = True

    return images_adv

def main():
    parser = argparse.ArgumentParser(description="Zero‑shot eval of CLIP")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the fine‑tuned CLIP weights")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--eps", type=str, default="4/255")
    parser.add_argument("--alpha", type=float, default=1/255)
    parser.add_argument("--adv-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model
    model, preprocess = clip.load("ViT-B/32", device=device)
    # Load fine‑tuned weights only for the vision encoder
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_loader = get_dataloaders(args.dataset, args.batch_size)

    # Prepare text prompts for the 10 CIFAR‑10 classes
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    prompts = [f"a photo of a {name}" for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)

    eps = parse_eps(args.eps)
    alpha = args.alpha

    clean_acc = 0
    adv_acc = 0
    total = 0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Clean
        logits = zero_shot_logits(imgs, model, text_tokens, device)
        preds = logits.argmax(dim=-1)
        clean_acc += (preds == labels).sum().item()

        # Adversarial
        imgs_adv = pgd_attack(
            model, imgs, eps, alpha, args.adv_steps, device
        )
        logits_adv = zero_shot_logits(imgs_adv, model, text_tokens, device)
        preds_adv = logits_adv.argmax(dim=-1)
        adv_acc += (preds_adv == labels).sum().item()

        total += imgs.size(0)

    clean_acc = 100.0 * clean_acc / total
    adv_acc = 100.0 * adv_acc / total

    print(f"Clean accuracy  : {clean_acc:.2f}%")
    print(f"Adversarial accuracy (ε={eps:.4f}) : {adv_acc:.2f}%")

if __name__ == "__main__":
    main()