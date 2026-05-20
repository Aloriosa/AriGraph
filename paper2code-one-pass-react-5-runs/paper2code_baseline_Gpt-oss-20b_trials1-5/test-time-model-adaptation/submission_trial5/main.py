#!/usr/bin/env python3
"""
Minimal FOA implementation for reproducibility.
"""

import argparse
import random
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import cma

from utils import entropy, to_device, clip_tensor

# --------------------------------------------------------------
# 1. Argument parsing
# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="FOA – Forward‑Optimization Adaptation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for test data")
    parser.add_argument("--prompt_size", type=int, default=3, help="Number of prompt embeddings")
    parser.add_argument("--population", type=int, default=6, help="CMA population size")
    parser.add_argument("--lambda_fitness", type=float, default=0.4, help="Weight for activation discrepancy")
    parser.add_argument("--max_iters", type=int, default=25, help="Number of batches to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# --------------------------------------------------------------
# 2. Prompted Vision Transformer
# --------------------------------------------------------------
class PromptedViT(nn.Module):
    """
    Wrap a pretrained ViT and prepend learnable prompt embeddings.
    """
    def __init__(self, model_name: str, prompt_size: int, device: torch.device):
        super().__init__()
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1000)
        self.model.eval()
        self.model.to(device)

        # Prompt embedding dimension matches the patch embedding dim (768 for ViT-B/16)
        self.prompt_dim = self.model.patch_embed.proj.out_channels
        self.prompt_size = prompt_size
        # Prompt embeddings are not learned in this script; they are supplied externally.
        # We just reserve a buffer to concatenate later.
        self.prompt_buffer = None  # will be set to a tensor of shape (prompt_size, prompt_dim)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor = None):
        """
        Forward pass with optional prompt.

        Args:
            x: Input images, shape (B, 3, H, W)
            prompt: Optional prompt tensor, shape (prompt_size, prompt_dim)
        Returns:
            logits: (B, num_classes)
            cls_token: (B, prompt_size + 1, prompt_dim)  # CLS token is the first
        """
        B = x.shape[0]
        # Get patch embeddings
        patches = self.model.patch_embed(x)  # (B, num_patches, dim)

        # Prepend prompt embeddings
        if prompt is not None:
            # prompt: (prompt_size, dim)
            prompt = prompt.unsqueeze(0).expand(B, -1, -1)  # (B, prompt_size, dim)
            patches = torch.cat([prompt, patches], dim=1)

        # Add positional embeddings
        patches = self.model.pos_drop(patches + self.model.pos_embed)

        # Transformer encoder
        for block in self.model.blocks:
            patches = block(patches)

        # CLS token (first token)
        cls_token = patches[:, 0]  # (B, dim)

        # Classification head
        logits = self.model.head(cls_token)

        return logits, cls_token

    def get_source_stats(self, src_loader, device: torch.device):
        """
        Compute mean and std of CLS token activations on source data.
        """
        self.eval()
        with torch.no_grad():
            all_cls = []
            for imgs, _ in src_loader:
                imgs = imgs.to(device)
                _, cls = self.forward(imgs)
                all_cls.append(cls.cpu())
            all_cls = torch.cat(all_cls, dim=0)
        mu = all_cls.mean(dim=0)
        sigma = all_cls.std(dim=0)
        return mu, sigma


# --------------------------------------------------------------
# 3. Fitness function
# --------------------------------------------------------------
def compute_fitness(logits, cls_token, src_mu, src_sigma, lambda_fitness):
    """
    Compute the fitness value for a prompt.

    Args:
        logits: (B, num_classes)
        cls_token: (B, dim)
        src_mu: (dim,)
        src_sigma: (dim,)
        lambda_fitness: weight scalar
    Returns:
        fitness: scalar (lower is better)
    """
    probs = F.softmax(logits, dim=1)
    ent = entropy(probs).mean()  # mean entropy over batch

    # Activation discrepancy: L2 distance between batch mean/std and source mean/std
    batch_mu = cls_token.mean(dim=0)
    batch_sigma = cls_token.std(dim=0)

    act_discrepancy = torch.norm(batch_mu - src_mu) + torch.norm(batch_sigma - src_sigma)

    fitness = ent + lambda_fitness * act_discrepancy
    return fitness.item()


# --------------------------------------------------------------
# 4. Main experimental loop
# --------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 4.1 Load datasets
    # ------------------------------------------------------------------
    # Use the HuggingFace 'imagenet_c' dataset (source: https://huggingface.co/datasets/huggingface/imagenet_c)
    # We'll use the 'train' split as a small source set (32 samples) and the 'test' split as the OOD test set (200 samples).
    dataset_name = "huggingface/imagenet_c"
    print(f"Downloading '{dataset_name}' dataset (train split, 32 images)...")
    ds_source = load_dataset(dataset_name, split="train[:32]")  # 32 source samples
    print(f"Downloading '{dataset_name}' dataset (test split, 200 images)...")
    ds_test = load_dataset(dataset_name, split="test[:200]")   # 200 test samples

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def collate_fn(batch):
        imgs = []
        labels = []
        for item in batch:
            img = transform(item["image"])
            imgs.append(img)
            labels.append(item["label"])
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels, dtype=torch.long)
        return imgs, labels

    src_loader = DataLoader(ds_source, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # 4.2 Initialise model
    # ------------------------------------------------------------------
    model_name = "vit_base_patch16_224"
    print(f"Loading pretrained {model_name} from timm.")
    prompt_model = PromptedViT(model_name, prompt_size=args.prompt_size, device=device)
    prompt_model.eval()

    # Compute source CLS statistics (mu, sigma)
    src_mu, src_sigma = prompt_model.get_source_stats(src_loader, device)
    src_mu = src_mu.to(device)
    src_sigma = src_sigma.to(device)

    # ------------------------------------------------------------------
    # 4.3 CMA-ES setup
    # ------------------------------------------------------------------
    prompt_dim = prompt_model.prompt_dim
    num_params = args.prompt_size * prompt_dim  # total number of prompt parameters
    # Initial mean and covariance for CMA-ES
    mean = np.zeros(num_params, dtype=np.float64)
    sigma_cma = 0.3  # initial standard deviation
    es = cma.CMAEvolutionStrategy(mean, sigma_cma, {"popsize": args.population})

    # ------------------------------------------------------------------
    # 4.4 Adaptation loop
    # ------------------------------------------------------------------
    total_correct = 0
    total_samples = 0
    print(f"Running FOA adaptation on {len(test_loader.dataset)} test images...")
    for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader), 1):
        if batch_idx > args.max_iters:
            break  # limit number of batches for quick demo

        imgs = imgs.to(device)
        labels = labels.to(device)

        # CMA-ES generates a population of prompts
        pop = es.ask()
        fitnesses = []
        logits_list = []
        cls_list = []

        for candidate in pop:
            # candidate shape: (num_params,)
            prompt_tensor = torch.tensor(candidate, dtype=torch.float32, device=device)
            prompt_tensor = prompt_tensor.view(args.prompt_size, prompt_dim)

            # Forward pass
            with torch.no_grad():
                logits, cls_token = prompt_model.forward(imgs, prompt_tensor)

            # Compute fitness
            fit = compute_fitness(logits, cls_token, src_mu, src_sigma, args.lambda_fitness)
            fitnesses.append(fit)
            logits_list.append(logits)
            cls_list.append(cls_token)

        # CMA-ES update
        es.tell(pop, fitnesses)

        # Select best candidate for inference
        best_idx = np.argmin(fitnesses)
        best_logits = logits_list[best_idx]
        preds = best_logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        print(f"Batch {batch_idx}/{len(test_loader)} - Best fitness: {fitnesses[best_idx]:.4f}")

    accuracy = 100.0 * total_correct / total_samples
    print(f"\nAccuracy on test subset: {accuracy:.2f}%")

if __name__ == "__main__":
    main()