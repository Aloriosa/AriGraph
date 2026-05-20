#!/usr/bin/env python3
"""
FOA – Forward‑Only Adaptation for ViT
======================================

This script implements a lightweight, forward‑only test‑time adaptation
procedure for a pre‑trained ViT‑Base model on ImageNet‑C (severity level 5).
The implementation follows the key ideas from
“Test‑Time Model Adaptation with Only Forward Passes” (Niu et al., 2024):

  * A small learnable prompt is inserted at the input of the transformer.
  * The prompt is optimized online using the Covariance‑Matrix Adaptation
    Evolution Strategy (CMA‑ES) – a derivative‑free optimizer.
  * The fitness function is the entropy of the predictions plus an
    activation‑discrepancy term.
  * The final CLS activation is shifted toward the source (clean ImageNet)
    mean to reduce domain shift.
  * No gradients are ever computed – the procedure is fully forward‑only.

Author:  <your email here>
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pycma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
import timm
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dataset wrappers
# --------------------------------------------------------------------------- #
class ImageNetC(torch.utils.data.Dataset):
    """ImageNet‑C dataset (severity level 5)."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []

        # Ground‑truth file mapping validation image indices to class id
        gt_file = Path("ILSVRC2012_validation_ground_truth.txt")
        if not gt_file.exists():
            raise FileNotFoundError(
                f"Ground‑truth file {gt_file} not found – "
                "please run reproduce.sh first."
            )
        with open(gt_file, "r") as f:
            gt_list = [int(l.strip()) - 1 for l in f.readlines()]

        for corruption in sorted(os.listdir(self.root)):
            corr_dir = self.root / corruption / "5"
            if not corr_dir.is_dir():
                continue
            for img_file in sorted(os.listdir(corr_dir)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    # The corrupted file name matches the original one.
                    idx = int(img_file.split("_")[0]) - 1
                    if 0 <= idx < len(gt_list):
                        self.samples.append(corr_dir / img_file)
                        self.targets.append(gt_list[idx])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert("RGB")
        img = T.ToTensor()(img)
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageNetVal(torch.utils.data.Dataset):
    """ImageNet validation set (clean images)."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []

        gt_file = Path("ILSVRC2012_validation_ground_truth.txt")
        if not gt_file.exists():
            raise FileNotFoundError(
                f"Ground‑truth file {gt_file} not found – "
                "please run reproduce.sh first."
            )
        with open(gt_file, "r") as f:
            gt_list = [int(l.strip()) - 1 for l in f.readlines()]

        for file in sorted(os.listdir(self.root)):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_id = int(file.split(".")[0].split("_")[-1])
                if 1 <= img_id <= len(gt_list):
                    label = gt_list[img_id - 1]
                    self.samples.append(self.root / file)
                    self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert("RGB")
        img = T.ToTensor()(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------------- #
# Prompted ViT wrapper
# --------------------------------------------------------------------------- #
class PromptedViT(nn.Module):
    """
    Wrap a timm ViT model so that we can prepend a learnable prompt
    (sequence of Np tokens) before the patch tokens.
    """

    def __init__(self, base_model: nn.Module, prompt_len: int = 3):
        super().__init__()
        self.base = base_model
        self.prompt_len = prompt_len

        # Copy sub‑modules that we will use directly
        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.dist_token = getattr(base_model, "dist_token", None)
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.blocks = base_model.blocks
        self.norm = base_model.norm
        self.head = base_model.head

        # Learnable prompt
        self.prompt = nn.Parameter(
            torch.randn(prompt_len, self.patch_embed.proj.out_channels)
        )

    def forward(self, x: torch.Tensor, shift: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.  Returns logits and the CLS token before the head.
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, L, D)

        # Prompt tokens
        prompt = self.prompt.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)

        # Concatenate: [cls, prompt, patches] (+ dist token if present)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        if self.dist_token is not None:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_token, prompt, x, dist_token), dim=1)
        else:
            x = torch.cat((cls_token, prompt, x), dim=1)

        # Adjust positional embeddings
        pos = self.pos_embed
        zeros_prompt = torch.zeros(1, self.prompt_len, pos.shape[-1], device=x.device)
        pos = torch.cat((pos[:, :1], zeros_prompt, pos[:, 1:]), dim=1)

        x = x + pos
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Layer norm
        x = self.norm(x)

        # CLS token
        cls_token_final = x[:, 0]

        # Logits
        logits = self.head(cls_token_final)

        return logits, cls_token_final


# --------------------------------------------------------------------------- #
# Forward‑Only Adaptation (FOA) core
# --------------------------------------------------------------------------- #
class FOA:
    """
    Forward‑Only Adaptation engine.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 64,
        prompt_len: int = 3,
        popsize: int = 28,
        lambda_: float = 0.4,
        gamma: float = 1.0,
        alpha: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.popsize = popsize
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha

        # CMA‑ES initialisation
        dim = prompt_len * self.model.patch_embed.proj.out_channels
        self.cma_mean = torch.zeros(dim, device=device)
        self.cma = pycma.CMAEvolutionStrategy(
            self.cma_mean.tolist(), 0.05, {"popsize": self.popsize}
        )

        # Source statistics
        self.source_mean: torch.Tensor | None = None
        self.source_std: torch.Tensor | None = None

        # Running target mean for shifting
        self.target_mean: torch.Tensor | None = None

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def set_source_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.source_mean = mean
        self.source_std = std
        self.target_mean = mean.clone()

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        logp = torch.log(probs + 1e-12)
        return -torch.sum(probs * logp, dim=-1).mean()

    def fitness(self, logits: torch.Tensor, cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        Lower fitness is better.
        """
        ent = self.entropy(logits)
        if self.lambda_ > 0 and self.source_mean is not None:
            tgt_mean = cls_tokens.mean(0)
            tgt_std = cls_tokens.std(0)
            act_dis = torch.norm(tgt_mean - self.source_mean, 2) + torch.norm(tgt_std - self.source_std, 2)
            return ent + self.lambda_ * act_dis
        return ent

    def shift_cls(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        if self.source_mean is None:
            return cls_tokens
        return cls_tokens + self.gamma * (self.source_mean - self.target_mean)

    # --------------------------------------------------------------------- #
    # Adaptation loop
    # --------------------------------------------------------------------- #
    def adapt_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = images.shape[0]
        images = images.to(self.device)
        labels = labels.to(self.device)

        prompts = self.cma.ask()
        fitness_vals = []

        best_logits = None
        best_preds = None
        best_fitness = float("inf")

        for prompt_vec in prompts:
            prompt_tensor = torch.tensor(prompt_vec, dtype=torch.float32, device=self.device)
            self.model.prompt.data = prompt_tensor.reshape(
                self.prompt_len, self.model.patch_embed.proj.out_channels
            )

            logits, cls_tokens = self.model(images, shift=False)
            cls_tokens_shifted = self.shift_cls(cls_tokens)
            logits_shifted = self.model.head(cls_tokens_shifted)

            f_val = self.fitness(logits_shifted, cls_tokens_shifted).item()
            fitness_vals.append(f_val)

            if f_val < best_fitness:
                best_fitness = f_val
                best_logits = logits_shifted.detach()
                best_preds = torch.argmax(logits_shifted, dim=-1)

            # Update running target mean
            cls_mean_shifted = cls_tokens_shifted.mean(0)
            self.target_mean = self.alpha * cls_mean_shifted + (1 - self.alpha) * self.target_mean

        # CMA‑ES update
        self.cma.tell(prompts, fitness_vals)

        return best_logits, best_preds

    # --------------------------------------------------------------------- #
    # Evaluation helpers
    # --------------------------------------------------------------------- #
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for imgs, lbls in tqdm(dataloader, desc="Eval"):
                imgs = imgs.to(self.device)
                logits, _ = self.model(imgs, shift=False)
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(lbls)
                all_logits.append(logits.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        acc = (all_preds == all_labels).float().mean().item() * 100.0
        ece = self.compute_ece(all_logits, all_labels, num_bins=10)
        return acc, ece

    @staticmethod
    def compute_ece(logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE) with `num_bins` bins.
        """
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = predictions == labels

        ece = 0.0
        for i in range(num_bins):
            lower = i / num_bins
            upper = (i + 1) / num_bins
            in_bin = (confidences > lower) & (confidences <= upper)
            prop = in_bin.float().mean()
            if prop.item() > 0:
                bin_acc = accuracies[in_bin].float().mean()
                bin_conf = confidences[in_bin].mean()
                ece += torch.abs(bin_conf - bin_acc) * prop
        return ece.item() * 100.0  # percentage


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="FOA – Forward‑Only Adaptation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--prompt-len", type=int, default=3, help="Number of prompt tokens")
    parser.add_argument("--popsize", type=int, default=28, help="CMA population size")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.4,
                        help="Weight for activation discrepancy (default 0.4)")
    parser.add_argument("--gamma", type=float, default=1.0, help="CLS shift step size")
    parser.add_argument("--alpha", type=float, default=0.1, help="EMA factor for dynamic shift")
    parser.add_argument("--quantize", type=int, default=None,
                        help="Quantize model to 8‑bit (8) or 6‑bit (6)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # --------------------------------------------------------------------- #
    # Load datasets
    # --------------------------------------------------------------------- #
    imagenetc_root = Path("imagenet_c")
    if not imagenetc_root.exists():
        print("[ERROR] imagenet_c folder not found – run reproduce.sh first.")
        sys.exit(1)

    val_root = Path("ILSVRC2012_img_val")
    if not val_root.exists():
        print("[ERROR] ILSVRC2012_img_val folder not found – run reproduce.sh first.")
        sys.exit(1)

    transform = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

    imagenetc_loader = torch.utils.data.DataLoader(
        ImageNetC(imagenetc_root, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        ImageNetVal(val_root, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # --------------------------------------------------------------------- #
    # Load pre‑trained ViT‑Base
    # --------------------------------------------------------------------- #
    base_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)

    # --------------------------------------------------------------------- #
    # Compute source CLS statistics from 32 clean images
    # --------------------------------------------------------------------- #
    with torch.no_grad():
        cls_tokens_list = []
        count = 0
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            _, cls_tokens = base_model(imgs)
            cls_tokens_list.append(cls_tokens.cpu())
            count += imgs.shape[0]
            if count >= 32:
                break
        all_cls = torch.cat(cls_tokens_list, dim=0)
        source_mean = all_cls.mean(0).to(device)
        source_std = all_cls.std(0).to(device)

    print("[INFO] Source CLS mean and std computed.")

    # --------------------------------------------------------------------- #
    # Quantize model if requested
    # --------------------------------------------------------------------- #
    if args.quantize == 8:
        print("[INFO] Applying 8‑bit dynamic quantization.")
        base_model = quantization.quantize_dynamic(
            base_model,
            {nn.Linear, nn.Conv2d, nn.MultiheadAttention},
            dtype=torch.qint8,
        )
    elif args.quantize == 6:
        print("[WARN] 6‑bit quantization not supported by dynamic quantizer; using 8‑bit.")
        base_model = quantization.quantize_dynamic(
            base_model,
            {nn.Linear, nn.Conv2d, nn.MultiheadAttention},
            dtype=torch.qint8,
        )
    elif args.quantize is not None:
        print(f"[WARN] Unknown quantize option {args.quantize}; ignoring.")

    # --------------------------------------------------------------------- #
    # Wrap with prompt
    # --------------------------------------------------------------------- #
    prompted_model = PromptedViT(base_model, prompt_len=args.prompt_len)
    prompted_model.eval()

    # --------------------------------------------------------------------- #
    # Initialise FOA engine
    # --------------------------------------------------------------------- #
    foa = FOA(
        prompted_model,
        device,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        popsize=args.popsize,
        lambda_=args.lambda_,
        gamma=args.gamma,
        alpha=args.alpha,
    )
    foa.set_source_stats(source_mean, source_std)

    # --------------------------------------------------------------------- #
    # Adaptation loop over the test set
    # --------------------------------------------------------------------- #
    all_preds = []
    all_logits = []
    all_labels = []

    start_time = time.time()
    for imgs, lbls in tqdm(imagenetc_loader, desc="FOA Adaptation"):
        logits, preds = foa.adapt_batch(imgs, lbls)
        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(lbls)

    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item() * 100.0
    ece = foa.compute_ece(all_logits, all_labels, num_bins=10)
    elapsed = time.time() - start_time

    print("\n=== FOA Results ===")
    print(f"Accuracy: {acc:.2f}%")
    print(f"ECE: {ece:.2f}%")
    print(f"Time: {elapsed/60:.1f} min")

    # Save to results.txt
    results_path = Path("results.txt")
    results_path.write_text(
        f"Accuracy: {acc:.2f}%\nECE: {ece:.2f}%\nTime: {elapsed/60:.1f} min\n"
    )


if __name__ == "__main__":
    main()