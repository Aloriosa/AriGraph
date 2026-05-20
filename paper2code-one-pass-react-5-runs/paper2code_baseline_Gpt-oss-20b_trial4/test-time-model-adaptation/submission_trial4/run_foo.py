"""
Main script that demonstrates FOA on a small CIFAR‑10 test batch.
It prints predictions and writes them to output.json.
"""

import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import tqdm
import numpy as np
import pycma
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
PROMPT_LEN = 3
K_POP = 14              # population size for CMA‑ES
CMA_ITERS = 5           # number of CMA iterations per batch
LAMBDA = 0.4            # trade‑off between entropy and discrepancy
SHIFT_STEP = 1.0        # activation shifting step size

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def entropy(preds: torch.Tensor) -> torch.Tensor:
    """Compute mean entropy of predictions (batch x classes)."""
    eps = 1e-12
    logp = torch.log(preds + eps)
    return -torch.sum(preds * logp, dim=1).mean()

def cls_stats(model, x: torch.Tensor):
    """Return mean and std of CLS token over a batch."""
    cls = model.get_cls_token(x)  # shape: (B, D)
    return cls.mean(dim=0), cls.std(dim=0)

# --------------------------------------------------------------------------- #
# Model wrapper that adds learnable prompt tokens
# --------------------------------------------------------------------------- #

class PromptViT(timm.models.vision_transformer.ViTBase):
    """
    ViTBase with learnable prompt tokens added before the transformer blocks.
    The prompt is a trainable embedding of shape (prompt_len, D).
    """

    def __init__(self, prompt_len: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.prompt_len = prompt_len
        # Prompt embeddings (to be learned)
        self.prompt_emb = nn.Parameter(
            torch.randn(prompt_len, self.embed_dim) * 0.02
        )
        # Prompt positional embeddings (reuse first tokens of original pos_embed)
        # pos_embed shape: (1, N+1, D)
        self.prompt_pos = nn.Parameter(
            self.pos_embed[:, :prompt_len, :].clone()
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        """
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # (B, N+1, D)
        # Add prompt tokens
        prompt = self.prompt_emb.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)
        pos = torch.cat([self.prompt_pos, self.pos_embed], dim=1)  # (1, P+N+1, D)
        x = torch.cat([prompt, x], dim=1)  # (B, P+N+1, D)
        x = x + pos
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # CLS token is the first token after prompt
        cls = x[:, self.prompt_len, :]  # (B, D)
        logits = self.head(cls)
        return logits, cls

    def get_cls_token(self, x: torch.Tensor):
        """
        Return CLS token from the last layer before head.
        """
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # (B, N+1, D)
        # Add prompt tokens
        prompt = self.prompt_emb.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)
        pos = torch.cat([self.prompt_pos, self.pos_embed], dim=1)  # (1, P+N+1, D)
        x = torch.cat([prompt, x], dim=1)  # (B, P+N+1, D)
        x = x + pos
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, self.prompt_len, :]  # (B, D)
        return cls

# --------------------------------------------------------------------------- #
# FOA class
# --------------------------------------------------------------------------- #

class FOA:
    """
    Forward‑Optimization Adaptation using CMA‑ES on prompt tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        src_cls_mean: torch.Tensor,
        src_cls_std: torch.Tensor,
        prompt_len: int = 3,
        embed_dim: int = 768,
        k_pop: int = 14,
        cma_iters: int = 5,
        lambda_: float = 0.4,
        shift_step: float = 1.0,
    ):
        self.model = model
        self.src_cls_mean = src_cls_mean
        self.src_cls_std = src_cls_std
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.k_pop = k_pop
        self.cma_iters = cma_iters
        self.lambda_ = lambda_
        self.shift_step = shift_step
        self.model.eval()

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt prompt for the batch x and return predictions.
        """
        B = x.shape[0]
        # Flatten prompt parameters for CMA: shape (P*D,)
        prompt_shape = (self.prompt_len * self.embed_dim,)

        # Initialize CMA distribution
        mean = torch.zeros(prompt_shape, device=x.device)
        sigma = torch.ones(prompt_shape, device=x.device) * 0.1
        es = pycma.CMAEvolutionStrategy(mean.cpu().numpy(), sigma.cpu().numpy(), {'popsize': self.k_pop})

        best_prompt = None
        best_fitness = float("inf")

        for _ in range(self.cma_iters):
            # Sample K prompts
            solutions = es.ask()
            fitnesses = []

            for sol in solutions:
                # Convert sol to tensor
                prompt_vec = torch.tensor(sol, dtype=torch.float32, device=x.device)
                prompt_vec = prompt_vec.view(self.prompt_len, self.embed_dim)

                # Temporarily set the model's prompt to this sample
                orig_prompt = self.model.prompt_emb.data.clone()
                self.model.prompt_emb.data.copy_(prompt_vec)

                # Forward pass
                with torch.no_grad():
                    logits, cls = self.model(x)

                # Entropy loss (lower is better)
                probs = F.softmax(logits, dim=1)
                ent = entropy(probs)

                # Discrepancy between CLS stats and source stats
                mean_cls, std_cls = cls.mean(dim=0), cls.std(dim=0)
                disc = torch.norm(mean_cls - self.src_cls_mean, p=2) + torch.norm(std_cls - self.src_cls_std, p=2)

                # Total fitness
                fitness = ent.item() + self.lambda_ * disc.item()
                fitnesses.append(fitness)

                # Track best
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_prompt = prompt_vec.clone()

                # Restore original prompt
                self.model.prompt_emb.data.copy_(orig_prompt)

            # Update CMA distribution
            es.tell(solutions, fitnesses)
            es.disp()

        # After CMA, set the model's prompt to the best found
        self.model.prompt_emb.data.copy_(best_prompt.view(self.prompt_len, self.embed_dim))

        # Activation shifting on the CLS token
        with torch.no_grad():
            logits, cls = self.model(x)
            mean_cls, _ = cls.mean(dim=0), cls.std(dim=0)
            shift_dir = self.src_cls_mean - mean_cls
            # Shift CLS token
            cls_shifted = cls + self.shift_step * shift_dir
            # Replace CLS token in the internal representation
            # (skipping because we cannot modify the internal state; we just use cls_shifted for prediction)
            logits_shifted = self.model.head(cls_shifted)

        return logits_shifted

# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #

def main():
    # 1. Load CIFAR‑10 dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    cifar_train = datasets.CIFAR10(root=".", train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=".", train=False, download=True, transform=transform)

    # 2. Compute source CLS statistics from first 32 training images
    src_loader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=False)
    data_iter = iter(src_loader)
    src_imgs, _ = next(data_iter)
    src_imgs = src_imgs.to(DEVICE)

    # Load pre‑trained ViT‑Base
    model = PromptViT(prompt_len=PROMPT_LEN)
    model.load_state_dict(timm.models.vit_base_patch16_224(pretrained=True).state_dict())
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        _, src_cls = model(src_imgs)
    src_cls_mean = src_cls.mean(dim=0)
    src_cls_std = src_cls.std(dim=0)

    # 3. Prepare a test batch of 64 images
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=False)
    test_imgs, test_labels = next(iter(test_loader))
    test_imgs = test_imgs.to(DEVICE)

    # 4. Run FOA adaptation
    foa = FOA(
        model=model,
        src_cls_mean=src_cls_mean,
        src_cls_std=src_cls_std,
        prompt_len=PROMPT_LEN,
        embed_dim=model.embed_dim,
        k_pop=K_POP,
        cma_iters=CMA_ITERS,
        lambda_=LAMBDA,
        shift_step=SHIFT_STEP,
    )

    logits = foa.adapt_batch(test_imgs)
    probs = F.softmax(logits, dim=1)
    top5 = torch.topk(probs, 5, dim=1).indices.cpu().tolist()

    # 5. Save predictions
    output = []
    for idx, preds in enumerate(top5):
        output.append({"image_idx": idx, "top5": preds})
    Path("output.json").write_text(json.dumps(output, indent=2))
    print("Adaptation finished. Predictions written to output.json")

if __name__ == "__main__":
    main()