"""
FOA core implementation.

This module contains:
- FOAModel: a wrapper that inserts a learnable prompt into a ViT backbone.
- FOA: the forward‑only adaptation algorithm (CMA‑ES + unsupervised fitness).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cma
from tqdm import tqdm


class FOAModel(nn.Module):
    """
    Wraps a pretrained ViT model and inserts a learnable prompt.
    The backbone parameters are frozen.
    """

    def __init__(self, backbone: nn.Module, num_prompts: int = 3):
        super().__init__()
        self.backbone = backbone
        self.num_prompts = num_prompts

        # Prompt shape: (num_prompts, dim) where dim is the patch embedding dim
        if not hasattr(backbone, "patch_embed"):
            raise ValueError("Backbone must expose a 'patch_embed' module.")
        dim = backbone.patch_embed.proj.out_channels

        # Learnable prompt embeddings (requires_grad=True)
        self.prompt = nn.Parameter(torch.zeros(num_prompts, dim))

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Forward pass with prompt injection.

        Returns:
            logits: (B, num_classes)
            cls_token: (B, dim) – CLS token after the final transformer block
        """
        B = x.size(0)

        # Patch embeddings: (B, N, D)
        patches = self.backbone.patch_embed(x)  # (B, N, D)

        # CLS token and prompt
        cls_token = self.backbone.cls_token.expand(B, -1, -1)  # (B, 1, D)
        prompt = self.prompt.expand(B, -1, -1)                # (B, P, D)

        # Concatenate: CLS, prompt, patches
        x = torch.cat((cls_token, prompt, patches), dim=1)    # (B, 1+P+N, D)

        # Add positional embedding and dropout
        # The original pos_embed has shape (1, N+1, D). We extend it for the
        # prompt tokens by appending zeros.
        pos_embed = self.backbone.pos_embed
        if pos_embed.shape[1] != patches.shape[1] + 1:
            raise RuntimeError("Unexpected positional embedding shape.")
        zero_prompt_pos = torch.zeros(
            1, self.num_prompts, pos_embed.shape[2], device=pos_embed.device
        )
        new_pos_embed = torch.cat([pos_embed, zero_prompt_pos], dim=1)
        x = self.backbone.pos_drop(x + new_pos_embed)

        # Transformer blocks
        for blk in self.backbone.blocks:
            x = blk(x)

        # Layer norm
        x = self.backbone.norm(x)

        # CLS token
        cls_token = x[:, 0]

        # Classification head
        logits = self.backbone.head(cls_token)

        return logits, cls_token


class FOA:
    """
    Forward‑only adaptation algorithm.

    Parameters
    ----------
    model : FOAModel
        The wrapped ViT model with prompt.
    source_stats : dict
        {'mu': tensor, 'sig': tensor} – CLS statistics from source data.
    lambda_discrep : float
        Weight for the activation‑discrepancy term in the fitness.
    population_size : int
        CMA‑ES population size.
    gamma_shift : float
        Step size for activation shifting.
    device : torch.device
    """

    def __init__(
        self,
        model: FOAModel,
        source_stats: dict,
        lambda_discrep: float = 0.4,
        population_size: int = 28,
        gamma_shift: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.source_mu = source_stats["mu"].to(device)
        self.source_sig = source_stats["sig"].to(device)
        self.lambda_discrep = lambda_discrep
        self.gamma_shift = gamma_shift

        # CMA‑ES configuration
        # The search space is the prompt vector: (P, D)
        self.dim = model.num_prompts * model.prompt.shape[1]
        self.cma_es = cma.CMAEvolutionStrategy(
            np.zeros(self.dim),  # initial mean
            0.1,                 # initial sigma (step size)
            {"popsize": population_size},
        )

        # Running mean of CLS tokens for activation shifting (EMA)
        self.test_mu = None
        self.alpha = 0.1  # EMA factor

    # --------------------------------------------------------------------- #
    # Utility methods
    # --------------------------------------------------------------------- #
    def _prompt_from_flat(self, flat: np.ndarray) -> torch.Tensor:
        """Reshape flat numpy array to prompt tensor."""
        prompt = torch.from_numpy(flat.astype(np.float32)).to(self.device)
        prompt = prompt.reshape(self.model.num_prompts, self.model.prompt.shape[1])
        return prompt

    def _compute_stats(self, cls_tokens: torch.Tensor):
        """Compute mean and std of CLS tokens."""
        mu = cls_tokens.mean(dim=0)
        sig = cls_tokens.std(dim=0)
        return mu, sig

    def _fitness(self, flat: np.ndarray, x: torch.Tensor):
        """
        Unsupervised fitness: entropy + λ * (μ‑diff + σ‑diff).

        Parameters
        ----------
        flat : np.ndarray
            Flattened prompt vector.
        x : torch.Tensor
            Input batch of images.

        Returns
        -------
        float
            Fitness value (lower is better).
        """
        prompt = self._prompt_from_flat(flat)
        with torch.no_grad():
            # Temporarily set prompt (no gradient tracking)
            self.model.prompt.copy_(prompt)

            logits, cls_tokens = self.model(x)
            probs = F.softmax(logits, dim=-1)

            # Entropy (mean over batch)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item()

            mu_batch, sig_batch = self._compute_stats(cls_tokens)
            mu_diff = torch.norm(mu_batch - self.source_mu, p=2).item()
            sig_diff = torch.norm(sig_batch - self.source_sig, p=2).item()

            return entropy + self.lambda_discrep * (mu_diff + sig_diff)

    def _shift_activation(self, cls_tokens: torch.Tensor):
        """Shift CLS token towards source mean using EMA‑based direction."""
        if self.test_mu is None:
            # Initialise EMA with the first batch mean
            self.test_mu = cls_tokens.mean(dim=0)
        else:
            # Update EMA
            batch_mean = cls_tokens.mean(dim=0)
            self.test_mu = self.alpha * batch_mean + (1 - self.alpha) * self.test_mu

        # Compute direction from current EMA to source mean
        shift = self.gamma_shift * (self.source_mu - self.test_mu)
        return cls_tokens + shift

    # --------------------------------------------------------------------- #
    # Core adaptation
    # --------------------------------------------------------------------- #
    def adapt_batch(self, x: torch.Tensor):
        """
        Run one CMA‑ES generation on batch x and return predictions.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images (B, C, H, W).

        Returns
        -------
        logits: torch.Tensor
            Logits after prompt adaptation and optional activation shifting.
        """
        # Sample candidate prompts
        candidates = self.cma_es.ask()
        fitness_vals = [self._fitness(c, x) for c in candidates]

        # Update CMA‑ES
        self.cma_es.tell(candidates, fitness_vals)

        # Pick best prompt
        best_idx = np.argmin(fitness_vals)
        best_prompt = self._prompt_from_flat(candidates[best_idx])

        # Apply best prompt
        with torch.no_grad():
            self.model.prompt.copy_(best_prompt)

            logits, cls_tokens = self.model(x)
            # Shift CLS tokens
            cls_shifted = self._shift_activation(cls_tokens)
            logits_shifted = self.model.backbone.head(cls_shifted)

        return logits_shifted

    def evaluate(self, loader: torch.utils.data.DataLoader):
        """Evaluate on a dataloader – returns top‑1 and top‑5 accuracy."""
        self.model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="FOA evaluation"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.adapt_batch(imgs)
                preds = logits.argmax(dim=1)
                correct_top1 += (preds == labels).sum().item()
                top5 = logits.topk(5, dim=1).indices
                correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)
        return 100.0 * correct_top1 / total, 100.0 * correct_top5 / total