# foa.py
"""
Core FOA implementation:
- Prompt learning via CMA‑ES
- Unsupervised fitness (entropy + activation discrepancy)
- Back‑to‑source activation shifting
"""

import torch
import numpy as np
import types
import math


class FOA:
    """
    Forward‑Optimization Adaptation (FOA) for a frozen ViT model.
    """
    def __init__(
        self,
        model,
        device,
        prompt_dim: int,
        n_prompt: int,
        source_stats: dict,
        lambda_reg: float = 0.4,
        gamma: float = 1.0,
        popsize: int = 28,
        cma_iters: int = 5,
        verbose: bool = False,
    ):
        """
        Args:
            model: Frozen ViT model (timm).
            device: torch device.
            prompt_dim: dimension of a single prompt token (e.g. 768).
            n_prompt: number of prompt tokens.
            source_stats: dict with keys 'mu' and 'sigma' (CLS statistics on source).
            lambda_reg: trade‑off weight for activation discrepancy.
            gamma: step size for activation shifting.
            popsize: population size for CMA‑ES.
            cma_iters: number of CMA iterations per batch.
            verbose: if True, prints CMA progress.
        """
        self.model = model
        self.device = device
        self.prompt_dim = prompt_dim
        self.n_prompt = n_prompt
        self.source_mu = source_stats["mu"].to(device)
        self.source_sigma = source_stats["sigma"].to(device)
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.popsize = popsize
        self.cma_iters = cma_iters
        self.verbose = verbose

        # Ensure model weights are frozen
        for p in self.model.parameters():
            p.requires_grad = False

        # Monkey‑patch forward_features to accept prompt tokens
        self._patch_forward_features()

        # CMA‑ES state
        self.cma_mean = np.zeros(self.n_prompt * self.prompt_dim, dtype=np.float64)
        self.cma_sigma = 0.1  # initial step size
        self.cma_cov = np.eye(self.n_prompt * self.prompt_dim, dtype=np.float64)

    def _patch_forward_features(self):
        """
        Insert prompt tokens into the forward pass.
        The prompt is inserted after the CLS token.
        """
        def forward_features_with_prompt(self, x, prompt):
            # x: (B, 3, H, W)
            # prompt: (B, n_prompt, dim)
            patch_embed = self.patch_embed(x)          # (B, num_patches, dim)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, dim)
            # Concatenate CLS, prompt, patches
            x = torch.cat((cls_token, prompt, patch_embed), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x

        self.model.forward_features = types.MethodType(forward_features_with_prompt, self.model)

    def _forward_with_prompt(self, images, prompt):
        """
        Forward pass using a given prompt.
        Returns:
            logits: (B, num_classes)
            cls_token: (B, dim)
        """
        # Expand prompt for batch
        B = images.shape[0]
        prompt_exp = prompt.unsqueeze(0).expand(B, -1, -1)  # (B, n_prompt, dim)

        features = self.model.forward_features(images, prompt_exp)  # (B, N, dim)
        cls_token = features[:, 0, :]  # CLS token before shifting

        # Activation shifting
        batch_mu = cls_token.mean(dim=0)
        cls_token_shifted = cls_token + self.gamma * (self.source_mu - batch_mu)

        logits = self.model.head(cls_token_shifted)
        return logits, cls_token

    def _entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        logp = torch.log(probs + 1e-12)
        ent = -torch.sum(probs * logp, dim=-1)  # (B,)
        return ent.mean().item()

    def _activation_discrepancy(self, cls_token):
        mu = cls_token.mean(dim=0)
        sigma = cls_token.std(dim=0)
        diff_mu = torch.norm(mu - self.source_mu).item()
        diff_sigma = torch.norm(sigma - self.source_sigma).item()
        return diff_mu + diff_sigma

    def _fitness(self, prompt_vec, images):
        """
        Compute fitness for a single prompt vector.
        Lower fitness is better.
        """
        prompt = torch.from_numpy(prompt_vec.reshape(self.n_prompt, self.prompt_dim)).float().to(self.device)
        logits, cls_token = self._forward_with_prompt(images, prompt)
        ent = self._entropy(logits)
        act_diff = self._activation_discrepancy(cls_token)
        fitness = ent + self.lambda_reg * act_diff
        return fitness

    def adapt_batch(self, images):
        """
        Run CMA‑ES for a single batch and return the best prompt.
        """
        # Use a simple CMA‑ES loop
        mean = self.cma_mean.copy()
        sigma = self.cma_sigma
        cov = self.cma_cov.copy()

        for _ in range(self.cma_iters):
            # Sample population
            samples = mean + sigma * np.random.multivariate_normal(np.zeros_like(mean), cov, self.popsize)

            fitnesses = []
            for s in samples:
                f = self._fitness(s, images.to(self.device))
                fitnesses.append(f)

            # Rank and compute weighted mean of top 25%
            sorted_idx = np.argsort(fitnesses)
            topk = int(0.25 * self.popsize)
            elite = samples[sorted_idx[:topk]]
            mean = elite.mean(axis=0)

            # Update covariance (simple approximation)
            diff = elite - mean
            cov = np.cov(diff, rowvar=False) + 1e-8 * np.eye(cov.shape[0])

            # Step size adaptation (decrease sigma)
            sigma *= 0.8

            if self.verbose:
                best = fitnesses[sorted_idx[0]]
                print(f"  CMA iter: {_+1:02d}  best fitness: {best:.4f}")

        # Store updated CMA state (optional)
        self.cma_mean = mean
        self.cma_sigma = sigma
        self.cma_cov = cov

        # Return best prompt
        best_prompt = mean.reshape(self.n_prompt, self.prompt_dim)
        return torch.from_numpy(best_prompt).float().to(self.device)

    def predict(self, images, prompt):
        """
        Predict logits for given images and prompt.
        """
        logits, _ = self._forward_with_prompt(images, prompt)
        return logits