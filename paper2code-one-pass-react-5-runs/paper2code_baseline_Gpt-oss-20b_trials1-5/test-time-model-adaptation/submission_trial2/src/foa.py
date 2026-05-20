import torch
import torch.nn as nn
import numpy as np
import cma
from typing import List, Tuple
from .utils import entropy, mean_std, moving_average


class PromptedViT(nn.Module):
    """
    Wraps a ViT from timm and injects learnable prompt tokens.
    The prompt tokens are placed *before* the original patch embeddings.
    """
    def __init__(self, base_model: nn.Module, num_prompt: int):
        super().__init__()
        self.base = base_model
        self.num_prompt = num_prompt
        self.hidden_dim = base_model.patch_embed.proj.out_channels  # 768 for ViT‑Base
        # Prompt buffer (to be updated by CMA‑ES)
        self.register_buffer(
            "prompt", torch.zeros(num_prompt, self.hidden_dim)
        )

    def forward(self, x: torch.Tensor, source_mean: torch.Tensor = None,
                gamma: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with prompt and optional activation shifting.

        Returns
        -------
        logits : (B, C)
        cls_tokens : list of CLS tokens from each block
        """
        # Patch embeddings
        patch_emb = self.base.patch_embed(x)          # (B, N+1, D)
        patch_emb = patch_emb[:, 1:]                  # remove original CLS token

        # Expand prompt to batch
        batch = x.size(0)
        prompt = self.prompt.unsqueeze(0).expand(batch, -1, -1)  # (B, P, D)
        x = torch.cat([prompt, patch_emb], dim=1)                # (B, P+N, D)

        cls_tokens = []
        for blk in self.base.blocks:
            x = blk(x)
            cls_tokens.append(x[:, 0])  # CLS token after this block

        # Activation shifting (back‑to‑source)
        if source_mean is not None:
            target_mean = torch.mean(cls_tokens[-1], dim=0)  # (D,)
            shift = gamma * (source_mean - target_mean)     # (D,)
            x[:, 0] += shift

        # Classifier
        x = self.base.norm(x)
        logits = self.base.head(x[:, 0])
        return logits, cls_tokens


class FOA:
    """
    Forward‑Optimization Adaptation (FOA) implementation.
    """
    def __init__(self,
                 model: nn.Module,
                 source_stats: Tuple[List[torch.Tensor], List[torch.Tensor]],
                 device: torch.device,
                 num_prompt: int = 3,
                 lambda_: float = 0.4,
                 popsize: int = 28,
                 num_generations: int = 1,
                 gamma: float = 1.0):
        """
        Parameters
        ----------
        model : nn.Module
            Base ViT model (e.g. timm.create_model('vit_base_patch16_224', pretrained=True)).
        source_stats : (means, stds)
            Pre‑computed CLS token statistics for each block (source domain).
        device : torch.device
            Device for computation.
        num_prompt : int
            Number of prompt tokens to learn.
        lambda_ : float
            Weight for the activation‑discrepancy term in fitness.
        popsize : int
            CMA‑ES population size.
        num_generations : int
            CMA‑ES generations per batch.
        gamma : float
            Step size for activation shifting.
        """
        self.device = device
        self.num_prompt = num_prompt
        self.hidden_dim = model.patch_embed.proj.out_channels
        self.prompted = PromptedViT(model, num_prompt).to(device)
        self.source_means, self.source_stds = source_stats
        self.lambda_ = lambda_
        self.popsize = popsize
        self.num_generations = num_generations
        self.gamma = gamma

        # CMA‑ES initial parameters (mean vector, sigma)
        self.cma_mean = np.zeros(num_prompt * self.hidden_dim)
        self.cma_sigma = 0.1

    def _fitness(self, prompt_vec: np.ndarray,
                 batch: torch.Tensor,
                 targets: torch.Tensor) -> float:
        """
        Compute fitness for a single candidate prompt.
        Lower fitness -> better (entropy + discrepancy).
        """
        prompt = torch.from_numpy(prompt_vec).float().to(self.device)
        prompt = prompt.view(self.num_prompt, self.hidden_dim)
        self.prompted.prompt.copy_(prompt)

        logits, cls_tokens = self.prompted(batch, source_mean=self.source_means[-1],
                                          gamma=self.gamma)
        # Entropy term
        ent = entropy(logits).mean().item()

        # Activation discrepancy term
        diff = 0.0
        for i, tokens in enumerate(cls_tokens):
            mean_i, std_i = mean_std(tokens, dim=0)
            diff += torch.norm(mean_i - self.source_means[i]).item()
            diff += torch.norm(std_i - self.source_stds[i]).item()

        loss = ent + self.lambda_ * diff
        # CMA‑ES maximizes fitness, so return negative loss
        return -loss

    def adapt_batch(self, batch: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Adapt on a single batch and return predictions.
        """
        batch = batch.to(self.device)
        targets = targets.to(self.device)

        # CMA‑ES
        es = cma.CMAEvolutionStrategy(self.cma_mean, self.cma_sigma,
                                      {'popsize': self.popsize})
        for _ in range(self.num_generations):
            solutions = es.ask()
            fitnesses = []
            for sol in solutions:
                fitnesses.append(self._fitness(sol, batch, targets))
            es.tell(solutions, fitnesses)

        # Select best prompt
        best_idx = np.argmax(es.result[1])  # es.result[1] = best fitness
        best_prompt = es.result[0][best_idx]
        prompt = torch.from_numpy(best_prompt).float().to(self.device)
        prompt = prompt.view(self.num_prompt, self.hidden_dim)
        self.prompted.prompt.copy_(prompt)

        # Update CMA mean & sigma for next batch
        self.cma_mean = es.result[0]
        self.cma_sigma = es.result[2]

        # Forward with best prompt
        logits, _ = self.prompted(batch, source_mean=self.source_means[-1],
                                  gamma=self.gamma)
        return logits

    def evaluate(self, loader: torch.utils.data.DataLoader):
        """
        Run adaptation on the entire test set.
        Returns overall accuracy.
        """
        total = 0
        correct = 0
        for imgs, targets in tqdm(loader, desc="FOA"):
            logits = self.adapt_batch(imgs, targets)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets.to(logits.device)).sum().item()
            total += imgs.size(0)
        return correct / total