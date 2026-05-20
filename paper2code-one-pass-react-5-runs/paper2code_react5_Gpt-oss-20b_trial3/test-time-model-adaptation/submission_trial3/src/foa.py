import torch
import torch.nn.functional as F
import pycma
import numpy as np
from tqdm import tqdm


class FOA:
    """
    Minimal FOA implementation for a ViT‑like model.
    The model is assumed to expose the following attributes:
        - patch_embed: nn.Module that maps (B,3,H,W) -> (B,N,dim)
        - cls_token: nn.Parameter of shape (1,1,dim)
        - pos_embed: nn.Parameter of shape (1,N+1,dim)
        - blocks: nn.ModuleList of transformer blocks
        - head: nn.Linear mapping dim -> num_classes
    """

    def __init__(
        self,
        model,
        device,
        num_prompt_tokens=3,
        prompt_dim=None,
        popsize=28,
        lambda_activation=0.4,
    ):
        self.model = model
        self.device = device
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_dim = prompt_dim or model.head.in_features
        self.popsize = popsize
        self.lambda_activation = lambda_activation
        self.cma = None
        self.source_mean = None
        self.source_std = None

    # ------------------------------------------------------------------
    # 1. Compute source statistics (CLS token mean / std) on clean data
    # ------------------------------------------------------------------
    def compute_source_stats(self, dataloader):
        means = []
        stds = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in tqdm(dataloader, desc="Computing source stats"):
                imgs = imgs.to(self.device)
                cls = self._forward_cls(imgs)
                means.append(cls.mean(0))
                stds.append(cls.std(0))
        self.source_mean = torch.stack(means).mean(0)
        self.source_std = torch.stack(stds).mean(0)

    # ------------------------------------------------------------------
    # 2. Forward pass to obtain CLS token (before the head)
    # ------------------------------------------------------------------
    def _forward_cls(self, imgs):
        # imgs: (B,3,H,W)
        patches = self.model.patch_embed(imgs)  # (B,N,dim)
        B, N, dim = patches.shape

        # prepend cls token
        cls_token = self.model.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, patches], dim=1)  # (B,N+1,dim)

        # add position embedding
        tokens = tokens + self.model.pos_embed

        # transformer blocks
        for blk in self.model.blocks:
            tokens = blk(tokens)

        # CLS token is at position 0
        return tokens[:, 0, :]  # (B,dim)

    # ------------------------------------------------------------------
    # 3. Fitness function: entropy + activation discrepancy
    # ------------------------------------------------------------------
    def fitness(self, prompt, imgs):
        """
        prompt: (num_prompt_tokens, prompt_dim)
        imgs: (B,3,H,W)
        Returns a scalar fitness value (lower is better).
        """
        B = imgs.size(0)
        prompt_t = prompt.expand(B, -1, -1).to(self.device)  # (B,Np,dim)
        patches = self.model.patch_embed(imgs)  # (B,N,dim)
        tokens = torch.cat([prompt_t, patches], dim=1)
        cls_token = self.model.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.model.pos_embed

        for blk in self.model.blocks:
            tokens = blk(tokens)

        cls = tokens[:, 0, :]  # (B,dim)

        # Activation shifting towards source mean
        shift = self.source_mean - cls.mean(0)
        cls = cls + self.lambda_activation * shift

        logits = self.model.head(cls)
        probs = F.softmax(logits, dim=1)

        # entropy
        entropy = -(probs * probs.log()).sum(1).mean()

        # activation discrepancy
        mean_cls = cls.mean(0)
        std_cls = cls.std(0)
        disc = torch.norm(mean_cls - self.source_mean) + torch.norm(std_cls - self.source_std)

        return entropy + self.lambda_activation * disc

    # ------------------------------------------------------------------
    # 4. Adaptation loop (online over test batches)
    # ------------------------------------------------------------------
    def adapt(self, dataloader):
        dim = self.num_prompt_tokens * self.prompt_dim
        self.cma = pycma.CMAEvolutionStrategy([0.0] * dim, 0.3, {"popsize": self.popsize})

        for imgs, _ in tqdm(dataloader, desc="Adapting"):
            imgs = imgs.to(self.device)

            # CMA sampling
            solutions = self.cma.ask()
            fitness_vals = []
            for sol in solutions:
                prompt = torch.tensor(sol, dtype=torch.float32).view(
                    self.num_prompt_tokens, self.prompt_dim
                )
                val = self.fitness(prompt, imgs)
                fitness_vals.append(val.item())

            self.cma.tell(solutions, fitness_vals)

            # Keep the best prompt for this batch
            best_idx = np.argmin(fitness_vals)
            best_prompt = torch.tensor(solutions[best_idx], dtype=torch.float32).view(
                self.num_prompt_tokens, self.prompt_dim
            )
            # We don't store the prompt – the model is never updated.
            # In a real setting you would keep the best prompt for subsequent batches.

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, dataloader, prompt=None):
        """
        Evaluate the model on the given dataloader.
        If prompt is None, use zero prompt (i.e. no prompt).
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="Evaluating"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                if prompt is None:
                    prompt_t = torch.zeros(
                        imgs.size(0),
                        self.num_prompt_tokens,
                        self.prompt_dim,
                        device=self.device,
                    )
                else:
                    prompt_t = prompt.expand(imgs.size(0), -1, -1)

                patches = self.model.patch_embed(imgs)
                tokens = torch.cat([prompt_t, patches], dim=1)
                cls_token = self.model.cls_token.expand(imgs.size(0), -1, -1)
                tokens = torch.cat([cls_token, tokens], dim=1)
                tokens = tokens + self.model.pos_embed

                for blk in self.model.blocks:
                    tokens = blk(tokens)

                cls = tokens[:, 0, :]
                logits = self.model.head(cls)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total