import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple
import timm
import cma
from tqdm.auto import tqdm


class PromptModule(nn.Module):
    """Learnable prompt to be prepended to the patch embeddings."""
    def __init__(self, prompt_len: int, embed_dim: int, device: torch.device):
        super().__init__()
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        # We keep the prompt as a plain tensor (no requires_grad),
        # because CMA‑ES handles the optimisation externally.
        self.register_buffer('prompt', torch.randn(prompt_len, embed_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Np+Npatches, D)
        """
        # Prepend the prompt to the input embeddings
        return torch.cat([self.prompt.unsqueeze(0).expand(x.size(0), -1, -1), x], dim=1)


class FOAAdapter:
    """
    Core FOA algorithm.
    """
    def __init__(
        self,
        model_name: str,
        prompt_len: int,
        batch_size: int,
        population_size: int,
        device: torch.device,
        seed: int = 42,
        lambda_discrepancy: float = 0.3,
        num_id_samples: int = 32,
        ema_alpha: float = 0.1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.population_size = population_size
        self.prompt_len = prompt_len
        self.lambda_discrepancy = lambda_discrepancy
        self.ema_alpha = ema_alpha
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load ViT model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1000)
        self.model.eval()
        self.model.to(device)

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Prompt module
        embed_dim = self.model.embed_dim
        self.prompt_module = PromptModule(prompt_len, embed_dim, device).to(device)

        # For activation shifting
        self.shift_step = 1.0
        self.shift_dir = torch.zeros(embed_dim, device=device)

        # Source statistics
        self.src_cls_mean = None
        self.src_cls_std = None
        self.num_id_samples = num_id_samples

        # CMA‑ES parameters
        self.cma_opt = None
        self.cma_context = None
        self.current_mean = None
        self.current_cov = None

    def _extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS token from the final transformer block."""
        # The timm ViT model expects shape (B, 3, H, W)
        # We will use the internal forward that returns the logits.
        # To get CLS features, we can access the internal embed layer.
        # For simplicity, we will run a forward pass and slice out the CLS token.
        # This requires us to modify the model temporarily.
        with torch.no_grad():
            # get patch embeddings
            patch_emb = self.model.patch_embed(x)  # (B, Npatch+1, D)
            # prepend prompt
            patch_emb = self.prompt_module(patch_emb)
            # transformer blocks
            for blk in self.model.blocks:
                patch_emb = blk(patch_emb)
            cls_token = patch_emb[:, 0]  # (B, D)
        return cls_token

    def _compute_fitness(self, cls_feats: torch.Tensor, logits: torch.Tensor) -> float:
        """
        cls_feats: (B, D)
        logits: (B, C)
        """
        # Entropy term
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * probs.log()).sum(dim=1).mean()

        # Discrepancy term
        mean_t = cls_feats.mean(dim=0)
        std_t = cls_feats.std(dim=0)

        discrepancy = torch.norm(mean_t - self.src_cls_mean) + torch.norm(std_t - self.src_cls_std)

        fitness = entropy + self.lambda_discrepancy * discrepancy.item()
        return fitness

    def _cma_sample(self) -> torch.Tensor:
        """Sample a prompt from CMA‑ES distribution."""
        vec = self.cma_opt.ask()  # list of floats
        # Convert to tensor shape (prompt_len, embed_dim)
        prompt_vec = torch.tensor(vec, dtype=torch.float32, device=self.device)
        prompt_vec = prompt_vec.view(self.prompt_len, self.model.embed_dim)
        return prompt_vec

    def _cma_update(self, fitnesses: List[float]):
        """Update CMA‑ES distribution."""
        self.cma_opt.tell(fitnesses)

    def _init_cma(self, dim: int):
        sigma = 0.5
        self.cma_opt = cma.CMAEvolutionStrategy(np.zeros(dim), sigma, {'popsize': self.population_size})

    def prepare_source_statistics(self, dataloader: torch.utils.data.DataLoader):
        """Compute CLS mean/std over a small set of ID samples."""
        cls_feats = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                # get CLS features
                patch_emb = self.model.patch_embed(imgs)
                patch_emb = self.prompt_module(patch_emb)
                for blk in self.model.blocks:
                    patch_emb = blk(patch_emb)
                cls = patch_emb[:, 0]
                cls_feats.append(cls.cpu())
                if len(cls_feats) >= self.num_id_samples:
                    break
        cls_feats = torch.cat(cls_feats, dim=0)
        self.src_cls_mean = cls_feats.mean(dim=0)
        self.src_cls_std = cls_feats.std(dim=0)

    def adapt_batch(self, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, C, H, W)
        labels: not used during adaptation
        Returns predictions (logits) after adaptation.
        """
        imgs = imgs.to(self.device)
        batch_size = imgs.size(0)

        # Prepare CMA‑ES
        dim = self.prompt_len * self.model.embed_dim
        self._init_cma(dim)

        best_fitness = float('inf')
        best_logits = None

        for _ in range(self.population_size):
            # Sample a prompt
            prompt_vec = self._cma_sample()
            # set prompt
            self.prompt_module.prompt.copy_(prompt_vec)

            # Forward pass
            patch_emb = self.model.patch_embed(imgs)
            patch_emb = self.prompt_module(patch_emb)
            for blk in self.model.blocks:
                patch_emb = blk(patch_emb)
            cls = patch_emb[:, 0]
            logits = self.model.head(cls)

            # Activation shifting
            cls_shift = cls + self.shift_step * self.shift_dir
            logits_shift = self.model.head(cls_shift)

            # Compute fitness
            fitness = self._compute_fitness(cls_shift, logits_shift)

            if fitness < best_fitness:
                best_fitness = fitness
                best_logits = logits_shift.detach().cpu()

            # CMA‑ES update
            self._cma_update([fitness])

        # Update shift direction using EMA
        mean_cls = cls_shift.mean(dim=0)
        self.shift_dir = self.ema_alpha * (self.src_cls_mean - mean_cls) + (1 - self.ema_alpha) * self.shift_dir

        return best_logits

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Return top‑1 accuracy and ECE over the dataset."""
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="FOA evaluation"):
                logits = self.adapt_batch(imgs, labels)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_probs.append(probs)
                all_labels.append(labels)

        acc = 100.0 * correct / total

        # Compute ECE
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        confidences, predictions = all_probs.max(dim=1)
        accuracies = predictions.eq(all_labels)

        bin_edges = torch.linspace(0, 1, 11)
        ece = 0.0
        for i in range(len(bin_edges)-1):
            mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
            if mask.sum() == 0:
                continue
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].float().mean()
            ece += torch.abs(bin_conf - bin_acc) * mask.sum().item() / total

        return acc, ece.item()