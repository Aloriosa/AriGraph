import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import defaultdict


class Adapter(nn.Module):
    """Simple two‑layer adapter."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return F.relu(self.down(x)) @ self.up.weight.t()


class RepresentationDescriptor(nn.Module):
    """Auto‑encoder used as a novelty detector."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon

    def loss(self, x):
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction='mean')


class SEMA(nn.Module):
    """
    Self‑Expansion of pre‑trained models with Modularized Adapters.
    """
    def __init__(self, base_model, dim, hidden_dim, expand_layers, num_classes):
        super().__init__()
        self.base = base_model
        self.base.eval()  # freeze backbone
        for p in self.base.parameters():
            p.requires_grad = False

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expand_layers = expand_layers
        self.num_layers = len(self.base.blocks)

        # Per‑layer modules
        self.adapters = defaultdict(list)   # {layer_idx: [Adapter, ...]}
        self.rds = defaultdict(list)       # {layer_idx: [RepresentationDescriptor, ...]}
        self.routers = dict()              # {layer_idx: nn.Linear}

        # Initialise with one adapter per expandable layer
        for l in self.expand_layers:
            self._add_adapter(l)

        # Classification head that grows with tasks
        self.head = nn.Linear(dim, num_classes, bias=True)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Hooks to inject adapters into the MLP of each block
        self.hooks = []
        for l in self.expand_layers:
            block = self.base.blocks[l]
            hook = block.mlp.register_forward_hook(
                self._make_mlp_hook(l))
            self.hooks.append(hook)

    def _add_adapter(self, layer_idx):
        """Add a new adapter, RD and expand the router."""
        self.adapters[layer_idx].append(
            Adapter(self.dim, self.hidden_dim))
        self.rds[layer_idx].append(
            RepresentationDescriptor(self.dim, self.hidden_dim))
        # Expand router
        old_router = self.routers.get(layer_idx)
        if old_router is None:
            # first adapter
            router = nn.Linear(self.dim, 1, bias=False)
            nn.init.normal_(router.weight, std=0.02)
        else:
            old_weight = old_router.weight.data
            new_weight = torch.cat([old_weight, torch.randn(self.dim, 1) * 0.02], dim=1)
            router = nn.Linear(self.dim, old_router.out_features + 1, bias=False)
            router.weight.data = new_weight
        self.routers[layer_idx] = router

    def _make_mlp_hook(self, layer_idx):
        """Return a forward hook that injects the adapter mixture."""
        def hook(module, input, output):
            x = output  # shape: (B, dim)
            routers = self.routers[layer_idx]
            adapters = self.adapters[layer_idx]
            # compute mix
            w = torch.softmax(routers(x), dim=-1)  # (B, K)
            mix = torch.zeros_like(x)
            for k, adapter in enumerate(adapters):
                mix += w[:, k].unsqueeze(-1) * adapter(x)
            return x + mix
        return hook

    def forward(self, x):
        # Pass through backbone
        features = self.base.patch_embed(x)
        features = self.base.pos_drop(features)
        for i, blk in enumerate(self.base.blocks):
            features = blk(features)
        # features: (B, N, dim)
        cls_token = features[:, 0]
        logits = self.head(cls_token)
        return logits, cls_token  # also return cls token for RD usage

    def get_rd_losses(self, cls_token):
        """Compute reconstruction loss for each RD."""
        losses = []
        for l in self.expand_layers:
            for rd in self.rds[l]:
                losses.append(rd.loss(cls_token))
        return sum(losses)

    def expand_if_needed(self, cls_token, mu, sigma, threshold):
        """
        Decide whether to expand at each layer based on z‑scores.
        mu, sigma are tensors of shape (layer, rd) containing running stats.
        """
        expanded = False
        for l in self.expand_layers:
            # compute z‑score for each RD
            z = (mu[l] - mu[l].mean()) / (sigma[l] + 1e-6)
            if torch.all(z > threshold):
                self._add_adapter(l)
                expanded = True
        return expanded

    def add_class(self, num_new):
        """Expand the classification head to accommodate new classes."""
        old_weight = self.head.weight.data
        old_bias = self.head.bias.data
        new_weight = torch.randn(num_new, self.dim) * 0.02
        new_bias = torch.zeros(num_new)
        self.head = nn.Linear(self.dim, old_weight.size(0) + num_new, bias=True)
        self.head.weight.data[:old_weight.size(0)] = old_weight
        self.head.bias.data[:old_bias.size(0)] = old_bias
        self.head.weight.data[old_weight.size(0):] = new_weight
        self.head.bias.data[old_weight.size(0):] = new_bias