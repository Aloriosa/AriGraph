"""
Core SEMA components: AdapterModule, RepresentationDescriptor,
ExpandableRouter, and the full SEMA model.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

# ---------- Adapter Module ----------
class AdapterModule(nn.Module):
    """
    Lightweight adapter for a transformer block.
    Down‑projection -> ReLU -> Up‑projection.
    """
    def __init__(self, hidden_dim=768, adapter_dim=64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim, bias=False)
        self.up   = nn.Linear(adapter_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x):
        return torch.relu(self.down(x)).matmul(self.up.weight.t())

# ---------- Representation Descriptor ----------
class RepresentationDescriptor(nn.Module):
    """
    Small auto‑encoder used as a novelty detector.
    """
    def __init__(self, hidden_dim=768, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
        )
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.xavier_uniform_(self.decoder[0].weight)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def loss(self, x):
        recon = self.forward(x)
        return nn.functional.mse_loss(recon, x, reduction='mean')

# ---------- Expandable Router ----------
class ExpandableRouter(nn.Module):
    """
    Soft‑max weighting over adapters in a layer.
    """
    def __init__(self, hidden_dim=768, num_adapters=1):
        super().__init__()
        self.W_mix = nn.Parameter(torch.randn(hidden_dim, num_adapters))
        nn.init.xavier_uniform_(self.W_mix)

    def forward(self, x):
        """
        x: (B, H)
        returns weights: (B, K) where K = num_adapters
        """
        logits = x @ self.W_mix  # (B, K)
        return torch.softmax(logits, dim=-1)

    def expand(self, new_adapters=1):
        """
        Expand the router to accommodate new adapters.
        """
        old_W = self.W_mix.data
        new_W = torch.randn(old_W.size(0), old_W.size(1) + new_adapters, device=old_W.device)
        new_W[:, :old_W.size(1)] = old_W
        with torch.no_grad():
            self.W_mix.copy_(new_W)

# ---------- Full SEMA Model ----------
class SEMA(nn.Module):
    """
    ViT backbone (frozen) + modular adapters + routers.
    """
    def __init__(self, num_expandable_layers=3,
                 adapter_dim=64, hidden_dim=768, num_tasks=5):
        super().__init__()
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.hidden_dim = hidden_dim
        self.num_expandable_layers = num_expandable_layers
        self.adapter_dim = adapter_dim
        self.num_tasks = num_tasks

        # For each expandable layer we store list of adapters, descriptors, router
        self.adapters = nn.ModuleDict()
        self.descriptors = nn.ModuleDict()
        self.routers = nn.ModuleDict()

        # Initialize with one adapter per expandable layer (used for first task)
        for l in range(self.num_expandable_layers):
            layer_key = f'layer_{l}'
            self.adapters[layer_key] = nn.ModuleList([AdapterModule(hidden_dim, adapter_dim)])
            self.descriptors[layer_key] = nn.ModuleList([RepresentationDescriptor(hidden_dim, adapter_dim)])
            self.routers[layer_key] = ExpandableRouter(hidden_dim, 1)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 10)  # CIFAR‑10

    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, H, H)
        # Process from last to first (ViT outputs from first to last)
        # We only modify the last `num_expandable_layers` layers
        for idx in range(-self.num_expandable_layers, 0):
            layer_idx = idx + len(hidden_states)  # actual index
            layer_key = f'layer_{-idx-1}'
            h = hidden_states[layer_idx]  # (B, H, H)

            # For ViT, the token embeddings are of shape (B, num_patches+1, dim)
            # We aggregate across patches via mean
            h_patch = h.mean(dim=1)  # (B, H)

            # Compute adapter outputs and weighted sum
            adapter_outs = []
            for adapter in self.adapters[layer_key]:
                adapter_outs.append(adapter(h_patch))  # (B, H)
            if len(adapter_outs) == 0:
                adapt_sum = torch.zeros_like(h_patch)
            else:
                weights = self.routers[layer_key](h_patch)  # (B, K)
                stack = torch.stack(adapter_outs, dim=-1)   # (B, H, K)
                adapt_sum = torch.sum(stack * weights.unsqueeze(1), dim=-1)  # (B, H)

            h_patch = h_patch + adapt_sum
            hidden_states[layer_idx] = torch.cat([h_patch.unsqueeze(1), h[:, 1:, :]], dim=1)

        # Classification head uses the final [CLS] token of the last hidden state
        cls_token = hidden_states[-1][:, 0, :]  # (B, H)
        logits = self.classifier(cls_token)
        return logits

    # ---------- Methods for self-expansion ----------
    def scan_task(self, loader, device, threshold=1.0, epochs=1):
        """
        Scan the task data to decide where to expand.
        Returns a dict: layer_key -> bool (expand or not)
        """
        self.eval()
        expansions = {}
        with torch.no_grad():
            for _ in range(epochs):
                for imgs, _ in loader:
                    imgs = imgs.to(device)
                    outputs = self.backbone(imgs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    for idx in range(-self.num_expandable_layers, 0):
                        layer_idx = idx + len(hidden_states)
                        layer_key = f'layer_{-idx-1}'
                        h = hidden_states[layer_idx]
                        h_patch = h.mean(dim=1)  # (B, H)

                        # Compute reconstruction error for each descriptor
                        errors = []
                        for desc in self.descriptors[layer_key]:
                            err = desc.loss(h_patch).item()
                            errors.append(err)
                        # Compute z‑score assuming running mean/var stored in desc
                        # For simplicity, we use the mean of errors as a proxy
                        zscore = (np.mean(errors) - desc.running_mean) / (desc.running_std + 1e-6)
                        # If all z‑scores > threshold -> expand
                        if zscore > threshold:
                            expansions[layer_key] = True
                        else:
                            expansions[layer_key] = False
        return expansions

    def expand_layer(self, layer_key, device):
        """
        Add a new adapter, descriptor, and expand the router for the given layer.
        """
        # Append new adapter
        new_adapter = AdapterModule(self.hidden_dim, self.adapter_dim).to(device)
        new_descriptor = RepresentationDescriptor(self.hidden_dim, self.adapter_dim).to(device)
        self.adapters[layer_key].append(new_adapter)
        self.descriptors[layer_key].append(new_descriptor)

        # Expand router
        self.routers[layer_key].expand(new_adapters=1)

    def train_task(self, loader, device, lr=1e-4, epochs=5):
        """
        Train the model on the current task.
        Only newly added adapters/descriptors are trainable.
        """
        # Freeze all adapters/descriptors except the last ones
        for key in self.adapters:
            for i, adapter in enumerate(self.adapters[key]):
                for p in adapter.parameters():
                    p.requires_grad = (i == len(self.adapters[key]) - 1)
        for key in self.descriptors:
            for i, desc in enumerate(self.descriptors[key]):
                for p in desc.parameters():
                    p.requires_grad = (i == len(self.descriptors[key]) - 1)

        self.train()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = self(imgs)
                loss_cls = criterion(logits, labels)

                # Descriptor losses
                loss_rd = 0.0
                for key in self.descriptors:
                    for desc in self.descriptors[key]:
                        loss_rd += desc.loss(self.backbone(imgs).hidden_states[-1].mean(dim=1))
                loss = loss_cls + 0.1 * loss_rd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def save_results(self, results, path='results.json'):
        import json
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)