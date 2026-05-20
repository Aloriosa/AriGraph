import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

from .adapter import Adapter
from .rd import AutoEncoderDescriptor
from .router import ExpandingRouter

class SEMAViT(nn.Module):
    """
    Main SEMA model that wraps a frozen ViT backbone,
    adds modular adapters, routers and representation descriptors,
    and implements the self‑expansion logic.
    """
    def __init__(self,
                 backbone_name: str = "google/vit-base-patch16-224-in21k",
                 expansion_layers: int = 3,
                 adapter_bottleneck: int = 64,
                 rd_hidden: int = 128,
                 expansion_threshold: float = 1.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = ViTModel.from_pretrained(backbone_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.hidden_dim = self.backbone.config.hidden_size

        # Number of transformer layers in ViT
        self.num_layers = len(self.backbone.encoder.layer)
        self.expansion_layers = expansion_layers  # number of layers that can expand

        # For each layer we keep a list of adapters, routers and RDs
        self.adapters = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.rds = nn.ModuleList()          # list of list of RDs
        self.rds_stats = []                 # running mean & std for each RD

        # Initialize one adapter/descriptor/router per layer
        for i in range(self.num_layers):
            # Only allow expansion on the last `expansion_layers` layers
            if i >= self.num_layers - expansion_layers:
                num_adapters = 1
            else:
                num_adapters = 0  # no adapters yet
            # Adapter list
            adapter_list = nn.ModuleList()
            rd_list = nn.ModuleList()
            for _ in range(num_adapters):
                adapter_list.append(Adapter(self.hidden_dim, adapter_bottleneck))
                rd_list.append(AutoEncoderDescriptor(self.hidden_dim, rd_hidden))
            self.adapters.append(adapter_list)
            self.routers.append(ExpandingRouter(self.hidden_dim, num_adapters))
            self.rds.append(rd_list)
            self.rds_stats.append([])  # one dict per RD

        # Classification head (updated as new classes appear)
        self.cls_head = nn.Linear(self.hidden_dim, 10)  # placeholder, updated later
        self.cls_head.to(self.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT and adapters. Returns logits.
        """
        hidden = pixel_values  # [B, C, H, W]
        hidden = self.backbone.embeddings(hidden)  # [B, N+1, D]
        for i, layer in enumerate(self.backbone.encoder.layer):
            hidden = layer(hidden)  # [B, N+1, D]
            cls_token = hidden[:, 0, :]  # CLS token
            # Apply adapters
            if len(self.adapters[i]) > 0:
                weights = self.routers[i](cls_token)  # [B, K]
                adapter_outs = []
                for adapter in self.adapters[i]:
                    adapter_outs.append(adapter(cls_token))
                adapter_outs = torch.stack(adapter_outs, dim=1)  # [B, K, D]
                weighted = torch.sum(weights.unsqueeze(-1) * adapter_outs, dim=1)
                # add to all tokens
                hidden = hidden + weighted.unsqueeze(1).expand_as(hidden)
        # Pooling
        pooled = self.backbone.pooler(hidden)  # [B, D]
        logits = self.cls_head(pooled.to(self.device))
        return logits

    # ------------ Helper methods for self‑expansion ----------------

    def _extract_features(self, pixel_values: torch.Tensor):
        """
        Run a forward pass and return a list of CLS token features
        for each layer. Used for computing reconstruction errors.
        """
        hidden = pixel_values
        hidden = self.backbone.embeddings(hidden)
        features = []
        for layer in self.backbone.encoder.layer:
            hidden = layer(hidden)
            cls_token = hidden[:, 0, :]
            features.append(cls_token.detach())
        return features  # len = num_layers

    def compute_rds_stats(self, dataloader: torch.utils.data.DataLoader):
        """
        Compute running mean and std of reconstruction errors for each RD
        over the given dataloader. Results are stored in self.rds_stats.
        """
        device = self.device
        for layer_idx in range(self.num_layers):
            for rd_idx, rd in enumerate(self.rds[layer_idx]):
                errors = []
                rd.eval()
                for batch in dataloader:
                    imgs, _ = batch
                    imgs = imgs.to(device)
                    with torch.no_grad():
                        feats = self._extract_features(imgs)
                        feat = feats[layer_idx]
                        err = rd.reconstruction_error(feat)
                        errors.append(err.cpu())
                errors = torch.cat(errors, dim=0)  # all samples
                mean = errors.mean().item()
                std = errors.std().item() + 1e-6  # avoid division by zero
                self.rds_stats[layer_idx].append({'mean': mean, 'std': std})

    def decide_expansion(self, dataloader: torch.utils.data.DataLoader,
                         threshold: float = 1.0) -> list:
        """
        Decide whether to add a new adapter for each expandable layer.
        Returns a list of layer indices that will be expanded.
        """
        device = self.device
        expansions = []
        for layer_idx in range(self.num_layers - self.expansion_layers,
                               self.num_layers):
            # For each RD in this layer, compute z‑score of reconstruction error
            # on the current task
            z_scores = []
            for rd_idx, rd in enumerate(self.rds[layer_idx]):
                with torch.no_grad():
                    errors = []
                    rd.eval()
                    for batch in dataloader:
                        imgs, _ = batch
                        imgs = imgs.to(device)
                        feats = self._extract_features(imgs)
                        feat = feats[layer_idx]
                        err = rd.reconstruction_error(feat)
                        errors.append(err.cpu())
                    errors = torch.cat(errors, dim=0)
                    mean = errors.mean().item()
                    std = errors.std().item() + 1e-6
                # Use running stats from previous tasks
                if len(self.rds_stats[layer_idx]) > 0:
                    prev = self.rds_stats[layer_idx][-1]
                    z = (mean - prev['mean']) / prev['std']
                else:
                    # If this is the first task, always expand
                    z = float('inf')
                z_scores.append(z)
            # If *all* RDs report z > threshold, trigger expansion
            if all(z > threshold for z in z_scores):
                expansions.append(layer_idx)
        return expansions

    def add_adapter(self, layer_idx: int, adapter_bottleneck: int = 64,
                    rd_hidden: int = 128):
        """
        Add a new adapter/descriptor/expand router at the given layer.
        """
        # Adapter
        new_adapter = Adapter(self.hidden_dim, adapter_bottleneck).to(self.device)
        self.adapters[layer_idx].append(new_adapter)
        # RD
        new_rd = AutoEncoderDescriptor(self.hidden_dim, rd_hidden).to(self.device)
        self.rds[layer_idx].append(new_rd)
        # Router
        self.routers[layer_idx].expand()
        # Initialize stats entry
        self.rds_stats[layer_idx].append({'mean': 0.0, 'std': 1.0})

    def freeze_old_modules(self, new_layer_indices: list):
        """
        Freeze all adapters and routers *except* those newly added in this
        training phase. Also freeze the RDs that are not new.
        """
        # Freeze all adapters except the newest ones
        for i, adapter_list in enumerate(self.adapters):
            for j, adapter in enumerate(adapter_list):
                if i not in new_layer_indices or j != len(adapter_list) - 1:
                    for p in adapter.parameters():
                        p.requires_grad = False
        # Freeze routers except those with new column
        for i, router in enumerate(self.routers):
            if i not in new_layer_indices:
                for p in router.parameters():
                    p.requires_grad = False
        # Freeze RDs that are not new
        for i, rd_list in enumerate(self.rds):
            for j, rd in enumerate(rd_list):
                if i not in new_layer_indices or j != len(rd_list) - 1:
                    for p in rd.parameters():
                        p.requires_grad = False

    def train_on_task(self, train_loader, test_loader,
                      epochs_adapter: int = 5,
                      epochs_rd: int = 20,
                      lr_adapter: float = 5e-3,
                      lr_rd: float = 1e-2,
                      expansion_threshold: float = 1.0):
        """
        Main training routine for one task.
        """
        device = self.device
        # 1. Decide expansions
        expansions = self.decide_expansion(train_loader, threshold=expansion_threshold)
        for layer_idx in expansions:
            self.add_adapter(layer_idx)

        # 2. Freeze old modules
        self.freeze_old_modules(expansions)

        # 3. Optimizers
        # Collect parameters that require grad
        params = []
        for layer_idx in range(self.num_layers):
            for adapter in self.adapters[layer_idx]:
                params += list(adapter.parameters())
            for router in [self.routers[layer_idx]]:
                params += list(router.parameters())
            for rd in self.rds[layer_idx]:
                params += list(rd.parameters())
        params += list(self.cls_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr_adapter, weight_decay=0.0)

        # 4. Train adapters / routers
        self.train()
        for epoch in range(epochs_adapter):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = self(imgs)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
        # 5. Train RDs
        self.train()
        for epoch in range(epochs_rd):
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                feats = self._extract_features(imgs)
                for layer_idx in range(self.num_layers):
                    feat = feats[layer_idx]
                    for rd in self.rds[layer_idx]:
                        recon = rd(feat)
                        loss = F.mse_loss(recon, feat)
                        loss.backward()
                optimizer.step()
        # 6. Update RD stats
        self.compute_rds_stats(train_loader)

        # 7. Evaluation on test set
        acc = self.evaluate(test_loader)
        return acc

    def evaluate(self, loader):
        """
        Evaluate the model on the given loader.
        """
        device = self.device
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = self(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total