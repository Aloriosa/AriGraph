import torch
import torch.nn as nn
import timm
import math
from utils import freeze_parameters

class Adapter(nn.Module):
    """
    Small 2‑layer adapter: down → ReLU → up.
    """
    def __init__(self, dim, r=8):
        super().__init__()
        self.down = nn.Linear(dim, r, bias=False)
        self.up = nn.Linear(r, dim, bias=False)

    def forward(self, x):
        return F.relu(self.down(x)) @ self.up.weight

class AutoEncoder(nn.Module):
    """
    Simple AE: Linear encoder → ReLU → Linear decoder.
    """
    def __init__(self, dim, r=8):
        super().__init__()
        self.enc = nn.Linear(dim, r, bias=False)
        self.dec = nn.Linear(r, dim, bias=False)

    def forward(self, x):
        z = F.relu(self.enc(x))
        return self.dec(z)

class SEMA(nn.Module):
    """
    SEMA model: Frozen ViT backbone + per‑layer adapters + routers + RDs.
    """
    def __init__(self, backbone_name="vit_base_patch16_224", expand_layers=[9,10,11], r=8):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        freeze_parameters(self.backbone)

        self.expand_layers = expand_layers
        self.r = r

        # For each expandable layer we keep:
        #   - list of adapters (ModuleList)
        #   - list of routers (ModuleList)
        #   - list of RDs (ModuleList)
        self.adapters = nn.ModuleDict()
        self.routers = nn.ModuleDict()
        self.rds = nn.ModuleDict()

        for l in expand_layers:
            self.adapters[str(l)] = nn.ModuleList()
            self.routers[str(l)] = nn.ModuleList()
            self.rds[str(l)] = nn.ModuleList()

        # Final classifier
        self.classifier = nn.Linear(768, 10)  # 10 CIFAR‑10 classes

    def forward(self, x):
        # Forward through ViT and intercept at each expandable layer
        # timm's vit has `blocks` list
        h = x
        for i, blk in enumerate(self.backbone.blocks):
            h = blk(h)
            if i in self.expand_layers:
                # gather adapters and router
                adapters = self.adapters[str(i)]
                routers = self.routers[str(i)]
                if len(adapters) == 0:
                    continue
                # compute router weights
                w = self._router_forward(routers, h)
                # weighted sum of adapters
                adapt_out = sum(wk * adapter(h) for wk, adapter in zip(w, adapters))
                h = h + adapt_out
        # Global average pooling
        h = self.backbone.norm(self.backbone.avgpool(h))
        logits = self.classifier(h)
        return logits

    def _router_forward(self, routers, h):
        """
        Compute softmax weights from the last router in the list.
        For simplicity we use the last router (the one that was added most recently).
        """
        router = routers[-1]  # last added router
        w = F.softmax(router(h), dim=-1)  # shape (B, #adapters)
        return w

    def add_adapter(self, layer_idx):
        """
        Add a new adapter, router and RD to the specified layer.
        """
        l = str(layer_idx)
        new_adapter = Adapter(dim=768, r=self.r)
        new_router = nn.Linear(768, 1, bias=False)  # one output for the new adapter
        new_rd = AutoEncoder(dim=768, r=self.r)
        self.adapters[l].append(new_adapter)
        self.routers[l].append(new_router)
        self.rds[l].append(new_rd)
        # Freeze old routers (keeping only the new one trainable)
        for i, r in enumerate(self.routers[l]):
            if i < len(self.routers[l]) - 1:
                r.requires_grad = False

    def get_rd_stats(self, layer_idx):
        """
        Return mean and std of reconstruction error for each RD in the layer.
        """
        l = str(layer_idx)
        stats = []
        for rd in self.rds[l]:
            stats.append({"mu": 0.0, "sigma": 1.0})  # placeholder
        return stats