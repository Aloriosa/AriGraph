import torch
import torch.nn as nn
import timm
from typing import List

class AdapterModule(nn.Module):
    """Down–up linear adapter with a ReLU non‑linearity."""
    def __init__(self, dim: int = 768, r: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, r, bias=False)
        self.up   = nn.Linear(r, dim, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.relu(self.down(x)).matmul(self.up.weight.t())

class RepresentationDescriptor(nn.Module):
    """Simple auto‑encoder used as a novelty detector."""
    def __init__(self, dim: int = 768, hidden: int = 384):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden, bias=False)
        self.decoder = nn.Linear(hidden, dim, bias=False)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        z = self.relu(self.encoder(x))
        return self.decoder(z)

class LayerModule(nn.Module):
    """
    Container for the adapters, descriptors and router columns for a single ViT layer.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.adapters      = nn.ModuleList([AdapterModule(dim)])
        self.descriptors   = nn.ModuleList([RepresentationDescriptor(dim)])
        self.router_cols   = nn.ParameterList([nn.Parameter(torch.randn(dim))])  # one column
        self.mus           = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=False)])
        self.sigmas        = nn.ParameterList([nn.Parameter(torch.ones(1), requires_grad=False)])
        self.sample_counts = torch.tensor([0], dtype=torch.long)

    def add_adapter(self):
        """Add a new adapter/descriptor/router column."""
        self.adapters.append(AdapterModule(self.adapters[0].down.out_features))
        self.descriptors.append(RepresentationDescriptor(self.descriptors[0].encoder.out_features))
        new_col = nn.Parameter(torch.randn(self.router_cols[0].size(0)))
        self.router_cols.append(new_col)

        # Freeze all previous router columns
        for col in self.router_cols[:-1]:
            col.requires_grad = False
        new_col.requires_grad = True

        self.mus.append(nn.Parameter(torch.zeros(1), requires_grad=False))
        self.sigmas.append(nn.Parameter(torch.ones(1), requires_grad=False))
        self.sample_counts = torch.cat([self.sample_counts, torch.tensor([0], dtype=torch.long)])

    def forward(self, feat: torch.Tensor):
        """
        Compute weighted mixture of adapters for a batch of CLS token features.
        :param feat: (B, dim)
        :return: (B, dim)
        """
        # Adapter outputs: (B, K, dim)
        adapter_outs = torch.stack([adapter(feat) for adapter in self.adapters], dim=1)
        K = len(self.adapters)

        # Router weights: (dim, K)
        router = torch.stack(self.router_cols, dim=1)
        weights = torch.softmax(feat @ router, dim=-1).unsqueeze(1)  # (B,1,K)
        mix = torch.sum(weights * adapter_outs, dim=2)  # (B, dim)
        return mix

class SEMA(nn.Module):
    """
    Full SEMA model: frozen ViT backbone + per‑layer adapters/descriptors/routers + classifier.
    """
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained,
                                          num_classes=0, drop_path_rate=0.1)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.num_layers = len(self.backbone.blocks)
        self.layers: List[LayerModule] = nn.ModuleList(
            [LayerModule(self.backbone.embed_dim) for _ in range(self.num_layers)]
        )
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        """
        Forward pass through ViT with adapters and routers.
        :param x: (B, C, H, W)
        :return: logits, and optionally list of CLS features per layer
        """
        # Patch embedding
        x = self.backbone.patch_embed(x)          # (B, Np, D)
        x = x + self.backbone.pos_embed
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x)

        feats = []
        for l, block in enumerate(self.backbone.blocks):
            # Standard ViT block
            residual = x
            x = block.norm1(x)
            x = block.attn(x)
            x = residual + x
            residual = x
            x = block.norm2(x)
            x = block.mlp(x)
            x = residual + x

            # CLS token feature
            feat = x[:, 0, :]  # (B, D)
            feats.append(feat)

            # Adapter + router
            mix = self.layers[l](feat)  # (B, D)
            x[:, 0, :] = x[:, 0, :] + mix

        cls = x[:, 0, :]
        logits = self.classifier(cls)

        if return_feats:
            return logits, feats
        return logits

    def get_trainable_params(self):
        """Collect all parameters that should be optimized."""
        params = list(self.classifier.parameters())
        for layer in self.layers:
            params += list(layer.adapters.parameters())
            params += list(layer.descriptors.parameters())
            params += [col for col in layer.router_cols if col.requires_grad]
        return params