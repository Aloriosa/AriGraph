import torch
import torch.nn as nn
import timm
import math

class Adapter(nn.Module):
    """Functional adapter: down‑proj → ReLU → up‑proj."""
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.relu(self.down(x)) @ self.up.weight.t()

class RepresentationDescriptor(nn.Module):
    """Simple linear auto‑encoder for reconstruction."""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.enc = nn.Linear(dim, hidden)
        self.dec = nn.Linear(hidden, dim)
        nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dec.weight, a=math.sqrt(5))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

class Router(nn.Module):
    """Expandable soft‑weighting router."""
    def __init__(self, dim, num_adapters=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, num_adapters))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: (B, D)
        w = F.softmax(x @ self.weight, dim=1)  # (B, K)
        return w

class SEMA(nn.Module):
    """
    SEMA core:
    - frozen ViT backbone
    - per‑layer list of adapters, RDs, and routers
    - self‑expansion logic
    """
    def __init__(self, backbone_name="vit_base_patch16_224", num_classes=10,
                 expansion_layers=[8,9,10], expansion_zthreshold=1.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze everything

        # Register hooks to capture intermediate features
        self._register_hooks()

        # Layer metadata
        self.num_layers = len(self.backbone.blocks)
        self.expansion_layers = expansion_layers
        self.expansion_zthreshold = expansion_zthreshold

        # Per‑layer modules
        self.adapters = nn.ModuleList([nn.ModuleList() for _ in range(self.num_layers)])
        self.rds = nn.ModuleList([nn.ModuleList() for _ in range(self.num_layers)])
        self.routers = nn.ModuleList([nn.ModuleList() for _ in range(self.num_layers)])

        # Classifier head
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

        # Training state
        self.feature_buffer = {}  # {layer_idx: list of features}
        self.rd_stats = {}       # {layer_idx: (mu, sigma)}

    def _register_hooks(self):
        self.feature_buffer = {}
        def hook(module, input, output, idx):
            # output: (B, N, D) for block outputs
            # we flatten tokens and take [0] CLS token
            cls = output[:, 0, :]
            self.feature_buffer[idx] = cls.detach()
        for idx, block in enumerate(self.backbone.blocks):
            block.register_forward_hook(lambda m, i, o, idx=idx: hook(m, i, o, idx))

    def forward(self, x):
        B = x.size(0)
        # Forward through backbone
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            # get CLS token
            cls = x[:, 0, :]

            # apply adapters if any
            if len(self.adapters[i]) > 0:
                # compute weighted sum
                w = self.routers[i][0](cls)  # (B, K)
                out = cls.clone()
                for k, adapter in enumerate(self.adapters[i]):
                    out += w[:, k:k+1] * adapter(cls)
                cls = out

        # classification
        logits = self.classifier(cls)
        return logits

    # ----------------------------- expansion logic -----------------------------
    def compute_rd_stats(self):
        """Compute mean & std of reconstruction errors for each frozen RD."""
        stats = {}
        for l in range(self.num_layers):
            if len(self.rds[l]) == 0:
                continue
            errors = []
            for rd in self.rds[l]:
                feats = self.feature_buffer.get(l, None)
                if feats is None:
                    continue
                recon = rd(feats)
                err = (feats - recon).pow(2).mean(dim=1).detach().cpu().numpy()
                errors.append(err)
            if errors:
                errors = np.vstack(errors).T  # (B, K)
                mu = errors.mean(axis=0)
                sigma = errors.std(axis=0)
                stats[l] = (mu, sigma)
        self.rd_stats = stats

    def should_expand_layer(self, l):
        """Return True if all RDs in layer l exceed z‑threshold."""
        if l not in self.rd_stats:
            return False
        mu, sigma = self.rd_stats[l]
        # compute errors for current batch (use buffered)
        feats = self.feature_buffer.get(l, None)
        if feats is None:
            return False
        recon_errs = []
        for rd in self.rds[l]:
            recon = rd(feats)
            err = (feats - recon).pow(2).mean(dim=1).detach().cpu().numpy()
            recon_errs.append(err)
        if not recon_errs:
            return False
        recon_errs = np.vstack(recon_errs).T  # (B, K)
        z = (recon_errs - mu) / (sigma + 1e-6)
        # if all adapters exceed threshold for *all* samples? we check mean
        mean_z = z.mean(axis=0)
        return (mean_z > self.expansion_zthreshold).all()

    def expand_layer(self, l):
        """Add a new adapter, RD and router column to layer l."""
        dim = self.backbone.embed_dim
        new_adapter = Adapter(dim)
        new_rd = RepresentationDescriptor(dim)
        if len(self.adapters[l]) == 0:
            # first adapter for this layer
            self.adapters[l].append(new_adapter)
            self.rds[l].append(new_rd)
            self.routers[l].append(Router(dim, num_adapters=1))
        else:
            # expand existing router
            old_router = self.routers[l][0]
            new_weight = nn.Parameter(torch.randn(dim, len(self.adapters[l]) + 1))
            with torch.no_grad():
                new_weight[:, :-1] = old_router.weight
            self.routers[l][0] = Router(dim, num_adapters=len(self.adapters[l]) + 1)
            self.routers[l][0].weight = new_weight
            # add adapter and RD
            self.adapters[l].append(new_adapter)
            self.rds[l].append(new_rd)

    # ----------------------------- training helpers -----------------------------
    def freeze_except_new(self, new_adapter_idxs):
        """Freeze all adapters except those in new_adapter_idxs."""
        for l in range(self.num_layers):
            for idx, adapter in enumerate(self.adapters[l]):
                for p in adapter.parameters():
                    p.requires_grad = (l, idx) in new_adapter_idxs
            for idx, rd in enumerate(self.rds[l]):
                for p in rd.parameters():
                    p.requires_grad = (l, idx) in new_adapter_idxs
            # routers
            for p in self.routers[l][0].parameters():
                p.requires_grad = False
            # enable router weights for new adapters only
            new_router = self.routers[l][0]
            if len(new_adapter_idxs) > 0:
                new_router.weight.requires_grad = True

    def get_new_adapter_params(self, new_adapter_idxs):
        params = []
        for l, idx in new_adapter_idxs:
            params += list(self.adapters[l][idx].parameters())
            params += list(self.rds[l][idx].parameters())
        return params