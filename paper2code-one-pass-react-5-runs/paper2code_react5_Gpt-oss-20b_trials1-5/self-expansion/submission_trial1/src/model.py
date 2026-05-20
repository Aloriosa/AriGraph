import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class FunctionalAdapter(nn.Module):
    """
    Lightweight adapter: down‑project → ReLU → up‑project.
    """
    def __init__(self, dim, hidden_dim=32):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return F.relu(self.down(x)).matmul(self.up.weight)


class RepresentationDescriptor(nn.Module):
    """
    Simple auto‑encoder used as a novelty detector.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def loss(self, x):
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction="mean")


class ExpandableRouter(nn.Module):
    """
    Soft‑max over adapters. For K adapters, a weight matrix of shape (dim, K) is learned.
    """
    def __init__(self, dim, num_adapters=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, num_adapters))

    def forward(self, x):
        # x: (B, dim)
        w = F.softmax(x @ self.weight, dim=1)  # (B, K)
        return w


class SEMA(nn.Module):
    """
    Minimal SEMA implementation that:
      - Uses a frozen ViT backbone.
      - Adds one adapter per task to the last transformer layer.
      - Keeps adapters and representation descriptors frozen after training.
      - Uses a router to mix adapter outputs.
    """
    def __init__(self, num_classes=100, device="cpu"):
        super().__init__()
        self.device = device
        # Pre‑trained ViT, frozen
        self.backbone = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Backbone feature dimension
        self.feat_dim = self.backbone.embed_dim  # 768 for ViT‑B/16

        # Storage for adapters, routers, RDs per layer (here only last layer)
        self.adapters = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.rds = nn.ModuleList()

        # Final classification head (dynamic output size)
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier.weight.data.uniform_(-0.01, 0.01)

        self.num_tasks_seen = 0

    def forward(self, x):
        """
        Forward pass through ViT, apply adapters and router at last layer,
        then classification.
        """
        # ViT forward_features returns (B, N+1, D) where first is CLS token
        features = self.backbone.forward_features(x)  # (B, N+1, D)
        cls_token = features[:, 0]  # (B, D)

        # Apply adapters (if any)
        if len(self.adapters) > 0:
            # Compute router weights
            router = self.routers[-1]  # only one router
            w = router(cls_token)  # (B, K)
            # Compute adapter outputs
            adapter_out = torch.stack([ad(cls_token) for ad in self.adapters], dim=1)  # (B, K, D)
            # Weighted sum
            mix = torch.sum(w.unsqueeze(-1) * adapter_out, dim=1)  # (B, D)
            cls_token = cls_token + mix

        logits = self.classifier(cls_token)
        return logits

    def add_adapter(self):
        """
        Add a new adapter, router, and RD for the next task.
        """
        adapter = FunctionalAdapter(self.feat_dim).to(self.device)
        router = ExpandableRouter(self.feat_dim, num_adapters=len(self.adapters)+1).to(self.device)
        rd = RepresentationDescriptor(self.feat_dim).to(self.device)

        self.adapters.append(adapter)
        self.routers.append(router)
        self.rds.append(rd)

    def freeze_last(self):
        """
        After training the current task, freeze adapters and RD for this task.
        """
        for p in self.adapters[-1].parameters():
            p.requires_grad = False
        for p in self.rds[-1].parameters():
            p.requires_grad = False
        for p in self.routers[-1].parameters():
            p.requires_grad = False

    def parameters_to_optimize(self):
        """
        Return parameters that should be optimized during training:
        - the classifier (for all classes)
        - the newly added adapter, router, and RD
        """
        params = list(self.classifier.parameters())
        params += list(self.adapters[-1].parameters())
        params += list(self.routers[-1].parameters())
        params += list(self.rds[-1].parameters())
        return params

    def train_on_task(self, dataloader, epochs=5, lr_cls=1e-3, lr_adapt=1e-3, lr_rd=1e-3):
        """
        Train the model on a single task.
        """
        self.train()
        optimizer = torch.optim.SGD(self.parameters_to_optimize(), lr=lr_adapt, momentum=0.9)

        # Separate optimizers for classifier and RD
        optimizer_cls = torch.optim.SGD(self.classifier.parameters(), lr=lr_cls, momentum=0.9)
        optimizer_rd = torch.optim.Adam(self.rds[-1].parameters(), lr=lr_rd)

        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                optimizer_cls.zero_grad()
                optimizer_rd.zero_grad()

                logits = self.forward(x)
                loss = ce_loss(logits, y)
                # RD loss on features
                with torch.no_grad():
                    features = self.backbone.forward_features(x)
                    cls_token = features[:, 0]
                loss += self.rds[-1].loss(cls_token)

                loss.backward()
                optimizer.step()
                optimizer_cls.step()
                optimizer_rd.step()

                epoch_loss += loss.item() * x.size(0)

            epoch_loss /= len(dataloader.dataset)
            # print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss:.4f}")

        # After training, freeze the newly added modules
        self.freeze_last()