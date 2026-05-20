import torch
import torch.nn as nn
import torchvision.models as models
from .mask_generator import MaskGenerator
from .utils import get_mask

class BaseModel(nn.Module):
    """
    Baseline watermarking model: shared mask (full/medium/narrow).
    """
    def __init__(self, backbone_name="resnet18", num_target_classes=10,
                 mask_size="full", mapping_tensor=None, device="cpu"):
        super().__init__()
        self.device = device
        self.mask_size = mask_size

        # Build backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone_out = 1000
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_out = 1000
        elif backbone_name == "vit_b32":
            self.backbone = models.vit_b_32(weights=models.ViT_B32_Weights.IMAGENET1K_V1)
            backbone_out = 1000
        else:
            raise ValueError("Unsupported backbone")

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Learnable pattern δ (shared across all images)
        self.delta = nn.Parameter(torch.zeros(1, 3, 224, 224, device=device))

        # Mapping from target to source classes
        if mapping_tensor is not None:
            self.mapping = mapping_tensor.to(device)
        else:
            self.mapping = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,224,224)  target images
        Returns logits of shape (B, 1000)
        """
        mask = get_mask(self.mask_size, x.device, x.shape)
        y = x + self.delta * mask
        logits = self.backbone(y)
        return logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mapping is None:
            raise ValueError("Mapping tensor must be set before training")
        selected_logits = logits[:, self.mapping]
        return nn.functional.cross_entropy(selected_logits, targets)

class SMMModel(BaseModel):
    """
    Full SMM pipeline: sample‑specific mask generator + shared pattern.
    Inherits the backbone, delta and output mapping from BaseModel.
    """
    def __init__(self, backbone_name="resnet18", num_target_classes=10,
                 mask_size="full", mapping_tensor=None, device="cpu"):
        super().__init__(backbone_name, num_target_classes, mask_size,
                         mapping_tensor, device)
        self.mask_gen = MaskGenerator().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.mask_gen(x)          # (B,3,224,224)
        y = x + self.delta * mask
        logits = self.backbone(y)
        return logits