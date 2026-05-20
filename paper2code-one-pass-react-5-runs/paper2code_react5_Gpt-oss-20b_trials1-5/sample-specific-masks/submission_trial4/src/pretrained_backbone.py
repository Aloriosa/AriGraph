import torch
import torchvision.models as models

class PretrainedBackbone(torch.nn.Module):
    """Convenience wrapper that exposes a frozen ImageNet‑pretrained backbone."""
    def __init__(self, backbone_name: str):
        super().__init__()
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone_name == "vit_b32":
            self.backbone = models.vit_b_32(weights=models.ViT_B32_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)