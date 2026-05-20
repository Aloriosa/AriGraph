# Configuration file for the evaluation script
# -------------------------------------------

# List of models to evaluate.
# Each entry must contain:
#   - name: identifier used in output tables
#   - module: function that returns a torch.nn.Module
#   - pretrained: whether to load pretrained weights
#   - zero_shot: whether this is a zero‑shot VLM (CLIP)
#   - device: device string

from torchvision import models
import torch
import clip

MODEL_REGISTRY = [
    {
        "name": "ResNet18",
        "module": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "pretrained": True,
        "zero_shot": False,
    },
    {
        "name": "ResNet50",
        "module": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "pretrained": True,
        "zero_shot": False,
    },
    {
        "name": "EfficientNetB0",
        "module": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "pretrained": True,
        "zero_shot": False,
    },
    {
        "name": "ConvNeXtTiny",
        "module": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "pretrained": True,
        "zero_shot": False,
    },
    # Zero‑shot VLMs – CLIP
    {
        "name": "CLIP-RN50",
        "module": lambda: clip.load("RN50", device="cpu")[0],  # model
        "pretrained": True,
        "zero_shot": True,
    },
    {
        "name": "CLIP-ViT-B32",
        "module": lambda: clip.load("ViT-B/32", device="cpu")[0],
        "pretrained": True,
        "zero_shot": True,
    },
]

# ImageNet validation set path (downloaded by torchvision)
IMAGENET_VAL_DIR = "imagenet_val"

# OOD datasets and their download URLs (via datasets library)
OOD_DATASETS = {
    "imagenet_v2": "imagenet_v2",
    "imagenet_sketch": "imagenet_sketch",
    "imagenet_r": "imagenet_r",
    "imagenet_a": "imagenet_a",
    "objectnet": "objectnet",
}

# Evaluation hyper‑parameters
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# K‑means latent hierarchy parameters
NUM_CLASS_LEVELS = 9  # 2^9 < 1000
CLUSTER_K = lambda level: 2 ** level