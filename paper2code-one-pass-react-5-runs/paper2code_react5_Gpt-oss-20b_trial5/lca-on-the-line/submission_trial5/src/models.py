"""
Convenience wrappers to load the models used in the paper
(ResNet18, ResNet50, CLIP RN50, CLIP ViT‑B‑32).
"""

import torch
import torchvision.models as tvm
import open_clip
import os
from pathlib import Path
from typing import Dict, Any


def load_resnet18(device: torch.device):
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    model.to(device)
    # Build mapping from class index to synset ID
    model.class_to_synset = _build_imagenet_mapping()
    return model


def load_resnet50(device: torch.device):
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    model.class_to_synset = _build_imagenet_mapping()
    return model


def load_clip_rn50(device: torch.device):
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "RN50", pretrained="openai", device=device
    )
    # The model outputs logits for 1000 ImageNet classes
    model.eval()
    return model, preprocess


def load_clip_vitb32(device: torch.device):
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    return model, preprocess


def _build_imagenet_mapping() -> Dict[int, str]:
    """
    Build a dictionary mapping class index (0‑999) to ImageNet synset ID.
    The mapping is derived from the file src/imagenet_classes.txt.
    """
    mapping = {}
    with open("src/imagenet_classes.txt") as f:
        for idx, line in enumerate(f):
            syn_id, _ = line.strip().split(maxsplit=1)
            mapping[idx] = syn_id
    return mapping