# utils.py
import torch
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import math


def get_cifar10_loaders(batch_size=64):
    """Return CIFAR‑10 training and test data loaders."""
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_set = torch.utils.data.CIFAR10(root=".", train=True, download=True, transform=transform)
    test_set = torch.utils.data.CIFAR10(root=".", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def compute_source_stats(model, loader, device, n_samples=32):
    """
    Compute mean and std of CLS tokens on a small source set.
    """
    model.eval()
    mu = []
    sigma = []
    count = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            # Forward without prompt
            features = model.forward_features(images, torch.zeros(1, 0, model.patch_embed.out_dim, device=device))
            cls_token = features[:, 0, :]  # (B, dim)
            mu.append(cls_token.mean(dim=0))
            sigma.append(cls_token.std(dim=0))
            count += images.size(0)
            if count >= n_samples:
                break
    mu = torch.stack(mu, dim=0).mean(dim=0)
    sigma = torch.stack(sigma, dim=0).mean(dim=0)
    return {"mu": mu, "sigma": sigma}


def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log(probs + 1e-12)
    ent = -torch.sum(probs * logp, dim=-1)
    return ent.mean().item()


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (ECE)."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece_val = 0.0
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() == 0:
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].float().mean()
        ece_val += torch.abs(bin_confidence - bin_accuracy) * mask.sum() / probs.size(0)
    return ece_val.item()