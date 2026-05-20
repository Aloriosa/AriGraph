"""
Compute LPIPS (learned perceptual image patch similarity) and FID between
generated samples and the target domain.
"""
import glob
import os
from pathlib import Path

import lpips
import numpy as np
import scipy.linalg
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# --------------------------------------------------------------------------- #
# Helper dataset
# --------------------------------------------------------------------------- #
class ImageFolderDataset(Dataset):
    """Simple folder‑based image loader that returns tensors in [0,1]."""

    def __init__(self, root, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# --------------------------------------------------------------------------- #
# Metric implementations
# --------------------------------------------------------------------------- #
def compute_lpips(gen_dir, target_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    gen_loader = DataLoader(
        ImageFolderDataset(gen_dir, transform=transforms.ToTensor()),
        batch_size=32,
        shuffle=False,
    )
    target_loader = DataLoader(
        ImageFolderDataset(target_dir, transform=transforms.ToTensor()),
        batch_size=32,
        shuffle=False,
    )

    lpips_scores = []
    with torch.no_grad():
        for g_batch, t_batch in zip(gen_loader, target_loader):
            g_batch = g_batch.to(device)
            t_batch = t_batch.to(device)
            # Ensure same shape
            g_batch = g_batch.clamp(0, 1)
            t_batch = t_batch.clamp(0, 1)
            d = loss_fn(g_batch, t_batch)
            lpips_scores.extend(d.cpu().numpy().flatten().tolist())

    return np.mean(lpips_scores)


def compute_fid(real_dir, gen_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = models.inception_v3(pretrained=True, aux_logits=False).to(device)
    inception.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def get_activations(data_dir):
        dataset = ImageFolderDataset(data_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        act = []
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                feat = inception(x)
                act.append(feat.cpu().numpy())
        return np.concatenate(act, axis=0)

    real_act = get_activations(real_dir)
    gen_act = get_activations(gen_dir)

    mu_real = np.mean(real_act, axis=0)
    mu_gen = np.mean(gen_act, axis=0)
    sigma_real = np.cov(real_act, rowvar=False)
    sigma_gen = np.cov(gen_act, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = scipy.linalg.sqrtm(
        sigma_real @ sigma_gen, disp=False
    )  # may be complex
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


def compute_metrics(gen_dir, target_dir, args):
    print("Computing LPIPS...")
    lpips_score = compute_lpips(gen_dir, target_dir)
    print(f"Average LPIPS (generated vs target): {lpips_score:.4f}")

    print("Computing FID...")
    fid_score = compute_fid(target_dir, gen_dir)
    print(f"FID (generated vs target): {fid_score:.2f}")

    with open("metrics.txt", "w") as f:
        f.write(f"LPIPS: {lpips_score:.4f}\n")
        f.write(f"FID: {fid_score:.2f}\n")