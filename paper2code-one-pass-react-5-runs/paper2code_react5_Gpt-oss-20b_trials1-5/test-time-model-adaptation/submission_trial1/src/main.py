"""
Main script that demonstrates FOA (forward‑only adaptation) and a TENT baseline
on ImageNet‑C (severity 5).  The script downloads the required datasets
automatically and runs the evaluation in a single pass.
"""

import argparse
import os
import sys
import time
import torch
import torchvision
import torchvision.transforms as T
import timm
import numpy as np
from tqdm import tqdm
import torch.cuda  # noqa: F401

from foa import FOAModel, FOA

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def download_and_extract(url: str, dst_dir: str):
    """
    Download a zip file from `url` into `dst_dir` (folder). If the folder
    already exists, nothing is done.
    """
    if os.path.exists(dst_dir):
        return
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    tmp_zip = dst_dir + ".zip"
    print(f"Downloading {url} ...")
    os.system(f"wget -O {tmp_zip} {url}")
    print(f"Extracting {tmp_zip} ...")
    os.system(f"unzip -q {tmp_zip} -d {os.path.dirname(dst_dir)}")
    os.remove(tmp_zip)

def get_imagenet_c_loader(batch_size=64, num_workers=4, severity=5):
    """
    Returns a DataLoader for ImageNet‑C (severity level `severity`).
    The dataset is downloaded and extracted automatically.
    """
    data_root = "./data"
    data_dir = os.path.join(data_root, "imagenet_c")
    download_and_extract(
        "https://github.com/hendrycks/imagenet-c/releases/download/v0.1.0/imagenet_c.zip",
        data_dir,
    )
    # The extracted folder contains subfolders for each corruption type.
    # torchvision's ImageFolder will automatically pick up all images.
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=transform,
    )
    # Filter for the desired severity level
    # ImageNet‑C folders are named like 'gaussian_noise_5', so we keep only those
    indices = [
        i for i, (path, _) in enumerate(dataset.samples)
        if f"_{severity}" in os.path.basename(os.path.dirname(path))
    ]
    dataset.samples = [dataset.samples[i] for i in indices]
    dataset.targets = [dataset.targets[i] for i in indices]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader

def get_imagenet_val_loader(batch_size=64, num_workers=4):
    """
    Returns a DataLoader for ImageNet‑1K validation set.
    The dataset is downloaded and extracted automatically.
    """
    data_root = "./data"
    data_dir = os.path.join(data_root, "imagenet_val")
    download_and_extract(
        "https://image-net.org/archive/imagenet_2012_val.zip",
        data_dir,
    )
    val_dir = os.path.join(data_dir, "imagenet_2012_val")
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader

def compute_source_stats(model: FOAModel, loader, device, max_samples=32):
    """
    Computes mean and std of CLS tokens over a small source subset.
    """
    model.eval()
    cls_list = []
    collected = 0

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Collecting source CLS stats"):
            imgs = imgs.to(device)
            _, cls = model(imgs)
            cls_list.append(cls)
            collected += imgs.size(0)
            if collected >= max_samples:
                break

    all_cls = torch.cat(cls_list, dim=0)
    mu = all_cls.mean(dim=0)
    sig = all_cls.std(dim=0)
    return {"mu": mu, "sig": sig}

# --------------------------------------------------------------------------- #
# Static quantization helper
# --------------------------------------------------------------------------- #
def static_quantize(model, loader, device):
    """
    Apply static 8‑bit PTQ to `model` using calibration data from `loader`.
    The quantized model is returned (still on CPU).
    """
    model.eval()
    torch.backends.quantized.engine = "qnnpack"
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(model, inplace=True)

    # Calibration
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            model(imgs)
            break  # a single batch is enough for the small calibration set

    torch.quantization.convert(model, inplace=True)
    return model

# --------------------------------------------------------------------------- #
# TENT baseline
# --------------------------------------------------------------------------- #
class TentAdapter:
    """
    Test‑time entropy minimization (TENT) baseline.
    Only LayerNorm affine parameters are updated.
    """

    def __init__(self, model: torch.nn.Module, lr=0.001, momentum=0.9, device=None):
        self.model = model.to(device)
        self.device = device or torch.device("cpu")

        # Freeze all parameters except LayerNorm affine weights/biases
        for p in self.model.parameters():
            p.requires_grad = False
        self.tent_params = []
        for name, p in self.model.named_parameters():
            if "norm" in name and ("weight" in name or "bias" in name):
                p.requires_grad = True
                self.tent_params.append(p)

        self.optimizer = torch.optim.SGD(
            self.tent_params, lr=lr, momentum=momentum
        )
        self.model.train()

    def adapt_and_evaluate(self, loader):
        """
        For each batch: adapt (entropy minimization) and evaluate.
        Returns top‑1 and top‑5 accuracy over the entire dataset.
        """
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for imgs, labels in tqdm(loader, desc="TENT adaptation"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            # Forward
            logits, _ = self.model(imgs)
            probs = torch.softmax(logits, dim=-1)

            # Entropy loss
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()

            # Backward & step
            self.optimizer.zero_grad()
            entropy.backward()
            self.optimizer.step()

            # Evaluation
            preds = logits.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()
            top5 = logits.topk(5, dim=1).indices
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

        return 100.0 * correct_top1 / total, 100.0 * correct_top5 / total


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    # Default device (CPU for quantized model, GPU otherwise)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.quantize else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Preparing ImageNet-C loader...")
    test_loader = get_imagenet_c_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        severity=args.severity,
    )
    print("Preparing ImageNet‑1K validation loader...")
    val_loader = get_imagenet_val_loader(
        batch_size=args.source_samples,
        num_workers=args.num_workers,
    )

    # Build ViT‑Base (pretrained)
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=10,  # dummy, will be overridden by head
    )

    # Replace head to match ImageNet‑1K (1000 classes)
    backbone.head = nn.Linear(backbone.head.in_features, 1000).to(device)

    # Optional static 8‑bit PTQ via calibration
    if args.quantize:
        print("Applying static 8‑bit PTQ to ViT backbone...")
        backbone = static_quantize(backbone, val_loader, device)
        print("Static quantization completed.")
        device = torch.device("cpu")  # quantized models run on CPU

    # Wrap with FOAModel
    foamodel = FOAModel(backbone, num_prompts=args.num_prompts).to(device)

    # Compute source CLS statistics
    source_stats = compute_source_stats(
        foamodel, val_loader, device, max_samples=args.source_samples
    )

    # FOA evaluation
    foa = FOA(
        model=foamodel,
        source_stats=source_stats,
        lambda_discrep=args.lambda_discrep,
        population_size=args.population_size,
        gamma_shift=args.gamma_shift,
        device=device,
    )
    start = time.time()
    foa_acc_top1, foa_acc_top5 = foa.evaluate(test_loader)
    elapsed = time.time() - start
    print(f"\nFOA Top‑1 Accuracy on ImageNet‑C (severity {args.severity}): {foa_acc_top1:.2f}%")
    print(f"FOA Top‑5 Accuracy on ImageNet‑C (severity {args.severity}): {foa_acc_top5:.2f}%")
    print(f"Runtime: {elapsed:.1f} s")

    # TENT baseline
    if args.baseline_tent:
        # Re‑create fresh backbone to avoid contamination from FOA
        base_tent = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=10,
        )
        base_tent = base_tent.to(device)
        base_tent.head = nn.Linear(base_tent.head.in_features, 1000).to(device)

        if args.quantize:
            print("Applying static 8‑bit PTQ to TENT backbone...")
            base_tent = static_quantize(base_tent, val_loader, device)
            print("Static quantization completed.")
            device = torch.device("cpu")

        tent = TentAdapter(
            base_tent,
            lr=args.tent_lr,
            momentum=0.9,
            device=device,
        )
        start = time.time()
        tent_acc_top1, tent_acc_top5 = tent.adapt_and_evaluate(test_loader)
        elapsed = time.time() - start
        print(f"\nTENT Top‑1 Accuracy on ImageNet‑C (severity {args.severity}): {tent_acc_top1:.2f}%")
        print(f"TENT Top‑5 Accuracy on ImageNet‑C (severity {args.severity}): {tent_acc_top5:.2f}%")
        print(f"Runtime: {elapsed:.1f} s")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOA Reproduction on ImageNet‑C")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompt embeddings")
    parser.add_argument("--lambda-discrep", type=float, default=0.4, help="Weight for activation discrepancy")
    parser.add_argument("--population-size", type=int, default=28, help="CMA‑ES population size")
    parser.add_argument("--gamma-shift", type=float, default=1.0, help="Step size for activation shifting")
    parser.add_argument("--source-samples", type=int, default=32, help="Number of source samples for CLS stats")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--quantize", action="store_true", help="Apply static 8‑bit quantization")
    parser.add_argument("--baseline-tent", action="store_true", help="Run TENT baseline for comparison")
    parser.add_argument("--tent-lr", type=float, default=0.01, help="Learning rate for TENT")
    parser.add_argument("--severity", type=int, default=5, help="ImageNet‑C severity level (1‑5)")
    args = parser.parse_args()
    main(args)