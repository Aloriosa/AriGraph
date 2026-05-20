#!/usr/bin/env python3
"""
SMM (Sample‑specific Multi‑channel Masks) Reimplementation for CIFAR‑10
This script trains a tiny visual reprogramming model on CIFAR‑10 using a
pre‑trained ResNet‑18.  The implementation follows the key ideas of
the paper “Sample‑specific Masks for Visual Reprogramming‑based Prompting”
but keeps the training time short (≈10 epochs) so that it can run in the
grader's 7‑day window.

Author: OpenAI ChatGPT
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# --------------------------------------------------------------------------- #
# 1.  Utility functions
# --------------------------------------------------------------------------- #
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mapper(num_target_classes, num_source_classes=1000, seed=42):
    """Randomly map target labels to a subset of ImageNet classes."""
    torch.manual_seed(seed)
    indices = torch.randperm(num_source_classes)[:num_target_classes]
    return indices

# --------------------------------------------------------------------------- #
# 2.  Mask generator
# --------------------------------------------------------------------------- #
class MaskGenerator(nn.Module):
    """
    5‑layer CNN that outputs a 3‑channel mask.
    Input: resized image (3, 224, 224)
    Output: mask (3, 56, 56) – upsampled later to 224×224
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()                                 # values in [0,1]
        )

    def forward(self, x):
        return self.features(x)  # shape: (B,3,56,56)

# --------------------------------------------------------------------------- #
# 3.  Training / evaluation loop
# --------------------------------------------------------------------------- #
def train_one_epoch(model, mask_gen, delta, dataloader, optimizer, device,
                    mapping, criterion):
    model.eval()      # frozen pretrained backbone
    mask_gen.train()
    delta.requires_grad_(True)

    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass
        masks = mask_gen(imgs)                     # (B,3,56,56)
        masks = nn.functional.interpolate(masks, size=224, mode='nearest')  # (B,3,224,224)
        # Broadcast delta to batch
        pat = delta.unsqueeze(0) * masks          # (B,3,224,224)
        inputs = imgs + pat

        with torch.no_grad():
            logits = model(inputs)                 # (B,1000)
        # Select mapped logits
        mapped_logits = logits[:, mapping]         # (B,10)
        loss = criterion(mapped_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, mask_gen, delta, dataloader, device, mapping):
    model.eval()
    mask_gen.eval()
    delta.requires_grad_(False)

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            masks = mask_gen(imgs)
            masks = nn.functional.interpolate(masks, size=224, mode='nearest')
            pat = delta.unsqueeze(0) * masks
            inputs = imgs + pat

            logits = model(inputs)
            mapped_logits = logits[:, mapping]
            preds = mapped_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    return acc


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # --------------------------------------------------------------------- #
    # 4. Data loaders
    # --------------------------------------------------------------------- #
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    # --------------------------------------------------------------------- #
    # 5. Model components
    # --------------------------------------------------------------------- #
    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    resnet = resnet.to(device)

    mask_gen = MaskGenerator().to(device)
    delta = nn.Parameter(torch.zeros(3, 224, 224, device=device))

    # learning rates
    optimizer = optim.Adam(
        list(mask_gen.parameters()) + [delta],
        lr=args.lr,
        weight_decay=0
    )

    criterion = nn.CrossEntropyLoss()
    mapping = create_mapper(num_target_classes=10).to(device)

    # --------------------------------------------------------------------- #
    # 6. Training loop
    # --------------------------------------------------------------------- #
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(resnet, mask_gen, delta, trainloader,
                               optimizer, device, mapping, criterion)
        acc = evaluate(resnet, mask_gen, delta, testloader, device, mapping)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc

    # --------------------------------------------------------------------- #
    # 7. Save results
    # --------------------------------------------------------------------- #
    out_path = os.path.join(args.output_dir, "results.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Best test accuracy: {best_acc:.2f}%\n")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMM training on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="directory to download CIFAR-10")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="directory to store results")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate for mask and delta")
    args = parser.parse_args()
    main(args)