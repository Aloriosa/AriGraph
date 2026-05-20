import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import timm
from tqdm import tqdm
from src.utils.data import get_dataloader
from src.utils.mapping import random_mapping
from src.utils.patch_interpolate import patchwise_interpolate
from src.models.mask_generator import MaskGenerator

# ---------- Utility functions ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_backbone(backbone_name, pretrained=True):
    if backbone_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif backbone_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif backbone_name == 'vitb32':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        in_features = model.head.in_features
        model.head = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone {backbone_name}")
    return model, in_features

def resize_images(images, target_size):
    """
    images: Tensor of shape (B, 3, H, W)
    target_size: int (e.g., 224 or 384)
    Returns resized images of shape (B, 3, target_size, target_size)
    """
    return nn.functional.interpolate(images, size=(target_size, target_size),
                                     mode='bilinear', align_corners=False)

# ---------- Training loop ----------
def train_one_epoch(model, mask_gen, delta, dataloader, mapping, criterion,
                    optimizer, device):
    model.train()
    mask_gen.train()
    delta.requires_grad_(True)
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(dataloader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # Resize to backbone input size
        imgs_resized = resize_images(imgs, model.input_size)

        # Generate mask
        mask = mask_gen(imgs_resized)
        # Pad delta to match size
        if delta.shape[-2:] != imgs_resized.shape[-2:]:
            delta_resized = nn.functional.interpolate(delta,
                                                      size=imgs_resized.shape[-2:],
                                                      mode='bicubic')
        else:
            delta_resized = delta

        # Reprogrammed image
        prog_imgs = imgs_resized + delta_resized * mask

        # Forward
        logits = model(prog_imgs)
        # Map target labels to source indices
        src_indices = mapping[labels]
        loss = criterion(logits, src_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == src_indices).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, mask_gen, delta, dataloader, mapping, device):
    model.eval()
    mask_gen.eval()
    delta.requires_grad_(False)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs_resized = resize_images(imgs, model.input_size)
            mask = mask_gen(imgs_resized)
            if delta.shape[-2:] != imgs_resized.shape[-2:]:
                delta_resized = nn.functional.interpolate(delta,
                                                          size=imgs_resized.shape[-2:],
                                                          mode='bicubic')
            else:
                delta_resized = delta
            prog_imgs = imgs_resized + delta_resized * mask
            logits = model(prog_imgs)
            src_indices = mapping[labels]
            preds = logits.argmax(dim=1)
            correct += (preds == src_indices).sum().item()
            total += imgs.size(0)
    return correct / total

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="SMM training demo")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone: resnet18, resnet50, vitb32")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.1,
                        help="Learning rate decay factor after half epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Data -----
    train_loader = get_dataloader(args.dataset, "train",
                                 batch_size=args.batch_size, seed=args.seed)
    test_loader = get_dataloader(args.dataset, "test",
                                batch_size=args.batch_size, seed=args.seed)

    # ----- Model -----
    backbone, in_features = get_backbone(args.backbone)
    backbone.input_size = 224 if args.backbone in ["resnet18","resnet50"] else 384
    backbone = backbone.to(device)

    # ----- Mask generator -----
    mask_gen = MaskGenerator(in_channels=3, out_channels=3, num_features=32).to(device)

    # ----- Shared pattern delta -----
    # Initialize as zeros of shape (1,3,H,W) where H/W = backbone input size
    delta = torch.zeros(1, 3, backbone.input_size, backbone.input_size,
                       device=device, requires_grad=True)

    # ----- Mapping -----
    # For ImageNet, 1000 classes
    num_source = 1000
    mapping = random_mapping(num_source, len(train_loader.dataset.classes),
                             seed=args.seed).to(device)

    # ----- Loss & Optimizer -----
    criterion = nn.CrossEntropyLoss()
    params = list(backbone.parameters()) + list(mask_gen.parameters()) + [delta]
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//2,
                                         gamma=args.lr_decay)

    # ----- Training -----
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            backbone, mask_gen, delta, train_loader,
            mapping, criterion, optimizer, device)
        val_acc = evaluate(backbone, mask_gen, delta, test_loader,
                           mapping, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
    print(f"\nBest validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()