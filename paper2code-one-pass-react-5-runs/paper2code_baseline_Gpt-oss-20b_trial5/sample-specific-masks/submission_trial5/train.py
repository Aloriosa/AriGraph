import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import timm
from tqdm import tqdm
from dataset_utils import get_dataset
from mm_model import SMMWrapper

def get_backbone(name, pretrained=True):
    if name.lower() == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        input_size = 224
    elif name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        input_size = 224
    elif name.lower() == "vit_b32":
        model = timm.create_model("vit_base_patch32_384", pretrained=pretrained)
        input_size = 384
    else:
        raise ValueError(f"Unsupported backbone {name}")
    return model, input_size

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = logits.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        _, preds = logits.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / total

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    logging.basicConfig(filename=log_path,
                        filemode="w",
                        format="%(asctime)s %(message)s",
                        level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Data loaders
    train_loader = get_dataset(args.dataset, split="train")
    test_loader = get_dataset(args.dataset, split="test")

    # Backbone
    backbone, input_size = get_backbone(args.backbone)
    backbone = backbone.to(device)

    # SMM wrapper
    model = SMMWrapper(backbone, input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(model.mask_gen.parameters()) + [model.delta]
    optimizer = optim.Adam(params, lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        logging.info(f"Epoch {epoch}/{args.epochs} - "
                     f"Loss: {loss:.4f} - Train Acc: {train_acc:.4f} - "
                     f"Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best.pt"))

    print(f"Test Accuracy ({args.backbone}, {args.dataset}): {best_acc * 100:.2f}%")
    with open(log_path, "a") as f:
        f.write(f"Test Accuracy ({args.backbone}, {args.dataset}): {best_acc * 100:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMM Training")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["resnet18", "resnet50", "vit_b32"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "svhn", "gtsrb"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)