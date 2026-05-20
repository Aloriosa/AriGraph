import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from mm_mask import MaskGenerator
from utils import set_seed
import timm

# --------------------- Global Configuration --------------------- #
EPOCHS = 5
BATCH_SIZE = 128
LR = 0.01
WEIGHT_DECAY = 0.0
MILESTONES = [100, 145]
GAMMA = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# --------------------- Dataset Registry --------------------- #
# Each entry: (dataset_class, kwargs_for_train, kwargs_for_test, num_classes)
DATASETS = {
    "CIFAR10": (
        torchvision.datasets.CIFAR10,
        {"train": True, "download": True},
        {"train": False, "download": True},
        10
    ),
    "CIFAR100": (
        torchvision.datasets.CIFAR100,
        {"train": True, "download": True},
        {"train": False, "download": True},
        100
    ),
    "SVHN": (
        torchvision.datasets.SVHN,
        {"split": "train", "download": True, "transform": None},
        {"split": "test", "download": True, "transform": None},
        10
    ),
    "GTSRB": (
        torchvision.datasets.GTSRB,
        {"split": "train", "download": True},
        {"split": "test", "download": True},
        43
    ),
    "Flowers102": (
        torchvision.datasets.Flowers102,
        {"train": True, "download": True},
        {"train": False, "download": True},
        102
    ),
    "DTD": (
        torchvision.datasets.DTD,
        {"split": "train", "download": True},
        {"split": "test", "download": True},
        47
    ),
    "UCF101": (
        torchvision.datasets.UCF101,
        {"train": True, "download": True},
        {"train": False, "download": True},
        101
    ),
    "Food101": (
        torchvision.datasets.Food101,
        {"train": True, "download": True},
        {"train": False, "download": True},
        101
    ),
    "OxfordIIITPet": (
        torchvision.datasets.OxfordIIITPet,
        {"split": "train", "download": True},
        {"split": "val", "download": True},
        37
    ),
    "SUN397": (
        torchvision.datasets.SUN397,
        {"split": "train", "download": True},
        {"split": "test", "download": True},
        397
    ),
}

BACKBONES = {
    "resnet18": torchvision.models.resnet18,
    "vitb32": timm.create_model,
}

# --------------------- Helpers --------------------- #
def get_fixed_mask(mask_type, img_size=IMG_SIZE):
    """
    Return a fixed mask tensor of shape [3, img_size, img_size].
    mask_type: 'pad', 'full', 'medium', 'narrow'
    """
    mask = torch.zeros(3, img_size, img_size, dtype=torch.float32)
    if mask_type == 'pad':
        # No noise added
        return mask
    if mask_type == 'full':
        mask.fill_(1.0)
        return mask
    # For watermarking baselines: central square
    if mask_type == 'medium':
        size = 56
    elif mask_type == 'narrow':
        size = 28
    else:
        raise ValueError(f"Unknown mask_type {mask_type}")
    start = (img_size - size) // 2
    end = start + size
    mask[:, start:end, start:end] = 1.0
    return mask

def compute_ilm_mapping(model, mask_func, delta, train_loader, src_classes, tgt_classes, device):
    """
    Compute Iterative Label Mapping (ILM) for one epoch.
    mask_func: Callable that given a batch of images returns mask [B,3,H,W]
    """
    counts = torch.zeros((src_classes, tgt_classes), dtype=torch.int64, device=device)
    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = mask_func(images)  # [B,3,H,W]
            inp = images + delta * masks
            logits = model(inp)
            preds = logits.argmax(dim=1)  # [B]
            for i in range(len(labels)):
                s = preds[i].item()
                t = labels[i].item()
                counts[s, t] += 1
    mapping = torch.empty(tgt_classes, dtype=torch.long, device=device)
    used_src = set()
    for t in range(tgt_classes):
        temp_counts = counts[:, t].clone()
        for s_used in used_src:
            temp_counts[s_used] = -1
        s = torch.argmax(temp_counts).item()
        mapping[t] = s
        used_src.add(s)
    return mapping

def train_and_evaluate(dataset_name, cls, train_kwargs, test_kwargs, num_classes,
                       backbone_name, baseline_type=None, mask_generator=None):
    """
    baseline_type: None -> train SMM (mask_generator used)
                   'pad', 'full', 'medium', 'narrow' -> use fixed mask
    """
    print(f"\n=== Training {backbone_name} on {dataset_name} ({num_classes} classes) ===")
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    # Update transforms for datasets that require it
    for kw in [train_kwargs, test_kwargs]:
        kw["transform"] = transform

    train_dataset = cls(root='./data', **train_kwargs)
    test_dataset  = cls(root='./data', **test_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Backbone
    if backbone_name == "resnet18":
        backbone = torchvision.models.resnet18(pretrained=True)
    elif backbone_name == "vitb32":
        # timm expects default image size 384; we resize to 224 so we use a custom config
        backbone = timm.create_model('vit_base_patch32_384', pretrained=True, num_classes=1000)
    else:
        raise ValueError(f"Unknown backbone {backbone_name}")
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Shared noise pattern
    delta = nn.Parameter(torch.randn(3, IMG_SIZE, IMG_SIZE, device=DEVICE))
    if baseline_type is None:
        # Train mask generator
        mask_gen = mask_generator().to(DEVICE)
        mask_gen.train()
        params = list(mask_gen.parameters()) + [delta]
    else:
        # No mask generator; fixed mask
        mask_gen = None
        params = [delta]
    optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()

    # Initial random injective mapping
    init_mapping = torch.tensor(
        torch.randperm(1000)[:num_classes], dtype=torch.long, device=DEVICE
    )
    mapping = init_mapping

    for epoch in range(EPOCHS):
        mask_gen.train() if mask_gen is not None else None
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if mask_gen is not None:
                masks = mask_gen(images)  # [B,3,28,28]
            else:
                masks = get_fixed_mask(baseline_type, IMG_SIZE).unsqueeze(0).to(DEVICE)
                masks = masks.expand(images.size(0), -1, -1, -1)
            masks = F.interpolate(masks, size=(IMG_SIZE, IMG_SIZE), mode='nearest')
            inp = images + delta * masks

            logits = backbone(inp)
            logits_mapped = logits[:, mapping]
            loss = criterion(logits_mapped, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")
        scheduler.step()

        # Update ILM mapping after each epoch
        if mask_gen is not None:
            mask_func = lambda x: mask_gen(x)
        else:
            fixed_mask = get_fixed_mask(baseline_type, IMG_SIZE).unsqueeze(0).to(DEVICE)
            mask_func = lambda x: fixed_mask.expand(x.size(0), -1, -1, -1)
        mapping = compute_ilm_mapping(
            backbone, mask_func, delta, train_loader,
            src_classes=1000, tgt_classes=num_classes, device=DEVICE
        )

    # Evaluation
    backbone.eval()
    if mask_gen is not None:
        mask_gen.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            if mask_gen is not None:
                masks = mask_gen(images)
            else:
                masks = get_fixed_mask(baseline_type, IMG_SIZE).unsqueeze(0).to(DEVICE)
                masks = masks.expand(images.size(0), -1, -1, -1)
            masks = F.interpolate(masks, size=(IMG_SIZE, IMG_SIZE), mode='nearest')
            inp = images + delta * masks
            logits = backbone(inp)
            logits_mapped = logits[:, mapping]
            preds = logits_mapped.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy on {dataset_name} ({backbone_name}): {acc:.2f}%")
    return acc

def main():
    set_seed(SEED)
    results = []

    for dataset_name, (cls, train_kwargs, test_kwargs, num_classes) in DATASETS.items():
        # SMM
        acc_smm = train_and_evaluate(
            dataset_name, cls, train_kwargs, test_kwargs, num_classes,
            backbone_name="resnet18",
            baseline_type=None,
            mask_generator=MaskGenerator
        )
        results.append(f"{dataset_name} - SMM: {acc_smm:.2f}%")

        # Baselines
        for baseline in ['pad', 'full', 'medium', 'narrow']:
            acc_base = train_and_evaluate(
                dataset_name, cls, train_kwargs, test_kwargs, num_classes,
                backbone_name="resnet18",
                baseline_type=baseline,
                mask_generator=None
            )
            results.append(f"{dataset_name} - {baseline.capitalize()}: {acc_base:.2f}%")

    # Write results
    with open('results.txt', 'w') as f:
        for line in results:
            f.write(line + '\n')
    print("\nAll results written to results.txt")

if __name__ == "__main__":
    main()