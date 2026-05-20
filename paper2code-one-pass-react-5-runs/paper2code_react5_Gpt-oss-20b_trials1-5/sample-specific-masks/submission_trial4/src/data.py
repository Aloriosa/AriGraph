import torch
from torchvision import transforms, datasets

from .config import IMAGE_SIZE, NUM_WORKERS

def build_transform(augment: bool = True):
    if augment:
        transform_list = [
            transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    return transforms.Compose(transform_list)

def get_dataloaders(dataset_name: str, batch_size: int = 256):
    train_transform = build_transform(True)
    test_transform = build_transform(False)

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(root=".", train=True, download=True, transform=train_transform)
        test_set  = datasets.CIFAR10(root=".", train=False, download=True, transform=test_transform)
    elif dataset_name == "cifar100":
        train_set = datasets.CIFAR100(root=".", train=True, download=True, transform=train_transform)
        test_set  = datasets.CIFAR100(root=".", train=False, download=True, transform=test_transform)
    elif dataset_name == "svhn":
        # SVHN comes in two variants: 'train' and 'extra'; we use 'train'
        train_set = datasets.SVHN(root=".", split="train", download=True, transform=train_transform)
        test_set  = datasets.SVHN(root=".", split="test", download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, test_loader