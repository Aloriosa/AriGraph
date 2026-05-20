import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset: str,
                    batch_size: int,
                    augment: bool = True):
    """
    Return train and test DataLoaders for the chosen dataset.

    Parameters
    ----------
    dataset : str
        One of 'cifar10', 'cifar100', 'svhn'.
    batch_size : int
        Batch size.
    augment : bool
        Whether to use data augmentation for training.

    Returns
    -------
    train_loader, test_loader, num_classes : tuple
    """
    if dataset == "cifar10":
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
    elif dataset == "cifar100":
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
    elif dataset == "svhn":
        num_classes = 10
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=transform_train
        )
        test_set = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, num_classes