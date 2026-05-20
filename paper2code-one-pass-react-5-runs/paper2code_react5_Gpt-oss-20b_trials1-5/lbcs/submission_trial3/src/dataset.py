import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple

def load_mnist(batch_size: int = 128,
               download: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download MNIST and return train/test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root="./data",
                                          train=True,
                                          download=download,
                                          transform=transform)
    test_set  = torchvision.datasets.MNIST(root="./data",
                                          train=False,
                                          download=download,
                                          transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_set