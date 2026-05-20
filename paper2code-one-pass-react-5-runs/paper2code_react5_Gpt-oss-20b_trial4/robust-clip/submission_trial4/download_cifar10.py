import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

def download_cifar10(root="data/cifar10"):
    os.makedirs(root, exist_ok=True)
    datasets.CIFAR10(root=root, train=True, download=True)
    datasets.CIFAR10(root=root, train=False, download=True)
    print("CIFAR‑10 downloaded to", root)

if __name__ == "__main__":
    download_cifar10()