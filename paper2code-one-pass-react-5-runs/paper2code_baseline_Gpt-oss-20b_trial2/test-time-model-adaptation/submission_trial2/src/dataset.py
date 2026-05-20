import os
from torchvision import datasets, transforms


class ImageNetC(datasets.ImageFolder):
    """
    ImageNet‑C dataset loader.
    The official repo stores images in a class‑wise folder hierarchy.
    """

    def __init__(self, root: str, level: int = 5):
        """
        Parameters
        ----------
        root : str
            Path to the folder containing subfolders for each class.
        level : int
            Severity level (1–5). 5 is the most severe.
        """
        # The original repo already contains only severity 5 images.
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        super().__init__(root, transform=transform)