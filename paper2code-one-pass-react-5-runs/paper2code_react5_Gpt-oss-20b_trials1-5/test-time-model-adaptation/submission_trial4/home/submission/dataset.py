import os
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image


class ImageNetC(Dataset):
    """
    ImageNet-C dataset (corruptions, severity level 5).
    The folder structure expected after extraction:
    imagenet_c/
        images/
            imagenet_c/
                {corruption_type}/
                    {severity}/
                        {class_id}/
                            {image}.jpg
    We flatten all images into a single list.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        base_dir = os.path.join(root, "images", "imagenet_c")
        for corruption in sorted(os.listdir(base_dir)):
            corr_dir = os.path.join(base_dir, corruption)
            if not os.path.isdir(corr_dir):
                continue
            for severity in sorted(os.listdir(corr_dir)):
                severity_dir = os.path.join(corr_dir, severity)
                if not os.path.isdir(severity_dir):
                    continue
                for cls in sorted(os.listdir(severity_dir)):
                    cls_dir = os.path.join(severity_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    for img_name in sorted(os.listdir(cls_dir)):
                        img_path = os.path.join(cls_dir, img_name)
                        self.samples.append((img_path, int(cls)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageNetVal(Dataset):
    """
    Simple ImageNet‑1K validation loader from the Imagenette2‑320 dataset
    (used only for source statistics). The folder structure is:
    imagenet_val/
        imagenette2-320/
            val/
                {class_id}/
                    {image}.jpg
    """
    def __init__(self, root, transform=None):
        val_dir = os.path.join(root, "imagenette2-320", "val")
        self.dataset = ImageFolder(val_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]