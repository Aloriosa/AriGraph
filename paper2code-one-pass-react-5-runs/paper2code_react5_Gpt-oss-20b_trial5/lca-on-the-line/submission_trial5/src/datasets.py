"""
Dataset wrappers for ImageNet and its OOD variants.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageNetValDataset(Dataset):
    """
    ImageNet validation set.
    Folder structure:
        data/imagenet/val/<synset_id>/<image_files>
    The class index corresponds to the order of synset IDs in
    src/imagenet_classes.txt.
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        # Build a list of (image_path, synset_id) tuples
        self.samples = []
        for syn_dir in sorted(os.listdir(root)):
            syn_dir_path = os.path.join(root, syn_dir)
            if not os.path.isdir(syn_dir_path):
                continue
            for img_name in sorted(os.listdir(syn_dir_path)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                self.samples.append((os.path.join(syn_dir_path, img_name), syn_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, syn_id = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, syn_id


def get_imagenet_val_loader(batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    dataset = ImageNetValDataset(root='data/imagenet/val', transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader