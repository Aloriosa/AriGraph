import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image

class ImageInpaintingDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_type='center', mask_size=64):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_type = mask_type
        self.mask_size = mask_size
        
        # Use CelebA dataset for inpainting
        self.dataset = datasets.CelebA(
            root=root_dir, 
            split='train', 
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor()
            ]),
            download=True
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        
        # Create mask
        if self.mask_type == 'center':
            mask = torch.zeros_like(image)
            h, w = image.shape[1], image.shape[2]
            h_start = (h - self.mask_size) // 2
            w_start = (w - self.mask_size) // 2
            mask[:, h_start:h_start+self.mask_size, w_start:w_start+self.mask_size] = 1
        else:  # random mask
            mask = torch.rand_like(image) > 0.7
            mask = mask.float()
        
        # Create corrupted image
        corrupted_image = image * mask
        
        return corrupted_image, image, mask

class ImageSuperResolutionDataset(Dataset):
    def __init__(self, root_dir, transform=None, scale_factor=4):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        
        # Use DIV2K dataset for super-resolution
        # Note: In practice, you would need to download DIV2K dataset
        # For this reproduction, we'll use CelebA as a placeholder
        self.dataset = datasets.CelebA(
            root=root_dir, 
            split='train', 
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ]),
            download=True
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        
        # Create low-resolution version
        h, w = image.shape[1], image.shape[2]
        low_res = transforms.functional.resize(
            image, 
            (h // self.scale_factor, w // self.scale_factor),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        low_res = transforms.functional.resize(
            low_res, 
            (h, w),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        
        return low_res, image